from gpiozero import LED, Buzzer, Button
import time
import threading
import serial

# ===== GPIO Pin Assignments (BCM numbering) =====
PIN_LED_RED = 17
PIN_LED_YELLOW = 27
PIN_LED_GREEN = 22
PIN_BUZZER = 23
PIN_BTN_POWER = 5    # tact switch 1: ON/OFF toggle, every ON starts a new calibration
PIN_BTN_DISMISS = 6  # tact switch 2: the only way to clear a level-3 (asleep) alarm
PIN_FSR1 = 26        # force-sensitive resistor 1 (voltage divider -> digital HIGH when pressed)
PIN_FSR2 = 19        # force-sensitive resistor 2 (voltage divider -> digital HIGH when pressed)

# ===== Double-tap settings =====
# Both pads are gripped and released together and never fire at the same instant,
# so a release inside the grace window is absorbed into the same tap. A genuine
# second tap must land after the grace window but within DOUBLE_TAP_WINDOW.
SIMULTANEOUS_RELEASE_GRACE = 0.3   # releases within this gap count as ONE tap
DOUBLE_TAP_WINDOW = 1.0            # two confirmed taps within this window = double tap
DOUBLE_TAP_SUPPRESS_SEC = 3.0      # output muted this long after a double tap

# ===== Buzzer pattern settings (seconds) =====
# Each pattern is (on_time, off_time). "None" off_time means continuous ON.
BUZZER_PATTERNS = {
    1: (0.10, 2.90),  # state 1 (inattention): small chirp on a 3 s cycle
    2: (0.30, 0.60),  # state 2 (drowsy): medium beep, medium gap
    3: (None, None),  # state 3 (asleep): continuous ON
}

CALIBRATION_BLINK_INTERVAL = 0.3  # yellow LED blink speed while calibrating
RED_BLINK_SLOW = 0.5              # state 1 red blink
RED_BLINK_FAST = 0.3              # state 2 red blink

# ===== BLE (master HM-10 via GPIO UART) settings for vibration commands to Pro Mini =====
BLE_SERIAL_PORT = "/dev/serial0"
BLE_BAUD_RATE = 9600


class AlertSystem:
    """
    Output rules
      Yellow LED : blinks while calibrating, off once calibration ends,
                   solid whenever the IR sensor is judged unreliable.
      Red LED    : slow blink state 1, fast blink state 2, solid state 3.
      Green LED  : on only at state 0.
      Buzzer     : per BUZZER_PATTERNS above.
      Level 3    : latches. Only tact 2 clears it.
      Double tap : clears level 1 and 2 and mutes output for 3 s. Detection
                   upstream keeps running; only the output is held back.
      Tact 1     : ON/OFF toggle. Every ON requests a fresh calibration.
    """

    def __init__(self):
        self.led_red = LED(PIN_LED_RED)
        self.led_yellow = LED(PIN_LED_YELLOW)
        self.led_green = LED(PIN_LED_GREEN)
        self.buzzer = Buzzer(PIN_BUZZER)

        # bounce_time = simple debounce built into gpiozero
        self.btn_power = Button(PIN_BTN_POWER, pull_up=True, bounce_time=0.05)
        self.btn_dismiss = Button(PIN_BTN_DISMISS, pull_up=True, bounce_time=0.05)

        # FSR pads wired as voltage dividers (FSR on the 3V3 side, 10k to GND),
        # so the pin reads HIGH while pressed.
        self.fsr1 = Button(PIN_FSR1, pull_up=False, bounce_time=0.05)
        self.fsr2 = Button(PIN_FSR2, pull_up=False, bounce_time=0.05)

        self.system_on = False
        self.is_calibrating = False
        self.ir_reliable = True

        self._calib_requested = False   # main loop consumes this to run calibration
        self._level3_latched = False    # level 3 holds until tact 2
        self._suppress_until = 0.0      # double-tap mute window

        self._lock = threading.Lock()

        self._buzzer_thread = None
        self._buzzer_stop_flag = threading.Event()
        self._buzzer_level = None

        self._calib_stop_flag = threading.Event()
        self._calib_blink_thread = None

        self._red_blink_thread = None
        self._red_blink_stop_flag = threading.Event()
        self._red_blink_interval = None

        # ----- Double-tap detection state -----
        self._last_tap_time = None      # None = no pending first tap
        self._pending_release_timer = None
        self._release_pending = False

        # Wire up button callbacks (gpiozero calls these automatically)
        self.btn_power.when_pressed = self._on_power_button_pressed
        self.btn_dismiss.when_pressed = self._on_dismiss_button_pressed
        # when_released, not when_pressed: a tap only counts once the hand lifts,
        # so holding a pad down cannot keep re-triggering.
        self.fsr1.when_released = self._on_fsr_released
        self.fsr2.when_released = self._on_fsr_released

        # ----- BLE serial connection to master HM-10 (bridges to Pro Mini) -----
        self.ble_serial = None
        try:
            self.ble_serial = serial.Serial(BLE_SERIAL_PORT, BLE_BAUD_RATE, timeout=1)
            print(f"BLE serial connected on {BLE_SERIAL_PORT}")
        except serial.SerialException as e:
            print(f"Could not open BLE serial port: {e}")
            print("Continuing without vibration output (BLE commands will not be sent).")

        self._last_sent_state = None
        self._stop_all_outputs()

    # ---------- Button handling ----------
    def _on_power_button_pressed(self):
        with self._lock:
            self.system_on = not self.system_on

        if self.system_on:
            print("System turned ON. Calibration will start.")
            self._level3_latched = False
            self._suppress_until = 0.0
            self._last_tap_time = None
            self._calib_requested = True
        else:
            print("System turned OFF.")
            self.is_calibrating = False
            self._calib_stop_flag.set()
            self._level3_latched = False
            self._stop_all_outputs()

    def _on_dismiss_button_pressed(self):
        """Tact 2 exists solely to release a latched level-3 alarm."""
        if not self.system_on:
            return
        if self._level3_latched:
            self._level3_latched = False
            self._suppress_until = time.time() + DOUBLE_TAP_SUPPRESS_SEC
            self._stop_buzzer()
            self._stop_red_blink()
            self.led_red.off()
            self._send_vibration_command(0)
            print("Level-3 alarm released by tact 2.")
        else:
            print("Tact 2 pressed (no level-3 alarm active).")

    # ---------- FSR double-tap handling ----------
    def _on_fsr_released(self):
        """
        Either pad releasing lands here. The tap is not confirmed immediately:
        a grace timer runs first, and if the other pad releases before it fires,
        both releases resolve to a single tap. Without this, the small skew
        between the two pads alone would look like a double tap.
        """
        if not self.system_on:
            return

        if self._release_pending:
            # The other hand lifting inside the grace window - same tap.
            return

        self._release_pending = True
        self._pending_release_timer = threading.Timer(
            SIMULTANEOUS_RELEASE_GRACE, self._confirm_tap
        )
        self._pending_release_timer.start()

    def _confirm_tap(self):
        """One confirmed tap, once the grace window passed with no further release."""
        self._release_pending = False
        now = time.time()

        if self._last_tap_time is None:
            self._last_tap_time = now
            return

        gap = now - self._last_tap_time
        if gap <= DOUBLE_TAP_WINDOW:
            self._last_tap_time = None      # consume the pair
            if self._level3_latched:
                print("Double tap ignored - level 3 needs tact 2.")
            else:
                self._suppress_until = now + DOUBLE_TAP_SUPPRESS_SEC
                self._stop_buzzer()
                self._stop_red_blink()
                self.led_red.off()
                self.led_green.on()
                self._send_vibration_command(0)
                print(f"Double tap - output muted for {DOUBLE_TAP_SUPPRESS_SEC:.0f} s.")
        else:
            # Too late to pair; treat this as a new first tap.
            self._last_tap_time = now

    # ---------- Calibration ----------
    def begin_calibration(self):
        """Main loop calls this when it starts a calibration run."""
        self.is_calibrating = True
        self._level3_latched = False
        self._suppress_until = 0.0
        self._stop_all_outputs()

        self._calib_stop_flag.clear()
        self._calib_blink_thread = threading.Thread(
            target=self._blink_yellow_while_calibrating, daemon=True)
        self._calib_blink_thread.start()

    def end_calibration(self):
        """Main loop calls this when calibration finishes."""
        self.is_calibrating = False
        self._calib_stop_flag.set()
        if self._calib_blink_thread is not None:
            self._calib_blink_thread.join(timeout=1.0)
        # Yellow goes out unless the IR sensor is untrustworthy.
        if self.ir_reliable:
            self.led_yellow.off()
        else:
            self.led_yellow.on()
        print("Calibration complete.")

    def consume_calibration_request(self):
        """True once per power-on, so the main loop knows to run calibration."""
        if self._calib_requested:
            self._calib_requested = False
            return True
        return False

    def request_recalibration(self):
        """Ask for another calibration pass (e.g. the quality check failed)."""
        self._calib_requested = True

    def _blink_yellow_while_calibrating(self):
        while not self._calib_stop_flag.is_set():
            self.led_yellow.on()
            if self._calib_stop_flag.wait(timeout=CALIBRATION_BLINK_INTERVAL):
                break
            self.led_yellow.off()
            if self._calib_stop_flag.wait(timeout=CALIBRATION_BLINK_INTERVAL):
                break
        self.led_yellow.off()

    # ---------- BLE vibration command to Pro Mini ----------
    def _send_vibration_command(self, state):
        """Send '0'/'1'/'2'/'3' to the Pro Mini via the master HM-10 (GPIO UART)."""
        if self.ble_serial is None or state == self._last_sent_state:
            return
        try:
            self.ble_serial.write(str(state).encode())
            self._last_sent_state = state
        except serial.SerialException as e:
            print(f"BLE send failed: {e}")

    # ---------- Buzzer ----------
    def _stop_buzzer(self):
        self._buzzer_stop_flag.set()
        if self._buzzer_thread is not None:
            self._buzzer_thread.join(timeout=1.0)
        self._buzzer_thread = None
        self._buzzer_level = None
        self.buzzer.off()

    def _start_buzzer_pattern(self, state):
        # Already running this pattern: leave the cycle alone, otherwise the
        # 3 s gap of state 1 would restart on every frame and never sound.
        if (self._buzzer_level == state
                and self._buzzer_thread is not None
                and self._buzzer_thread.is_alive()):
            return
        self._stop_buzzer()
        self._buzzer_stop_flag.clear()
        self._buzzer_level = state
        self._buzzer_thread = threading.Thread(
            target=self._buzzer_loop, args=(state,), daemon=True)
        self._buzzer_thread.start()

    def _buzzer_loop(self, state):
        on_time, off_time = BUZZER_PATTERNS[state]

        if off_time is None:
            self.buzzer.on()
            while not self._buzzer_stop_flag.is_set():
                time.sleep(0.1)
            self.buzzer.off()
            return

        while not self._buzzer_stop_flag.is_set():
            self.buzzer.on()
            if self._buzzer_stop_flag.wait(timeout=on_time):
                break
            self.buzzer.off()
            if self._buzzer_stop_flag.wait(timeout=off_time):
                break
        self.buzzer.off()

    # ---------- Red LED ----------
    def _start_red_blink(self, interval):
        if (self._red_blink_interval == interval
                and self._red_blink_thread is not None
                and self._red_blink_thread.is_alive()):
            return
        self._stop_red_blink()
        self._red_blink_interval = interval
        self._red_blink_stop_flag.clear()
        self._red_blink_thread = threading.Thread(target=self._red_blink_loop, daemon=True)
        self._red_blink_thread.start()

    def _red_blink_loop(self):
        interval = self._red_blink_interval
        while not self._red_blink_stop_flag.is_set():
            self.led_red.on()
            if self._red_blink_stop_flag.wait(timeout=interval):
                break
            self.led_red.off()
            if self._red_blink_stop_flag.wait(timeout=interval):
                break
        self.led_red.off()

    def _stop_red_blink(self):
        self._red_blink_stop_flag.set()
        if self._red_blink_thread is not None:
            self._red_blink_thread.join(timeout=1.0)
        self._red_blink_thread = None
        self._red_blink_interval = None

    # ---------- Main output update ----------
    def update_output(self, state, ir_reliable=True):
        """
        state: 0 (normal), 1 (inattention), 2 (drowsy), 3 (asleep)
        ir_reliable: False keeps the yellow LED solid
        """
        self.ir_reliable = ir_reliable

        if not self.system_on:
            return
        if self.is_calibrating:
            # The yellow blink already communicates what is happening.
            return

        # Yellow LED: solid while the IR sensor cannot be trusted.
        if ir_reliable:
            self.led_yellow.off()
        else:
            self.led_yellow.on()

        # Level 3 latches until tact 2 releases it.
        if state >= 3:
            self._level3_latched = True
        if self._level3_latched:
            self._stop_red_blink()
            self.led_red.on()
            self.led_green.off()
            self._start_buzzer_pattern(3)
            self._send_vibration_command(3)
            return

        # Double tap mutes level 1 and 2 briefly. Detection upstream continues.
        if time.time() < self._suppress_until and state in (1, 2):
            self._stop_buzzer()
            self._stop_red_blink()
            self.led_red.off()
            self.led_green.on()
            self._send_vibration_command(0)
            return

        if state == 0:
            self._stop_buzzer()
            self._stop_red_blink()
            self.led_red.off()
            self.led_green.on()
            self._send_vibration_command(0)
        elif state == 1:
            self._start_red_blink(RED_BLINK_SLOW)
            self.led_green.off()
            self._start_buzzer_pattern(1)
            self._send_vibration_command(1)
        elif state == 2:
            self._start_red_blink(RED_BLINK_FAST)
            self.led_green.off()
            self._start_buzzer_pattern(2)
            self._send_vibration_command(2)

    # ---------- Shutdown ----------
    def _stop_all_outputs(self):
        self._stop_buzzer()
        self._stop_red_blink()
        self.led_red.off()
        self.led_yellow.off()
        self.led_green.off()
        self._send_vibration_command(0)

    def cleanup(self):
        self._calib_stop_flag.set()
        self._stop_all_outputs()
        if self._pending_release_timer is not None:
            self._pending_release_timer.cancel()
        self.led_red.close()
        self.led_yellow.close()
        self.led_green.close()
        self.buzzer.close()
        self.btn_power.close()
        self.btn_dismiss.close()
        self.fsr1.close()
        self.fsr2.close()
        if self.ble_serial is not None:
            self.ble_serial.close()


# ===== Example usage / test loop =====
if __name__ == "__main__":
    alert = AlertSystem()

    print("Press tact 1 to turn the system ON.")
    print("This demo fakes a 3 s calibration, then cycles states 0 -> 1 -> 2 -> 3.")
    print("Double tap an FSR to mute levels 1-2. Press tact 2 to clear level 3.")

    demo_states = [0, 1, 2, 3, 0]
    demo_index = 0
    last_demo_change = time.time()
    DEMO_STATE_INTERVAL = 5.0

    calib_start = None
    calib_done = False

    try:
        while True:
            if alert.consume_calibration_request():
                alert.begin_calibration()
                calib_start = time.time()
                calib_done = False

            if alert.is_calibrating and calib_start and time.time() - calib_start >= 3.0:
                alert.end_calibration()
                calib_done = True

            if alert.system_on and calib_done:
                if time.time() - last_demo_change >= DEMO_STATE_INTERVAL:
                    last_demo_change = time.time()
                    demo_index = (demo_index + 1) % len(demo_states)
                    print(f"[DEMO] Setting state = {demo_states[demo_index]}")
                alert.update_output(state=demo_states[demo_index], ir_reliable=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        alert.cleanup()
