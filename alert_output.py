from gpiozero import LED, Buzzer, Button
import time
import threading
import serial

# ===== GPIO Pin Assignments (BCM numbering) =====
PIN_LED_RED = 17
PIN_LED_YELLOW = 27
PIN_LED_GREEN = 22
PIN_BUZZER = 23
PIN_BTN_POWER = 5    # tact switch 1: ON/OFF toggle + triggers recalibration on ON
PIN_BTN_DISMISS = 6  # tact switch 2: dismiss/silence current alert
PIN_FSR1 = 26         # force-sensitive resistor 1 (voltage divider -> digital HIGH/LOW)
PIN_FSR2 = 19         # force-sensitive resistor 2 (voltage divider -> digital HIGH/LOW)

# ===== Double-tap settings =====
DOUBLE_TAP_WINDOW = 0.5   # seconds: two confirmed taps within this window = double tap
SIMULTANEOUS_RELEASE_GRACE = 0.15  # seconds: releases from FSR1/FSR2 within this gap count as ONE tap

# ===== Buzzer pattern settings (seconds) =====
# Each pattern is (on_time, off_time). "None" off_time means continuous ON.
BUZZER_PATTERNS = {
    1: (0.15, 1.5),   # state 1 (mild): short beep, long gap
    2: (0.3, 0.6),    # state 2 (drowsy): medium beep, medium gap
    3: (None, None),  # state 3 (asleep): continuous ON
}

CALIBRATION_BLINK_INTERVAL = 0.3  # yellow LED blink speed while calibrating
CALIBRATION_DURATION = 3.0        # seconds the (placeholder) calibration routine takes

# ===== BLE (master HM-10 via GPIO UART) settings for vibration commands to Pro Mini =====
BLE_SERIAL_PORT = "/dev/serial0"
BLE_BAUD_RATE = 9600


class AlertSystem:
    def __init__(self):
        self.led_red = LED(PIN_LED_RED)
        self.led_yellow = LED(PIN_LED_YELLOW)
        self.led_green = LED(PIN_LED_GREEN)
        self.buzzer = Buzzer(PIN_BUZZER)

        # bounce_time = simple debounce built into gpiozero
        self.btn_power = Button(PIN_BTN_POWER, pull_up=True, bounce_time=0.05)
        self.btn_dismiss = Button(PIN_BTN_DISMISS, pull_up=True, bounce_time=0.05)

        # FSR pads wired as individual voltage dividers -> digital HIGH when pressed.
        self.fsr1 = Button(PIN_FSR1, pull_up=False, bounce_time=0.05)
        self.fsr2 = Button(PIN_FSR2, pull_up=False, bounce_time=0.05)

        self.system_on = False
        self.needs_recalibration = True  # requires calibration before first use
        self.is_calibrating = False
        self.alert_dismissed = False

        self._lock = threading.Lock()

        self._buzzer_thread = None
        self._buzzer_stop_flag = threading.Event()

        self._calib_thread = None
        self._calib_stop_flag = threading.Event()

        self._red_blink_thread = None
        self._red_blink_stop_flag = threading.Event()

        # ----- Double-tap detection state -----
        self._last_tap_time = 0.0

        # ----- Simultaneous-release grace window state -----
        # When one FSR releases, we wait a short grace period before counting it as
        # a confirmed tap. If the *other* FSR also releases within that window
        # (both hands lifting at slightly different instants), it still counts as
        # only ONE tap, not two.
        self._pending_release_timer = None
        self._release_pending = False

        # Wire up button callbacks (gpiozero calls these automatically on press)
        self.btn_power.when_pressed = self._on_power_button_pressed
        self.btn_dismiss.when_pressed = self._on_dismiss_button_pressed
        # Use when_released (not when_pressed) so a tap only counts once the finger
        # is fully lifted -- holding an FSR down must NOT keep re-triggering.
        self.fsr1.when_released = self._on_fsr_released
        self.fsr2.when_released = self._on_fsr_released

        # ----- BLE serial connection to master HM-10 (bridges to Pro Mini for vibration) -----
        self.ble_serial = None
        try:
            self.ble_serial = serial.Serial(BLE_SERIAL_PORT, BLE_BAUD_RATE, timeout=1)
            print(f"BLE serial connected on {BLE_SERIAL_PORT}")
        except serial.SerialException as e:
            print(f"Could not open BLE serial port: {e}")
            print("Continuing without vibration output (BLE commands will not be sent).")

        self._last_sent_state = None

        # Make sure everything starts off
        self._stop_all_outputs()

    # ---------- Button handling ----------
    def _on_power_button_pressed(self):
        print(f"[DEBUG] Power button callback fired. Before toggle: system_on={self.system_on}")
        with self._lock:
            self.system_on = not self.system_on
        print(f"[DEBUG] After toggle: system_on={self.system_on}")

        if self.system_on:
            print("System turned ON. Starting calibration...")
            self._start_calibration()
        else:
            print("System turned OFF.")
            self._stop_all_outputs()

    def _on_dismiss_button_pressed(self):
        print(f"[DEBUG] Dismiss button callback fired. system_on={self.system_on}")
        if self.system_on:
            print("Alert dismissed by user.")
            with self._lock:
                self.alert_dismissed = True
            self._stop_buzzer()
            self.led_red.off()
            self._send_vibration_command(0)  # tell Pro Mini to stop vibrating immediately

    # ---------- FSR double-tap handling ----------
    def _on_fsr_released(self):
        # Either FSR releasing lands here. Instead of confirming the tap immediately,
        # start (or restart) a short grace-period timer. If the OTHER FSR also
        # releases before that timer fires, we just let the existing timer run --
        # so two near-simultaneous releases still resolve to a single confirmed tap.
        print(f"[DEBUG] FSR released (raw event). system_on={self.system_on}")
        if not self.system_on:
            return

        if self._release_pending:
            # A release is already being debounced -- this is the "other hand"
            # lifting off within the grace window. Don't start a second timer.
            print("[DEBUG] Second near-simultaneous release absorbed into the same tap.")
            return

        self._release_pending = True
        self._pending_release_timer = threading.Timer(
            SIMULTANEOUS_RELEASE_GRACE, self._confirm_tap
        )
        self._pending_release_timer.start()

    def _confirm_tap(self):
        """Called once the grace window has passed with no further FSR release -- this is one confirmed tap."""
        self._release_pending = False

        now = time.time()
        gap = now - self._last_tap_time
        print(f"[DEBUG] Tap confirmed. Gap since last confirmed tap: {gap:.3f}s (window={DOUBLE_TAP_WINDOW}s)")
        if self._last_tap_time > 0 and gap <= DOUBLE_TAP_WINDOW:
            # Second confirmed tap arrived within the window -> double tap detected
            print("Double tap detected. Resetting to state 0 (normal).")
            self._last_tap_time = 0.0  # reset so a third quick tap doesn't chain-trigger
            self._reset_to_normal()
        else:
            # First confirmed tap of a possible pair -> record the time and wait
            self._last_tap_time = now

    def _reset_to_normal(self):
        """Force the alert state back to 0 (normal), as if a fresh 'normal' reading came in."""
        with self._lock:
            self.alert_dismissed = False  # clear any prior dismissal so state 0 renders cleanly
        self.update_output(state=0, ir_reliable=True)

    # ---------- Calibration ----------
    def _start_calibration(self):
        self.is_calibrating = True
        self._calib_stop_flag.clear()
        self._calib_thread = threading.Thread(target=self._calibration_routine, daemon=True)
        self._calib_thread.start()

    def _calibration_routine(self):
        blink_thread = threading.Thread(target=self._blink_yellow_while_calibrating, daemon=True)
        blink_thread.start()

        # ----- Placeholder: replace this with the actual calibration procedure -----
        time.sleep(CALIBRATION_DURATION)
        # -----------------------------------------------------------------------------

        self.is_calibrating = False
        self._calib_stop_flag.set()
        blink_thread.join(timeout=1.0)

        with self._lock:
            self.needs_recalibration = False

        self.led_yellow.off()
        print("Calibration complete.")

    def _blink_yellow_while_calibrating(self):
        while not self._calib_stop_flag.is_set():
            self.led_yellow.on()
            if self._calib_stop_flag.wait(timeout=CALIBRATION_BLINK_INTERVAL):
                break
            self.led_yellow.off()
            if self._calib_stop_flag.wait(timeout=CALIBRATION_BLINK_INTERVAL):
                break

    def request_recalibration(self):
        """Call this if upstream logic (e.g. IR unreliable) decides recalibration is needed."""
        with self._lock:
            self.needs_recalibration = True
        if not self.is_calibrating:
            self.led_yellow.on()  # solid yellow = recalibration needed

    # ---------- BLE vibration command to Pro Mini ----------
    def _send_vibration_command(self, state):
        """Send '0'/'1'/'2'/'3' to the Pro Mini via the master HM-10 (GPIO UART)."""
        if self.ble_serial is None:
            return
        if state == self._last_sent_state:
            return  # avoid spamming the same command every loop
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
        self.buzzer.off()

    def _start_buzzer_pattern(self, state):
        self._stop_buzzer()
        self._buzzer_stop_flag.clear()
        self._buzzer_thread = threading.Thread(
            target=self._buzzer_loop, args=(state,), daemon=True
        )
        self._buzzer_thread.start()

    def _buzzer_loop(self, state):
        on_time, off_time = BUZZER_PATTERNS[state]

        if off_time is None:
            # Continuous ON (state 3)
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

    # ---------- Red LED blinking for state 2 ----------
    def _blink_red_if_needed(self):
        if self._red_blink_thread is not None and self._red_blink_thread.is_alive():
            return  # already blinking
        self._red_blink_stop_flag.clear()
        self._red_blink_thread = threading.Thread(target=self._red_blink_loop, daemon=True)
        self._red_blink_thread.start()

    def _red_blink_loop(self):
        while not self._red_blink_stop_flag.is_set():
            self.led_red.on()
            if self._red_blink_stop_flag.wait(timeout=0.3):
                break
            self.led_red.off()
            if self._red_blink_stop_flag.wait(timeout=0.3):
                break

    def _stop_red_blink(self):
        self._red_blink_stop_flag.set()
        if self._red_blink_thread is not None:
            self._red_blink_thread.join(timeout=1.0)

    # ---------- Main output update ----------
    def update_output(self, state, ir_reliable):
        """
        state: 0 (normal), 1 (mild inattention), 2 (drowsy), 3 (asleep)
        ir_reliable: True (o) or False (x)
        """
        if not self.system_on:
            return

        if self.is_calibrating:
            # While calibrating, ignore state updates (yellow blink already handles feedback)
            return

        if not ir_reliable:
            self.request_recalibration()

        if state > 0:
            self.alert_dismissed = False

        # ----- Red LED -----
        if state == 2:
            self._blink_red_if_needed()
        elif state == 3:
            self._stop_red_blink()
            self.led_red.on()
        else:
            self._stop_red_blink()
            self.led_red.off()

        # ----- Green LED: ON only when state is 0 (normal) -----
        if state == 0:
            self.led_green.on()
        else:
            self.led_green.off()

        # ----- Buzzer -----
        if self.alert_dismissed:
            self._stop_buzzer()
        elif state in (1, 2, 3):
            self._start_buzzer_pattern(state)
        else:
            self._stop_buzzer()

        # ----- Vibration (BLE to Pro Mini) -----
        if self.alert_dismissed:
            self._send_vibration_command(0)
        else:
            self._send_vibration_command(state)

    # ---------- Shutdown ----------
    def _stop_all_outputs(self):
        self._stop_buzzer()
        self._stop_red_blink()
        self.led_red.off()
        self.led_yellow.off()
        self.led_green.off()
        self._send_vibration_command(0)

    def cleanup(self):
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

    print("Press the power button (tact 1) to turn the system ON.")
    print("Once ON, this demo will cycle through states 0->1->2->3 automatically for testing.")
    print("Press the dismiss button (tact 2) to silence an alert. Ctrl+C to exit.")

    demo_states = [0, 1, 2, 3, 0]
    demo_index = 0
    last_demo_change = time.time()
    DEMO_STATE_INTERVAL = 5.0  # change demo state every 5 seconds, for testing only

    try:
        while True:
            if alert.system_on and not alert.is_calibrating:
                if time.time() - last_demo_change >= DEMO_STATE_INTERVAL:
                    last_demo_change = time.time()
                    demo_index = (demo_index + 1) % len(demo_states)
                    state = demo_states[demo_index]
                    print(f"[DEMO] Setting state = {state}")
                    alert.update_output(state=state, ir_reliable=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        alert.cleanup()
