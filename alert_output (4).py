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

        # Wire up button callbacks (gpiozero calls these automatically on press)
        self.btn_power.when_pressed = self._on_power_button_pressed
        self.btn_dismiss.when_pressed = self._on_dismiss_button_pressed

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
        self.led_red.close()
        self.led_yellow.close()
        self.led_green.close()
        self.buzzer.close()
        self.btn_power.close()
        self.btn_dismiss.close()
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
