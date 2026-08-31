"""
Shared BLE serial link for the HM-10 master <-> Pro Mini (HM-10 slave) pair.

Wiring
  Raspberry Pi GPIO 8/10 (TX/RX) -- wired directly -- HM-10 master
  HM-10 master <---- BLE (wireless) ----> HM-10 slave -- Pro Mini

  So there is exactly ONE UART link on the Pi side (/dev/serial0), and it
  carries traffic in both directions over the same wire:
    Pi  -> Pro Mini : vibration state command ('0'-'3')
    Pro Mini -> Pi  : IR sensor value ("IR value: NNN")

Two classes opening /dev/serial0 independently would fight over the same
port, so AlertSystem owns the connection and this class reads incoming IR
lines from it. AlertSystem still does its own writes for vibration commands;
this class only adds a background reader on top of the same serial.Serial
object, guarded by a lock so reads and writes never interleave mid-line.
"""

import threading


class SharedBLELink:
    """
    Reads incoming IR lines on top of the serial.Serial connection that
    AlertSystem already owns, guarded by AlertSystem's ble_serial_lock so
    reads and writes never interleave mid-line on the shared wire.
    """

    def __init__(self, alert_system):
        self.ble_serial = alert_system.ble_serial
        self._lock = alert_system.ble_serial_lock
        self._latest_ir = None
        self._stop_flag = False

        self._thread = None
        if self.ble_serial is not None:
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        else:
            print("SharedBLELink: no serial connection, IR will stay unavailable.")

    def _read_loop(self):
        while not self._stop_flag:
            try:
                # readline() blocks up to the serial timeout, which is fine
                # here: writes (vibration commands) are short and infrequent,
                # so brief lock contention does not affect responsiveness.
                with self._lock:
                    line = self.ble_serial.readline()
                if line:
                    text = line.decode("utf-8", errors="ignore").strip()
                    if text:
                        self._latest_ir = text
            except Exception:
                break

    def get_latest_ir(self):
        return self._latest_ir

    def stop(self):
        self._stop_flag = True
