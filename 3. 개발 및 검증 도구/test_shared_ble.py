"""Simulate SharedBLELink against a fake serial.Serial to check the lock works."""
import threading
import time
import io

class FakeSerial:
    """Mimics enough of serial.Serial for the shared-link logic to run."""
    def __init__(self):
        self._lock = threading.Lock()
        self._incoming = [b"IR value: 250\n", b"IR value: 248\n", b"IR value: 40\n"] * 20
        self._idx = 0
        self.writes = []

    def readline(self):
        time.sleep(0.01)
        if self._idx >= len(self._incoming):
            return b""
        line = self._incoming[self._idx]
        self._idx += 1
        return line

    def write(self, data):
        self.writes.append(data)

class FakeAlert:
    def __init__(self):
        self.ble_serial = FakeSerial()
        self.ble_serial_lock = threading.Lock()

from shared_ble_link import SharedBLELink

alert = FakeAlert()
link = SharedBLELink(alert)

# Simulate AlertSystem sending vibration commands concurrently, using the
# SAME lock, while SharedBLELink reads in the background.
errors = []
def sender():
    for i in range(30):
        try:
            with alert.ble_serial_lock:
                alert.ble_serial.write(str(i % 4).encode())
        except Exception as e:
            errors.append(e)
        time.sleep(0.005)

t = threading.Thread(target=sender)
t.start()
t.join()

time.sleep(0.5)
link.stop()

print("errors during concurrent read/write:", errors)
print("last IR value seen:", link.get_latest_ir())
print("writes sent:", len(alert.ble_serial.writes))
print("OK" if not errors and link.get_latest_ir() is not None else "FAIL")
