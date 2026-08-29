from gpiozero import Button
import time

fsr = Button(26, pull_up=False, bounce_time=0.05)

print("Press Ctrl+C to stop. Watching GPIO26 state...")

try:
    while True:
        print(f"FSR is_pressed = {fsr.is_pressed}")
        time.sleep(0.3)
except KeyboardInterrupt:
    pass