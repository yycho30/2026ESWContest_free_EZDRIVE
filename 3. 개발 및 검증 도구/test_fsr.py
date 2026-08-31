"""FSR double-tap logic check (no hardware needed)"""

FSR_DOUBLE_TAP_MIN_GAP = 0.3
FSR_DOUBLE_TAP_MAX_GAP = 1.0


class Sim:
    def __init__(self):
        self.last = None
        self.released = []

    def tap(self, now):
        if self.last is None:
            self.last = now
            return "first tap"
        gap = now - self.last
        if gap < FSR_DOUBLE_TAP_MIN_GAP:
            return f"ignored (gap {gap:.2f}s - same grip)"
        if gap <= FSR_DOUBLE_TAP_MAX_GAP:
            self.released.append(round(now, 2))
            self.last = None
            return f"*** DOUBLE TAP (gap {gap:.2f}s) - released ***"
        self.last = now
        return f"too late (gap {gap:.2f}s) - new first tap"


def run(title, taps, expect):
    s = Sim()
    print(f"=== {title} ===")
    for t in taps:
        print(f"  t={t:.2f}  {s.tap(t)}")
    ok = "OK" if len(s.released) == expect else "FAIL"
    print(f"  released {len(s.released)}x (expected {expect}) -> {ok}\n")


run("both FSRs released 0.05s apart", [1.00, 1.05], 0)
run("real double tap (0.5s apart)", [1.00, 1.50], 1)
run("too slow (2s apart)", [1.00, 3.00], 0)
run("simultaneous-release noise then a real double tap", [1.00, 1.04, 1.60], 1)
run("boundary: exactly 0.3s", [1.00, 1.30], 1)
run("boundary: 0.29s (just under)", [1.00, 1.29], 0)
run("boundary: exactly 1.0s", [1.00, 2.00], 1)

# Repeated two-handed grip
s = Sim()
t = 1.0
for i in range(6):
    s.tap(t)
    s.tap(t + 0.05)
    t += 2.0
print("=== two-handed grip repeated 6x (0.05s skew each time) ===")
print(f"  released {len(s.released)}x (expected 0) -> {'OK' if not s.released else 'FAIL'}")
