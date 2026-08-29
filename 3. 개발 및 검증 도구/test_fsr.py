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
    print(f"  해제 {len(s.released)}회 (기대 {expect}) -> {ok}\n")

run("두 FSR 동시 놓음 (0.05초 시간차)", [1.00, 1.05], 0)
run("진짜 더블탭 (0.5초 간격)", [1.00, 1.50], 1)
run("너무 느린 두 번 (2초 간격)", [1.00, 3.00], 0)
run("동시놓음 노이즈 후 진짜 더블탭", [1.00, 1.04, 1.60], 1)
run("경계값: 정확히 0.3초", [1.00, 1.30], 1)
run("경계값: 0.29초 (직전)", [1.00, 1.29], 0)
run("경계값: 정확히 1.0초", [1.00, 2.00], 1)

# 양손 그립 반복
s = Sim()
t = 1.0
for i in range(6):
    s.tap(t); s.tap(t + 0.05)
    t += 2.0
print("=== 양손 그립 6회 반복 (매번 0.05초 시간차) ===")
print(f"  해제 {len(s.released)}회 (기대 0) -> {'OK' if not s.released else 'FAIL'}")
