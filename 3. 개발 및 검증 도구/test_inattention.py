"""Forward Inattention Engine Scenario Validation"""
import inattention_engine as ie
from inattention_engine import InattentionDetector, bbox_to_angle, bbox_area_ratio, risk_level

W, H = 1280, 720
def box(cx_ratio, area_ratio):
    """Create bbox using screen width position ratio and area ratio"""
    import math
    a = area_ratio * W * H
    w = math.sqrt(a * 1.0); h = a / w
    cx = cx_ratio * W
    return (int(cx-w/2), int(H/2-h/2), int(cx+w/2), int(H/2+h/2))

print("=== Angle Conversion Check ===")
for r in [0.0, 0.25, 0.5, 0.75, 1.0]:
    b = box(r, 0.05)
    print(f"  Screen position {r*100:3.0f}% -> {bbox_to_angle(b[0], b[2]):+6.1f} deg")

print("\n=== Risk Level Check ===")
for cls, ar in [("person",0.005),("person",0.02),("person",0.07),("car",0.02),("car",0.05),("car",0.20)]:
    print(f"  {cls:8s} Area {ar:.3f} -> level {risk_level(cls, ar)}")

print("\n=== Scenario 1: Right obstacle, not looking -> Inattention ===")
d = InattentionDetector()
b = box(0.90, 0.05)   # Far right, car
t = 0.0
for i in range(40):
    t += 0.1
    s, info = d.update([{"track_id":1,"class_name":"car","bbox":b}], head_yaw=0.0, now=t)
    if s == 1:
        print(f"  t={t:.1f}s Inattention occurred! {info['offenders']}")
        break
else:
    print("  No inattention occurred (Problem)")

print("\n=== Scenario 2: Right obstacle, looked within 2 secs -> Normal ===")
d = InattentionDetector()
t = 0.0
warned = False
for i in range(40):
    t += 0.1
    head = 0.0 if t < 1.0 else 30.0    # 30 degrees right after 1 sec
    s, info = d.update([{"track_id":1,"class_name":"car","bbox":b}], head_yaw=head, now=t)
    if s == 1: warned = True
print(f"  Inattention occurred: {warned} (Should be False)")

print("\n=== Scenario 3: Escalate to danger level as it gets closer after checking -> Re-check required ===")
d = InattentionDetector()
t = 0.0
events = []
for i in range(80):
    t += 0.1
    if t < 3.0:
        bb = box(0.90, 0.05)          # Obstacle level
        head = 30.0 if t > 0.5 else 0.0   # Checked early on
    else:
        bb = box(0.90, 0.25)          # Escalated to Danger level
        head = 0.0                     # Looking straight ahead again
    s, info = d.update([{"track_id":1,"class_name":"car","bbox":bb}], head_yaw=head, now=t)
    if s == 1: events.append(round(t,1))
print(f"  Inattention occurrence time: {events}  (Should be around 5.0s, which is 2 secs after escalation)")

print("\n=== Scenario 4: Frontal obstacle (within 20 deg) -> Normal even without turning head ===")
d = InattentionDetector()
b2 = box(0.55, 0.10)
t = 0.0; warned = False
print(f"  This obstacle angle: {bbox_to_angle(b2[0], b2[2]):+.1f} deg")
for i in range(40):
    t += 0.1
    s, _ = d.update([{"track_id":1,"class_name":"car","bbox":b2}], head_yaw=0.0, now=t)
    if s == 1: warned = True
print(f"  Inattention occurred: {warned} (Should be False)")

print("\n=== Scenario 5: Checking left obstacle direction ===")
d = InattentionDetector()
bl = box(0.10, 0.05)
print(f"  Left obstacle angle: {bbox_to_angle(bl[0], bl[2]):+.1f} deg")
t=0.0; warned=False
for i in range(40):
    t += 0.1
    head = 0.0 if t < 1.0 else -30.0   # Turned left
    s, _ = d.update([{"track_id":1,"class_name":"car","bbox":bl}], head_yaw=head, now=t)
    if s == 1: warned = True
print(f"  Inattention when looking left: {warned} (Should be False)")

d = InattentionDetector()
t=0.0; warned=False
for i in range(40):
    t += 0.1
    head = 0.0 if t < 1.0 else +30.0   # Turned opposite (right)
    s, _ = d.update([{"track_id":1,"class_name":"car","bbox":bl}], head_yaw=head, now=t)
    if s == 1: warned = True
print(f"  Inattention when looking opposite: {warned} (Should be True)")