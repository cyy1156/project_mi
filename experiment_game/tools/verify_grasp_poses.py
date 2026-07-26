import math
import re
from pathlib import Path

src = Path(r"d:\cyy\MI\experiment_game\web\js\scene.js").read_text(encoding="utf-8")
for name in [
    "_poseReach",
    "_poseLift",
    "_poseAway",
    "_attachCupToHand",
    "_hideCupAway",
    "full_grasp",
    "sameMi",
    "hand.attach",
]:
    assert name in src, f"missing {name}"


def grab(method: str) -> tuple[float, float, float]:
    mx = re.search(
        rf"{method}\(side\) \{{[\s\S]*?pos: new THREE\.Vector3\(sign \* ([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)",
        src,
    )
    assert mx, method
    return float(mx.group(1)), float(mx.group(2)), float(mx.group(3))


rest = (0.14, -0.16, -0.4)
reach = grab("_poseReach")
lift = grab("_poseLift")
away = grab("_poseAway")
reach_dist = math.dist(reach, rest)
lateral = abs(away[0] - rest[0])
print("reach", reach, "dist", round(reach_dist, 3))
print("lift", lift)
print("away", away, "lateral", round(lateral, 3))
assert reach_dist > 0.35 and lateral > 0.2
assert reach[2] < rest[2] - 0.3
assert reach[1] > rest[1] + 0.05
print("SCENE_STRUCTURE_PASS")
