import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    src = (_ROOT / "web" / "js" / "scene.js").read_text(encoding="utf-8")
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
        if name not in src:
            raise SystemExit(f"missing {name}")

    def grab(method: str) -> tuple[float, float, float]:
        mx = re.search(
            rf"{method}\(side\) \{{[\s\S]*?pos: new THREE\.Vector3\(sign \* ([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)",
            src,
        )
        if not mx:
            raise SystemExit(f"pattern not found: {method}")
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
    if not (reach_dist > 0.35 and lateral > 0.2):
        raise SystemExit("reach/away distance check failed")
    if not (reach[2] < rest[2] - 0.3):
        raise SystemExit("reach z check failed")
    if not (reach[1] > rest[1] + 0.05):
        raise SystemExit("reach y check failed")
    print("SCENE_STRUCTURE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
