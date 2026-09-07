# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import json

p = Path(r"d:/MI/code/train_lab/out/5070_challenge_exp36_pool_xtrack_accpaper/replay")
d0 = json.loads((p / "day0_latest.json").read_text(encoding="utf-8"))
d1 = json.loads((p / "day1_latest.json").read_text(encoding="utf-8"))
train = json.loads(
    Path(r"d:/MI/code/train_lab/out/5070_challenge_exp36_pool_xtrack_accpaper/day1_train_summary.json").read_text(
        encoding="utf-8"
    )
)
doc = {
    "experiment": 36,
    "scheme_version": "v0.1",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "status": "CLOSED_day0_day1_no_replace",
    "day0_gates": d0.get("gates"),
    "day1_gates": d1.get("gates"),
    "day1_any_pass": d1.get("any_pass_replace"),
    "train_jobs": train.get("jobs"),
    "c1": "deferred_pending_rules",
    "conclusion": (
        "No arm met Val>=0.568 AND Wilcoxon p<0.05. Keep Exp34 S0 CSV. "
        "M7/M7b strongest (~0.604, p=0.062). Pool expansion P1 negative."
    ),
}
out = Path(r"d:/MI/资料/模型训练/36_旁路_官方主交卷_扩池与跨轨融合_accpaper/总结")
text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
(out / "metrics_full_latest.json").write_text(text, encoding="utf-8")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
(out / f"metrics_full_{stamp}.json").write_text(text, encoding="utf-8")
print("wrote metrics", stamp)
