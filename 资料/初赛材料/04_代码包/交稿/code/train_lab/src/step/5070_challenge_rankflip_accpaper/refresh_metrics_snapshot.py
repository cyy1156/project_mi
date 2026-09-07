#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""刷新 metrics_full_latest.json（结案快照）。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(r"D:\MI")
OUT = REPO / "code" / "train_lab" / "out" / "5070_challenge_rankflip_accpaper"
DOC = REPO / "资料" / "模型训练" / "35_旁路_官方vsOpenBMI_三分类排名不一致_融合重标定与骨干消融_accpaper"

def main() -> None:
    decision = json.loads((OUT / "submissions" / "decision_latest.json").read_text(encoding="utf-8"))
    fm = json.loads((OUT / "replay" / "replay_FM_latest.json").read_text(encoding="utf-8"))
    d = json.loads((OUT / "replay" / "replay_D_latest.json").read_text(encoding="utf-8"))
    sig = json.loads((OUT / "stats" / "paired_sig_latest.json").read_text(encoding="utf-8"))
    payload = {
        "experiment": 35,
        "scheme_version": "v0.2.1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "P0_D_closed_S0_final_H_multiseed_running",
        "paths": {
            "replay_FM": str(OUT / "replay" / "replay_FM_latest.json"),
            "replay_D": str(OUT / "replay" / "replay_D_latest.json"),
            "decision": str(OUT / "submissions" / "decision_latest.json"),
            "paired_sig": str(OUT / "stats" / "paired_sig_latest.json"),
            "ranking_doc": str(REPO / "资料" / "模型训练" / "跨域三分类成员排名对照表.md"),
            "registry": str(DOC / "总结" / "结果登记表.md"),
            "report_draft": str(DOC / "总结" / "初赛报告素材_模型选择与验证.md"),
        },
        "external_submission": decision.get("external_submission"),
        "replace_gate": decision.get("replace_gate"),
        "exp34_csv_status": decision.get("exp34_csv_status"),
        "fm_decision": (fm.get("decision") or {}),
        "fm_arm_means": {
            k: {"val_acc_mean": v.get("val_acc_mean"), "val_acc_std": v.get("val_acc_std")}
            for k, v in (fm.get("results") or {}).items()
        },
        "d_arm_means": {
            k: {"val_acc_mean": v.get("val_acc_mean"), "val_acc_std": v.get("val_acc_std")}
            for k, v in (d.get("results") or {}).items()
        },
        "paired_sig_summary": [
            {
                "id": c.get("candidate_id"),
                "arm": c.get("source_arm"),
                "mean_diff": c.get("mean_diff"),
                "wilcoxon_p": c.get("wilcoxon_p"),
            }
            for c in (sig.get("comparisons") or [])
        ],
    }
    out = DOC / "总结" / f"metrics_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    out.write_text(text, encoding="utf-8")
    (DOC / "总结" / "metrics_full_latest.json").write_text(text, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
