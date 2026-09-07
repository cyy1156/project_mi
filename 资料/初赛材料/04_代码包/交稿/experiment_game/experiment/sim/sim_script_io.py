"""仿真 session 脚本落盘 / 加载（供 FT 按 mat cue 切窗）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiment_game.experiment.sim.run_to_session_map import (
    SimTrial,
    SimTrialScript,
    build_sim_script,
    build_sim_script_from_labels,
)


def sim_script_to_dict(script: SimTrialScript) -> Dict[str, Any]:
    return {
        "subject_id": script.subject_id,
        "run_id": script.run_id,
        "mat_path": script.mat_path,
        "fs": float(script.fs),
        "blocks": int(script.blocks),
        "trials_per_block": int(script.trials_per_block),
        "session_trials_total": int(script.session_trials_total),
        "align_mode": script.align_mode,
        "labels_by_block": script.labels_by_block,
        "trials": [
            {
                "cue_sample": int(t.cue_sample),
                "label": int(t.label),
                "mat_trial_index": int(t.mat_trial_index),
                "rest_start_sample": int(t.rest_start_sample),
                "rest_end_sample": int(t.rest_end_sample),
            }
            for t in script.trials
        ],
        "meta": dict(script.meta or {}),
    }


def sim_script_from_dict(data: Dict[str, Any]) -> SimTrialScript:
    trials = [
        SimTrial(
            cue_sample=int(t["cue_sample"]),
            label=int(t["label"]),
            mat_trial_index=int(t["mat_trial_index"]),
            rest_start_sample=int(t["rest_start_sample"]),
            rest_end_sample=int(t["rest_end_sample"]),
        )
        for t in data.get("trials") or []
    ]
    return SimTrialScript(
        subject_id=str(data["subject_id"]),
        run_id=str(data["run_id"]),
        mat_path=str(data["mat_path"]),
        fs=float(data["fs"]),
        x8=np.zeros((0, 8), dtype=np.float64),
        trials=trials,
        trials_unused=list(data.get("trials_unused") or []),
        labels_by_block=list(data.get("labels_by_block") or []),
        blocks=int(data.get("blocks") or 1),
        trials_per_block=int(data.get("trials_per_block") or len(trials)),
        session_trials_total=int(data.get("session_trials_total") or len(trials)),
        align_mode=str(data.get("align_mode") or "schedule_align"),
        meta=dict(data.get("meta") or {}),
    )


def write_sim_script(session_dir: Path | str, script: SimTrialScript) -> Path:
    p = Path(session_dir) / "sim_script.json"
    p.write_text(json.dumps(sim_script_to_dict(script), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def load_sim_script(session_dir: Path | str) -> Optional[SimTrialScript]:
    p = Path(session_dir) / "sim_script.json"
    if not p.is_file():
        return None
    return sim_script_from_dict(json.loads(p.read_text(encoding="utf-8")))


def rebuild_sim_script_from_session(session_dir: Path | str) -> Optional[SimTrialScript]:
    """旧 session 无 sim_script.json 时，优先按 trial_table 标签序重建。"""
    sd = Path(session_dir)
    meta_p = sd / "session.meta.json"
    if not meta_p.is_file():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if not meta.get("sim_mode") and meta.get("phase_mode") != "sim_v3_session":
        return None
    run_id = str(meta.get("source_run") or meta.get("session_id") or "")
    if not run_id.startswith("run"):
        return None
    mat_path = meta.get("source_mat")
    if not mat_path:
        from experiment_game.experiment.sim.bci2a_catalog import resolve_mat_path

        mat_path = str(resolve_mat_path(str(meta.get("subject_id") or "A01")))
    blocks = int((meta.get("v3_config_effective") or {}).get("blocks") or 2)
    align = str(meta.get("replay_align") or "schedule_align")
    rest_s = float((meta.get("v3_config_effective") or {}).get("inter_trial_rest_s") or 4.0)

    table_p = sd / "alignment" / "trial_table.csv"
    if table_p.is_file():
        import pandas as pd

        rows = list(pd.read_csv(table_p).to_dict(orient="records"))
        labels = [int(r["label"]) for r in rows if "label" in r]
        if labels:
            return build_sim_script_from_labels(
                mat_path,
                run_id,
                labels,
                blocks=blocks,
                align_mode=align,
                rest_s=rest_s,
            )

    seed = None
    rc = sd / "run_config.json"
    if rc.is_file():
        seed = json.loads(rc.read_text(encoding="utf-8")).get("experiment", {}).get("seed")
    n_trials = int(meta.get("session_trials_total") or meta.get("trial_count") or 36)
    return build_sim_script(
        mat_path,
        run_id,
        session_trials_total=n_trials,
        blocks=blocks,
        align_mode=align,
        seed=seed,
        rest_s=rest_s,
    )
