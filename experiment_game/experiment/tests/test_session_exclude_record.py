"""session 不记入实验记录（文件保留 · Campaign 释放 run）。"""

from __future__ import annotations

import json
from pathlib import Path

from experiment_game.experiment.sim.campaign import create_campaign, load_campaign
from experiment_game.experiment.sim.campaign_summary import (
    append_session_result,
    exclude_session_from_records,
)


def test_exclude_session_from_campaign_releases_run(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path
    sid = "A01"
    camp = create_campaign(sid, ["run3", "run4"], repo_root=repo)
    session_dir = repo / "experiment_game/data/sim_subjects/A01/sessions/A01_run3_test"
    session_dir.mkdir(parents=True)
    (session_dir / "session.meta.json").write_text(
        json.dumps(
            {
                "subject_id": sid,
                "session_id": "run3",
                "source_run": "run3",
                "phase_mode": "sim_v3_session",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    camp["runs_consumed"] = ["run3"]
    camp["current_index"] = 1
    from experiment_game.experiment.sim.campaign import save_campaign

    save_campaign(camp)
    append_session_result(camp, session_dir=session_dir, summary={"session_score": 1})

    out = exclude_session_from_records(session_dir, campaign_manifest=camp, repo_root=repo)
    assert out["ok"] is True
    meta = json.loads((session_dir / "session.meta.json").read_text(encoding="utf-8"))
    assert meta.get("record_excluded") is True

    reloaded = load_campaign(camp["manifest_path"])
    assert not any(
        Path(str(r.get("session_dir"))).resolve() == session_dir.resolve()
        for r in reloaded.get("sessions_completed") or []
    )
    assert "run3" not in (reloaded.get("runs_consumed") or [])
    assert reloaded.get("current_index") == 0

    out2 = exclude_session_from_records(session_dir, campaign_manifest=reloaded, repo_root=repo)
    assert out2.get("already_excluded") is True
