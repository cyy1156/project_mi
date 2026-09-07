#!/usr/bin/env python3
"""换场 + 问卷端到端冒烟：B1 换场分段 → B2 新 session 继续 → Q 问卷落盘。"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import websockets

from experiment_game.experiment.orchestrator import OperatorService

HTTP_PORT = 18082
WS_PORT = 18768

CFG = {
    "schema_version": 2,
    "subject": {"subject_id": "spsub01", "session_id": "ses01", "notes": "split_smoke"},
    "acquisition": {
        "enabled": True,
        "board_mode": "synthetic",
        "serial_port": "",
        "markers_lsl": True,
    },
    "experiment": {
        "acquire_trials": 4,
        "learn_trials_per_step": 1,
        "skip_adapt": True,
        "skip_learn": True,
        "skip_gate": True,
        "ready_timeout_s": 30,
        "timing": {
            "fixation_s": 0.2, "cue_s": 0.2, "mi_s": 1.0,
            "post_mi_hold_s": 0.0, "rest_s": 4.0, "transition_s": 0.5,
        },
        "split": {"settle_s": 5.0},
    },
    "storage": {"save_root": "experiment_game/data/sessions"},
}


async def drive(ws_url: str) -> dict:
    got = {
        "seg1": None, "segment_saved": None, "seg2": None,
        "final_saved": None, "q_ack": None,
    }
    async with websockets.connect(ws_url) as ws:
        async def send(m):
            await ws.send(json.dumps(m, ensure_ascii=False))

        await send({"type": "operator_hello"})
        deadline = time.time() + 420
        split_sent = False
        second_b_sent = False
        q_sent = False
        started = False
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                # 无消息时推进状态机动作
                if got["segment_saved"] and not second_b_sent:
                    await send({"type": "operator", "action": "split_session"})
                    second_b_sent = True
                    print("[client] B#2 sent")
                elif got["final_saved"] and not q_sent:
                    await send({"type": "questionnaire_open"})
                    q_sent = True
                    print("[client] Q sent")
                elif got["q_ack"]:
                    break
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "hello":
                await send({"type": "ready"})
                if not started:
                    started = True
                    await send({"type": "session_start", "run_config": CFG})
            elif t == "config_ack" and not msg.get("ok"):
                raise RuntimeError(f"config error: {msg.get('errors')}")
            elif t == "session_started":
                seg = msg.get("segment", 1)
                print(f"[client] session_started seg={seg} trials={msg.get('acquire_trials')} root={msg.get('session_root')}")
                if seg == 1:
                    got["seg1"] = msg
                else:
                    got["seg2"] = msg
            elif t == "stage":
                st = msg.get("stage")
                if st == "trial_start":
                    got["trial_seen"] = True
                # trial 2 一开始就请求换场（当前 trial 走完后分段）
                if (
                    st == "trial_start"
                    and msg.get("phase") == "acquire"
                    and msg.get("trial_id") == 2
                    and got["seg1"]
                    and not split_sent
                ):
                    await send({"type": "operator", "action": "split_session"})
                    split_sent = True
                    print("[client] B#1 sent")
            elif t == "session_segment_saved":
                got["segment_saved"] = msg
                print(f"[client] segment_saved: {msg.get('message')}")
            elif t == "session_saved" and got["seg2"]:
                got["final_saved"] = msg
                print("[client] final session_saved")
            elif t == "questionnaire" and not got.get("q_form"):
                got["q_form"] = msg
                answers = {}
                for q in msg["questions"]:
                    if q["kind"] == "scale5":
                        answers[q["id"]] = "4"
                    elif q["kind"] == "choice":
                        answers[q["id"]] = q["options"][0]
                await send({"type": "questionnaire_result", "form": "post", "answers": answers})
            elif t == "questionnaire_ack":
                got["q_ack"] = msg
                print(f"[client] questionnaire_ack ok={msg.get('ok')}")
    return got


def main() -> int:
    svc = OperatorService(http_port=HTTP_PORT, ws_port=WS_PORT)
    svc.start()
    result = {}
    try:
        result = asyncio.run(drive(f"ws://127.0.0.1:{WS_PORT}"))
    finally:
        time.sleep(1.0)
        svc.stop()

    seg1, seg2 = result.get("seg1"), result.get("seg2")
    assert seg1, "未收到第一段 session_started"
    assert result.get("segment_saved"), "未收到 session_segment_saved（B#1 未生效）"
    assert seg2, "未收到第二段 session_started（B#2 未生效）"
    assert result.get("final_saved"), "未收到最终 session_saved"
    assert result.get("q_ack", {}).get("ok"), f"问卷未成功: {result.get('q_ack')}"

    r1, r2 = Path(seg1["session_root"]), Path(seg2["session_root"])
    assert r1.is_dir() and r2.is_dir(), "分段目录缺失"
    assert r1.parent == r2.parent and "ses01" in r1.name and "ses02" in r2.name, (
        f"session 命名不符预期: {r1.name} / {r2.name}"
    )
    assert seg2["acquire_trials"] == 2, "第二段应只含剩余 2 个 trial"
    ev1 = (r1 / "events.jsonl").read_text(encoding="utf-8")
    assert "session_split" in ev1, "第一段缺 session_split 事件"
    ev2 = (r2 / "events.jsonl").read_text(encoding="utf-8")
    assert "settle_start" in ev2 and "settle_end" in ev2, "第二段缺 settle 事件"
    assert (r1 / "eeg.csv").is_file() and (r2 / "eeg.csv").is_file(), "分段 eeg.csv 缺失"
    # trial 计数：两段 events 里 acquire 的 cue 数应各为 2
    n1 = sum(1 for l in ev1.splitlines() if '"cue"' in l and '"acquire"' in l)
    n2 = sum(1 for l in ev2.splitlines() if '"cue"' in l and '"acquire"' in l)
    assert n1 == 2 and n2 == 2, f"分段 trial 数不符: {n1}/{n2}"
    # 问卷文件
    qdir = r2 / "99_summary"
    qfiles = list(qdir.glob("questionnaire_post_*.json")) if qdir.is_dir() else []
    assert qfiles, "问卷文件未落盘"
    print(f"\nSPLIT+QUESTIONNAIRE SMOKE OK")
    print(f"  seg1={r1.name} (2 trials, has session_split)")
    print(f"  seg2={r2.name} (2 trials, settle 5s)")
    print(f"  questionnaire={qfiles[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
