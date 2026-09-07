#!/usr/bin/env python3
"""仿真板块端到端冒烟：登录 · catalog · campaign · sim_v3 · FT · promote · UI 静态检查。"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import websockets

from experiment_game.experiment.orchestrator import OperatorService

HTTP_PORT = 18081
WS_PORT = 18766
SIM_SUBJECT = "A01"


def check_ui_static() -> Tuple[bool, List[str]]:
    """检查 operator.html / operator.js 是否包含仿真 UI 关键元素。"""
    web = _REPO_ROOT / "experiment_game" / "web"
    html = (web / "operator.html").read_text(encoding="utf-8")
    js = (web / "js" / "operator.js").read_text(encoding="utf-8")
    errors: List[str] = []

    html_need = [
        'name="login_work_mode" value="sim"',
        'value="sim_v3_session"',
        'name="sim_run_id"',
        'name="sim_replay_align"',
        'id="sim-run-queue"',
        'id="sim-campaign-select"',
        'id="btn-sim-campaign-create"',
        'name="sim_use_campaign_queue"',
        'id="login-sim-subject"',
    ]
    for token in html_need:
        if token not in html:
            errors.append(f"operator.html 缺少: {token}")

    js_need = [
        "sim_catalog_ack",
        "sim_campaign_ack",
        "sim_campaign_list_ack",
        "isSimV3Mode",
        "renderSimRunQueue",
        "fillSimCampaignSelect",
        "use_campaign_queue",
        "replay_align",
        "activeCampaign",
    ]
    for token in js_need:
        if token not in js:
            errors.append(f"operator.js 缺少: {token}")

    m = re.search(r'operator\.js\?v=([^"]+)', html)
    if not m:
        errors.append("operator.html 未引用 operator.js 版本号")

    ok = not errors
    return ok, errors


async def _recv_until(ws, deadline: float, *, want_type: Optional[str] = None) -> Dict[str, Any]:
    while time.time() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=min(60, deadline - time.time()))
        msg = json.loads(raw)
        if want_type is None or msg.get("type") == want_type:
            return msg
    raise TimeoutError(f"timeout waiting for {want_type or 'message'}")


async def _drain_type(ws, msg_type: str, deadline: float) -> Optional[Dict[str, Any]]:
    found = None
    while time.time() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=min(30, deadline - time.time()))
        msg = json.loads(raw)
        if msg.get("type") == msg_type:
            found = msg
            break
    return found


async def ws_sim_flow(ws_url: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"steps": []}

    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "operator_hello"}))
        hello = await _recv_until(ws, time.time() + 10, want_type="operator_hello")
        presets = hello.get("model_presets") or []
        result["steps"].append(f"hello presets={len(presets)}")

        # 1) 仿真登录
        await ws.send(json.dumps({"type": "subject_login", "subject_id": SIM_SUBJECT, "sim_mode": True}))
        login = await _recv_until(ws, time.time() + 15, want_type="subject_login_ack")
        if not login.get("ok"):
            raise RuntimeError(f"sim login fail: {login.get('message')}")
        if not login.get("sim_mode"):
            raise RuntimeError("login 未返回 sim_mode")
        if login.get("subject_id") != SIM_SUBJECT:
            raise RuntimeError("login subject_id 不匹配")
        if not login.get("runs"):
            raise RuntimeError("login 未返回 runs 列表")
        if not login.get("index", {}).get("suggest_session_id"):
            raise RuntimeError("login 未返回 suggest_session_id")
        result["steps"].append("sim_login_ok")
        result["suggest_run"] = login["index"]["suggest_session_id"]

        # 2) catalog
        await ws.send(json.dumps({"type": "sim_catalog", "subject_id": SIM_SUBJECT}))
        catalog = await _recv_until(ws, time.time() + 15, want_type="sim_catalog_ack")
        if not catalog.get("ok") or not catalog.get("runs"):
            raise RuntimeError(f"sim_catalog fail: {catalog.get('message')}")
        run_ids = [r["run_id"] for r in catalog["runs"]]
        if "run3" not in run_ids:
            raise RuntimeError(f"catalog 无 run3: {run_ids[:5]}")
        result["steps"].append(f"catalog_ok runs={len(run_ids)}")

        # 3) campaign create（单 run 快速测）
        await ws.send(
            json.dumps(
                {
                    "type": "sim_campaign_create",
                    "subject_id": SIM_SUBJECT,
                    "session_queue": ["run3"],
                    "session_trials_total": 6,
                    "replay_align": "schedule_align",
                    "replay_speed": 8.0,
                }
            )
        )
        camp_ack = await _recv_until(ws, time.time() + 15, want_type="sim_campaign_ack")
        if not camp_ack.get("ok"):
            raise RuntimeError(f"campaign create fail: {camp_ack.get('message')}")
        manifest = camp_ack.get("manifest") or {}
        manifest_path = manifest.get("manifest_path")
        if not manifest_path or not Path(manifest_path).is_file():
            raise RuntimeError("campaign manifest 未落盘")
        result["steps"].append(f"campaign_ok id={manifest.get('campaign_id')}")
        result["campaign_manifest"] = manifest_path

        # 4) campaign list
        await ws.send(json.dumps({"type": "sim_campaign_list", "subject_id": SIM_SUBJECT}))
        clist = await _recv_until(ws, time.time() + 15, want_type="sim_campaign_list_ack")
        if not clist.get("ok") or not clist.get("campaigns"):
            raise RuntimeError("sim_campaign_list 失败")
        result["steps"].append(f"campaign_list_ok n={len(clist['campaigns'])}")

        # 5) sim_v3 session（6 trial，Campaign 队列）
        save_root = (
            _REPO_ROOT / "experiment_game" / "data" / "sim_subjects" / SIM_SUBJECT / "sessions"
        ).resolve()
        cfg = {
            "schema_version": 2,
            "subject": {"subject_id": SIM_SUBJECT, "session_id": "auto", "notes": "smoke_sim"},
            "acquisition": {
                "enabled": True,
                "board_mode": "bci2a_replay",
                "serial_port": "COM5",
                "sample_rate_hz": 250,
                "markers_lsl": True,
            },
            "experiment": {
                "phase_mode": "sim_v3_session",
                "open_subject_page": False,
                "protocol_locked": True,
                "ready_timeout_s": 60,
                "seed": 42,
            },
            "storage": {
                "save_root": str(save_root),
                "save_layout": "phase_folders",
                "save_eeg": True,
                "save_events": True,
                "save_session_meta": True,
                "save_continuous_master": True,
                "save_phase_slices": True,
                "save_trial_index": True,
            },
            "extensions": {
                "sim": {
                    "run_id": "",
                    "use_campaign_queue": True,
                    "campaign_manifest": manifest_path,
                    "session_trials_total": 6,
                    "replay_speed": 8.0,
                    "replay_align": "schedule_align",
                }
            },
        }
        await ws.send(json.dumps({"type": "session_start", "run_config": cfg}))

        deadline = time.time() + 240
        saved = None
        while time.time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=min(90, deadline - time.time()))
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "config_ack":
                if not msg.get("ok"):
                    raise RuntimeError(f"config_ack: {msg.get('errors')}")
                result["steps"].append("config_ack_ok")
            elif t == "session_started":
                if msg.get("phase_mode") != "sim_v3_session":
                    raise RuntimeError("session_started phase 错误")
                if not msg.get("sim", {}).get("sim_mode"):
                    raise RuntimeError("session_started 缺少 sim meta")
                result["steps"].append("session_started_ok")
                await ws.send(json.dumps({"type": "ready"}))
            elif t == "session_saved":
                saved = msg
                break
            elif t == "session" and msg.get("status") == "error":
                raise RuntimeError(msg.get("message") or "session error")

        if saved is None:
            raise TimeoutError("sim session 未在 240s 内完成")

        root = Path(saved["root"])
        if saved.get("phase_mode") != "sim_v3_session":
            raise RuntimeError("session_saved phase 错误")
        if not saved.get("sim_index"):
            raise RuntimeError("session_saved 缺少 sim_index")
        camp = saved.get("campaign") or {}
        if not camp.get("summary_path") or not Path(camp["summary_path"]).is_file():
            raise RuntimeError("Campaign summary.md 未生成")
        result["steps"].append("session_saved_ok")
        result["session_dir"] = str(root)
        result["campaign_summary"] = camp["summary_path"]

        # 6) subject_info（仿真 sessions 列表）
        await ws.send(json.dumps({"type": "subject_info", "subject_id": SIM_SUBJECT}))
        info = await _recv_until(ws, time.time() + 15, want_type="subject_info_ack")
        if not info.get("ok") or not info.get("sim_mode"):
            raise RuntimeError("subject_info sim 失败")
        sessions = info.get("sessions") or []
        if not any(str(root).endswith(s["dir"]) or s.get("path", "").endswith(root.name) for s in sessions):
            # 宽松匹配 dir 名
            if not any(root.name in (s.get("dir") or "") for s in sessions):
                raise RuntimeError("subject_info 未列出新 session")
        result["steps"].append(f"subject_info_ok sessions={len(sessions)}")

        # 7) 微调（no_replay 加速）
        await ws.send(
            json.dumps(
                {
                    "type": "finetune_start",
                    "subject_id": SIM_SUBJECT,
                    "session_paths": [str(root)],
                    "exclude_invalid": False,
                    "no_replay": True,
                    "use_replay": False,
                }
            )
        )
        ft_ack = await _recv_until(ws, time.time() + 15, want_type="finetune_ack")
        if not ft_ack.get("ok"):
            raise RuntimeError(f"finetune_ack: {ft_ack.get('message')}")

        ft_done = None
        ft_deadline = time.time() + 600
        while time.time() < ft_deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=min(120, ft_deadline - time.time()))
            msg = json.loads(raw)
            if msg.get("type") == "finetune_done":
                ft_done = msg
                break
        if ft_done is None:
            raise TimeoutError("finetune 超时")
        if not ft_done.get("ok"):
            raise RuntimeError(f"finetune fail: {ft_done.get('message')}")
        out_dir = Path(ft_done["out_dir"])
        for name in ("best_task.pt", "best_three.pt"):
            if not (out_dir / name).is_file():
                raise RuntimeError(f"FT 缺少 {name}")
        result["steps"].append(f"finetune_ok out={out_dir.name}")

        # 8) promote
        await ws.send(
            json.dumps(
                {
                    "type": "finetune_promote",
                    "subject_id": SIM_SUBJECT,
                    "ft_run_dir": str(out_dir),
                    "reason": "smoke_test",
                }
            )
        )
        prom = await _recv_until(ws, time.time() + 15, want_type="finetune_promote_ack")
        if not prom.get("ok"):
            raise RuntimeError(f"promote fail: {prom.get('message')}")
        weights = prom.get("weights") or {}
        if not weights.get("ok"):
            raise RuntimeError("promote 后 weights 不可用")
        cur = _REPO_ROOT / "experiment_game" / "data" / "sim_subjects" / SIM_SUBJECT / "models" / "current"
        if not (cur / "best_task.pt").is_file():
            raise RuntimeError("current/best_task.pt 不存在")
        result["steps"].append("promote_ok")

        # 9) timing_align 单场（不建 campaign，快速校验）
        cfg2 = dict(cfg)
        cfg2["subject"] = {"subject_id": SIM_SUBJECT, "session_id": "run4", "notes": "smoke_timing"}
        cfg2["extensions"] = {
            "sim": {
                "run_id": "run4",
                "session_trials_total": 6,
                "replay_speed": 8.0,
                "replay_align": "timing_align",
            }
        }
        await ws.send(json.dumps({"type": "session_start", "run_config": cfg2}))
        saved2 = None
        deadline2 = time.time() + 240
        while time.time() < deadline2:
            raw = await asyncio.wait_for(ws.recv(), timeout=min(90, deadline2 - time.time()))
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "config_ack" and not msg.get("ok"):
                raise RuntimeError(f"timing config: {msg.get('errors')}")
            if t == "session_started":
                await ws.send(json.dumps({"type": "ready"}))
            if t == "session_saved":
                saved2 = msg
                break
        if saved2 is None:
            raise RuntimeError("timing_align session 未完成")
        meta_path = Path(saved2["root"]) / "session.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("replay_align") != "timing_align":
            raise RuntimeError("timing_align 未写入 meta")
        result["steps"].append("timing_align_session_ok")

    return result


def verify_artifacts(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    root = Path(result.get("session_dir") or "")
    if not root.is_dir():
        errors.append("session_dir 不存在")
    else:
        for rel in ("eeg.csv", "session.meta.json", "v3_report.json", "run_config.json"):
            if not (root / rel).is_file():
                errors.append(f"缺少 {rel}")
        meta = json.loads((root / "session.meta.json").read_text(encoding="utf-8"))
        if not meta.get("sim_mode"):
            errors.append("meta.sim_mode 未设置")
        if meta.get("source_run") != "run3":
            errors.append(f"meta.source_run 期望 run3 实际 {meta.get('source_run')}")

    idx = _REPO_ROOT / "experiment_game" / "data" / "sim_subjects" / SIM_SUBJECT / "index.json"
    if not idx.is_file():
        errors.append("index.json 未生成")
    return not errors, errors


def main() -> int:
    print("=== 仿真板块冒烟测试 ===")
    ui_ok, ui_errs = check_ui_static()
    print(f"[UI 静态] {'PASS' if ui_ok else 'FAIL'}")
    for e in ui_errs:
        print(f"  - {e}")

    mat = _REPO_ROOT / "DATA" / "bci2a" / f"{SIM_SUBJECT}T.mat"
    if not mat.is_file():
        print(f"[SKIP WS] 缺少 {mat}")
        return 1 if not ui_ok else 0

    svc = OperatorService(http_port=HTTP_PORT, ws_port=WS_PORT)
    t = threading.Thread(target=svc.serve_forever, name="sim-smoke-svc", daemon=True)
    t.start()
    time.sleep(1.0)
    ws_ok = False
    ws_result: Dict[str, Any] = {}
    try:
        ws_result = asyncio.run(ws_sim_flow(f"ws://127.0.0.1:{WS_PORT}"))
        ws_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"[WS 流程] FAIL: {exc}")
        if ws_result.get("steps"):
            print("  已完成步骤:", " → ".join(ws_result["steps"]))
    finally:
        svc.stop()

    if ws_ok:
        print("[WS 流程] PASS")
        for s in ws_result.get("steps", []):
            print(f"  · {s}")
        art_ok, art_errs = verify_artifacts(ws_result)
        print(f"[落盘校验] {'PASS' if art_ok else 'FAIL'}")
        for e in art_errs:
            print(f"  - {e}")
    else:
        art_ok = False

    all_ok = ui_ok and ws_ok and art_ok
    print("SIM_SMOKE_OK" if all_ok else "SIM_SMOKE_FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
