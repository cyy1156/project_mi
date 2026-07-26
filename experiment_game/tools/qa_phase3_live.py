"""Phase 3 live QA: start session, drive browser via CDP, check UI + operator + object swap."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[2]
PY = REPO / "collect_data" / "LSL_connect_model" / "LSL_connect_model" / ".venv" / "Scripts" / "python.exe"
LOG = REPO / "experiment_game" / "data" / "_p3_live_qa_log.txt"
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def find_edge() -> Path:
    for p in EDGE_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("Edge not found")


async def cdp_eval(ws, expression: str, nid: int):
    await ws.send(
        json.dumps(
            {
                "id": nid,
                "method": "Runtime.evaluate",
                "params": {"expression": expression, "returnByValue": True, "awaitPromise": True},
            }
        )
    )
    while True:
        raw = json.loads(await ws.recv())
        if raw.get("id") == nid:
            return raw


async def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # kill leftover
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-NetTCPConnection -LocalPort 8080,8765,9222 -ErrorAction SilentlyContinue | "
         "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"],
        check=False,
    )
    time.sleep(1)

    cmd = [
        str(PY),
        "-m",
        "experiment_game.tools.run_phase2_session",
        "--yes",
        "--no-acq",
        "--fast",
        "--auto-continue",
        "--no-browser",
        "--session",
        "ses_p3_liveqa",
        "--acquire-trials",
        "12",
        "--learn-trials",
        "1",
        "--skip-adapt",
        "--skip-learn",
        "--skip-gate",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        stdout=LOG.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )

    # wait http
    for _ in range(40):
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/", timeout=1)
            break
        except Exception:
            time.sleep(0.25)
    else:
        proc.kill()
        print("HTTP_FAIL")
        return 1

    edge = find_edge()
    profile = REPO / "experiment_game" / "data" / "_edge_qa_profile"
    profile.mkdir(parents=True, exist_ok=True)
    browser = subprocess.Popen(
        [
            str(edge),
            "--headless=new",
            "--disable-gpu",
            "--remote-debugging-port=9222",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    findings: list[str] = []
    try:
        tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json"))
        page = next(t for t in tabs if t.get("type") == "page")
        async with websockets.connect(page["webSocketDebuggerUrl"]) as ws:
            nid = 0

            async def call(method, params=None):
                nonlocal nid
                nid += 1
                payload = {"id": nid, "method": method}
                if params is not None:
                    payload["params"] = params
                await ws.send(json.dumps(payload))
                while True:
                    raw = json.loads(await ws.recv())
                    if raw.get("id") == nid:
                        return raw

            await call("Runtime.enable")
            await call("Page.enable")
            await call("Page.navigate", {"url": "http://127.0.0.1:8080/"})
            await asyncio.sleep(2.5)

            # UI checks
            checks = {
                "opbar": "!!document.getElementById('opbar')",
                "status": "document.getElementById('status')?.textContent || ''",
                "opState": "document.getElementById('op-state')?.textContent || ''",
                "phase": "document.getElementById('phase-tag')?.textContent || ''",
                "canvas": "!!document.getElementById('c')",
                "sceneObj": "window.__miScene?._objectId || ''",
                "sceneId": "window.__miScene?._sceneId || ''",
                "wsOk": "(document.getElementById('status')?.textContent || '').includes('连接') || (document.getElementById('status')?.textContent || '').includes('就绪') || (document.getElementById('status')?.textContent || '').includes('已')",
            }
            snap = {}
            for i, (k, expr) in enumerate(checks.items(), start=100):
                r = await cdp_eval(ws, expr, i)
                snap[k] = ((r.get("result") or {}).get("result") or {}).get("value")

            print("SNAP1", json.dumps(snap, ensure_ascii=False))
            if not snap.get("opbar"):
                findings.append("opbar missing")
            if not snap.get("canvas"):
                findings.append("canvas missing")
            if not snap.get("sceneObj"):
                findings.append("scene not ready (__miScene missing)")

            # wait until acquire progresses / object becomes bottle or apple
            seen_objs = set()
            seen_scenes = set()
            paused_ok = False
            for step in range(60):
                await asyncio.sleep(0.8)
                r = await cdp_eval(
                    ws,
                    "({obj: window.__miScene?._objectId, sc: window.__miScene?._sceneId, "
                    "st: document.getElementById('status')?.textContent, "
                    "op: document.getElementById('op-state')?.textContent, "
                    "phase: document.getElementById('phase-tag')?.textContent, "
                    "hud: document.getElementById('hud-text')?.textContent})",
                    200 + step,
                )
                val = ((r.get("result") or {}).get("result") or {}).get("value") or {}
                if val.get("obj"):
                    seen_objs.add(val["obj"])
                if val.get("sc"):
                    seen_scenes.add(val["sc"])
                if step == 8:
                    # try pause via page key
                    await call(
                        "Input.dispatchKeyEvent",
                        {"type": "keyDown", "windowsVirtualKeyCode": 80, "code": "KeyP", "key": "p"},
                    )
                    await call(
                        "Input.dispatchKeyEvent",
                        {"type": "keyUp", "windowsVirtualKeyCode": 80, "code": "KeyP", "key": "p"},
                    )
                    await asyncio.sleep(0.5)
                    r2 = await cdp_eval(
                        ws,
                        "document.getElementById('opbar')?.classList.contains('paused') || "
                        "(document.getElementById('op-state')?.textContent || '').includes('PAUSED')",
                        900,
                    )
                    paused_ok = bool(((r2.get("result") or {}).get("result") or {}).get("value"))
                    # resume
                    await call(
                        "Input.dispatchKeyEvent",
                        {"type": "keyDown", "windowsVirtualKeyCode": 80, "code": "KeyP", "key": "p"},
                    )
                    await call(
                        "Input.dispatchKeyEvent",
                        {"type": "keyUp", "windowsVirtualKeyCode": 80, "code": "KeyP", "key": "p"},
                    )
                if "完成" in str(val.get("st") or "") or "本会话结束" in str(val.get("hud") or ""):
                    break
                # stop early if process exited
                if proc.poll() is not None:
                    break

            print("SEEN_OBJS", sorted(seen_objs))
            print("SEEN_SCENES", sorted(seen_scenes))
            print("PAUSE_OK", paused_ok)
            if "cup" not in seen_objs:
                findings.append("never saw cup object")
            if "bottle" not in seen_objs and "apple" not in seen_objs:
                findings.append("object rotation not observed in UI")
            if not paused_ok:
                findings.append("pause key did not toggle opbar paused state")

    finally:
        browser.kill()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            findings.append("session process hang")

    # parse log for object_change / scene_change
    text = LOG.read_text(encoding="utf-8", errors="ignore") if LOG.exists() else ""
    print("LOG_HAS_OBJECT_CHANGE", "[object_change]" in text)
    print("LOG_HAS_SCENE_CHANGE", "[scene_change]" in text)
    if "[object_change]" not in text:
        findings.append("backend log missing object_change")
    if "[scene_change]" not in text:
        findings.append("backend log missing scene_change (expected with 12 trials)")

    # events reject path smoke via separate short run? skip if time
    if findings:
        print("FINDINGS:")
        for f in findings:
            print(" -", f)
        print("QA_FAIL")
        return 1
    print("QA_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
