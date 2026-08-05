"""P0/P1：切段 + 合法窗枚举（不推理）。

用法：
  python build_streams.py
  python build_streams.py --sessions sub02_ses01_20260723_180607
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import DEFAULT_SESSIONS, DOCS_OUT, SESSIONS_ROOT  # noqa: E402
from stream import build_eval_stream, save_stream_artifacts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="伪在线：切段+段内滑窗")
    p.add_argument(
        "--sessions",
        default=",".join(DEFAULT_SESSIONS),
        help="逗号分隔 session 目录名",
    )
    p.add_argument("--no-filter", action="store_true")
    args = p.parse_args()
    names = [s.strip() for s in args.sessions.split(",") if s.strip()]
    summary = []
    for name in names:
        path = SESSIONS_ROOT / name
        print(f"\n=== build {name} ===", flush=True)
        stream = build_eval_stream(path, apply_filter=not args.no_filter)
        out = save_stream_artifacts(stream)
        row = {
            "session": name,
            "subject_id": stream.subject_id,
            "out": str(out),
            **{k: stream.meta[k] for k in ("n_segments", "n_windows", "n_mi", "n_rest")},
            "n_skipped": len(stream.meta.get("skipped") or []),
        }
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (DOCS_OUT / "out" / "build_streams_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
