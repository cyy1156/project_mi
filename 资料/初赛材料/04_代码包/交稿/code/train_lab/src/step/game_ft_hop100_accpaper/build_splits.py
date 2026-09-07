"""切分冒烟：写出 split_manifest.json。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import DEFAULT_SESSIONS, DOCS_OUT, SESSIONS_ROOT  # noqa: E402
from data_split import build_trial_split, write_split_artifacts  # noqa: E402
from shared_hparams import SHARED  # noqa: E402
from stream import build_eval_stream  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="构建 trial 对半 split_manifest")
    p.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS))
    p.add_argument("--no-filter", action="store_true")
    args = p.parse_args()

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    summary = []
    for name in sessions:
        print(f"=== {name} ===", flush=True)
        stream = build_eval_stream(
            SESSIONS_ROOT / name, apply_filter=not args.no_filter
        )
        split = build_trial_split(
            stream, val_ratio=SHARED.val_ratio, seed=SHARED.seed
        )
        out = DOCS_OUT / "out" / name
        man_path = write_split_artifacts(stream, split, out)
        man = json.loads(man_path.read_text(encoding="utf-8"))
        c = man["counts"]
        print(
            f"  subject={stream.subject_id} "
            f"trials={split.n_all} train={split.n_train} eval={split.n_eval} "
            f"ft_tr_win={c['ft_train']['n_windows']} "
            f"ft_va_win={c['ft_val']['n_windows']} "
            f"eval_win={c['eval_half']['n_windows']}",
            flush=True,
        )
        print(f"  wrote {man_path}", flush=True)
        summary.append(
            {
                "session": name,
                "subject_id": stream.subject_id,
                "n_all": split.n_all,
                "n_train": split.n_train,
                "n_eval": split.n_eval,
                "counts": c,
            }
        )
    out_sum = DOCS_OUT / "out" / "build_splits_summary.json"
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    out_sum.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"summary -> {out_sum}", flush=True)


if __name__ == "__main__":
    main()
