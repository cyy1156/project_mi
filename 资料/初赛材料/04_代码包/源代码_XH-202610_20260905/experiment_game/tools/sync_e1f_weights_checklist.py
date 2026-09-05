"""列出 E1f 四成员需从 5090 同步的 fold0 权重路径。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.e1f import E1fStackConfig  # noqa: E402


def main() -> None:
    stack = E1fStackConfig.load_json(
        _REPO / "experiment_game/config/e1f_four_member.json",
        repo_root=_REPO,
    )
    missing = stack.missing_paths(repo_root=_REPO)
    print("E1f four-member weight sync checklist:")
    paths = [stack.task_ckpt]
    for m in stack.members:
        paths.append(m.three_ckpt)
        if m.task_ckpt:
            paths.append(m.task_ckpt)
    # de-dupe preserve order
    seen = set()
    uniq = []
    for rel in paths:
        if not rel or rel in seen:
            continue
        seen.add(rel)
        uniq.append(rel)
    for rel in uniq:
        p = _REPO / rel if not Path(rel).is_absolute() else Path(rel)
        status = "OK" if p.is_file() else "MISSING"
        print(f"  [{status}] {rel}")
    missing = stack.missing_paths(repo_root=_REPO)
    if missing:
        print(f"\nMissing {len(missing)} file(s). Copy from 5090 machine to same relative paths.")
    else:
        print("\nAll weights present.")
    out = _REPO / "experiment_game/data/models/e1f_sync_check.json"
    out.write_text(
        json.dumps({"missing": missing, "paths": uniq}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
