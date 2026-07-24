"""events.jsonl 写入。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from pylsl import local_clock


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: Optional[TextIO] = open(self.path, "a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> Dict[str, Any]:
        row: Dict[str, Any] = {"t_lsl": float(local_clock()), "event": event}
        for k, v in fields.items():
            if v is not None:
                row[k] = v
        assert self._fp is not None
        self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fp.flush()
        return row

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
