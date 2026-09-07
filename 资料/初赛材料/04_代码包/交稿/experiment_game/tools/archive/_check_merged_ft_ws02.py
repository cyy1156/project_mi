"""合并 FT 后：ws02 预测分布 sanity check。"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from experiment_game.tools.ft_subject_from_v3 import _build_session_windows, _pred_distribution
from adapt_engine.registry import load_head

WS02 = _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537"
CKPT = _REPO / "experiment_game/data/models/fnz/best_three.pt"
LABELS = {0: "Rest", 1: "Left", 2: "Right"}


def main():
    ds = _build_session_windows(WS02)
    X, y = ds["X"], ds["y_three"]
    entry = load_head(CKPT, n_chans=8, n_times=750, device="cpu")
    d = _pred_distribution(entry.model, X, "cpu")
    pred = []
    import torch
    with torch.no_grad():
        for s in range(0, len(X), 64):
            xb = torch.from_numpy(X[s : s + 64])
            logits = entry.model(xb)
            if logits.dim() == 3:
                logits = logits.reshape(logits.shape[0], -1)
            pred.extend(logits.argmax(-1).tolist())
    pred = np.array(pred)
    acc = float((pred == y).mean())
    pc = Counter(pred.tolist())
    print(f"ws02 windows={len(X)} acc={acc:.3f}")
    print(f"pred: {{ {', '.join(f'{LABELS[k]}:{v}' for k,v in sorted(pc.items()))} }}")
    print(f"mean_p Rest/L/R={[round(x,3) for x in d['mean_p']]}")
    print(f"max_class_frac={d['max_class_frac']:.3f}")


if __name__ == "__main__":
    main()
