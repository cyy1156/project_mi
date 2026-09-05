"""方案 26 · 导入与锚点路径校验（不训练）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG24 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
for p in (HERE, PKG24):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from s26_config import DEFAULT_MEMBERS  # noqa: E402
from patch_recipe import install_recipe  # noqa: E402
from fast_kan import FastKANClassifier  # noqa: E402
from fe_bandpower_3s import bandpower_lateral_features  # noqa: E402
from fe_riemannian import cov_tangent_features  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    m = DEFAULT_MEMBERS
    for name, path in m.as_dict().items():
        assert path.is_dir(), f"missing anchor {name}: {path}"
        ckpts = list(path.glob("fold*/best_three.pt"))
        assert ckpts, f"no ckpt under {path}"
    install_recipe("R1")
    x = np.random.randn(2, 8, 750).astype(np.float32)
    bp = bandpower_lateral_features(x)
    assert bp.shape == (2, 24), bp.shape
    ri = cov_tangent_features(x)
    assert ri.shape == (2, 36), ri.shape
    kan = FastKANClassifier(24, 3)
    import torch
    out = kan(torch.from_numpy(bp))
    assert out.shape == (2, 3)
    print(f"verify_imports: OK anchors={list(m.as_dict())}")


if __name__ == "__main__":
    main()
