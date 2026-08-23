"""快速校验本包导入与锚点权重路径（不跑训练）。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from domain_aug import aug_config_g1, apply_domain_aug_np  # noqa: E402
from s25_weights import resolve_anchor_s3_run, resolve_weight_run  # noqa: E402


def main() -> None:
    cfg = aug_config_g1()
    x = apply_domain_aug_np(
        __import__("numpy").zeros((8, 750), dtype="float32"),
        cfg,
        seed=0,
        index=0,
    )
    assert x.shape == (8, 750)
    anchor = resolve_anchor_s3_run()
    assert anchor.is_dir(), anchor
    print(f"verify_imports: OK anchor={anchor}")


if __name__ == "__main__":
    main()
