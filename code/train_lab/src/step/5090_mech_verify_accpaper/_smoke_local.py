"""方案 23 冒烟：四几何切片 + oracle 标记 + 模型前向。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms_registry import ARMS, assert_23_arm_flags  # noqa: E402
from feat_index import assert_default_map, segment_indices_for_geom  # noqa: E402
from geometry import GEOMETRIES, get_geom, make_masked, slice_pf1000  # noqa: E402
from model import MaskFutureDualExpert  # noqa: E402


def main() -> None:
    assert_23_arm_flags()
    assert_default_map()
    x = np.random.randn(4, 8, 1000).astype(np.float32)
    for gid in GEOMETRIES:
        g = get_geom(gid)
        xs = slice_pf1000(x, gid)
        assert xs.shape[-1] == g.total_pts, (gid, xs.shape)
        xt = torch.from_numpy(xs)
        xm = make_masked(xt, gid)
        if g.future_pts:
            assert (xm[..., -g.future_pts :] == 0).all()
        m = MaskFutureDualExpert(
            n_chans=8,
            n_times=g.total_pts,
            n_outputs=3,
            embed_dim=40,
            use_predictor=False,
            vis_pts=g.vis_pts,
        )
        out = m(xm, train_mode=False)
        assert out["p_cur"].shape == (4, 3)
        i_vis, i_fut = segment_indices_for_geom(gid, m.t_prime)
        print(f"geom {gid} T={g.total_pts} vis={len(i_vis)} fut={len(i_fut)} OK")

    arm_o = ARMS["O2s_f"]
    assert arm_o.oracle and arm_o.leak_eval_full
    arm_m = ARMS["O2s_m"]
    assert not arm_m.oracle
    arm_e1 = ARMS["E1"]
    m2 = MaskFutureDualExpert(
        n_chans=8,
        n_times=1000,
        n_outputs=3,
        embed_dim=40,
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        predictor_identity=True,
        vis_pts=600,
    )
    xf = torch.randn(2, 8, 1000)
    out2 = m2(xf, x_full=xf, train_mode=True)
    assert out2["p_final"].shape == (2, 3)
    print("oracle flags OK", arm_o.arm_id, arm_m.arm_id)
    print("E1 identity predictor OK", arm_e1.arm_id)
    print("scheme23 smoke ALL OK")


if __name__ == "__main__":
    main()
