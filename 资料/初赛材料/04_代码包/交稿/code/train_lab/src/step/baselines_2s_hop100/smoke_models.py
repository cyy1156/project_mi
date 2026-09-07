"""对全部 2s/hop100 基线做一次前向冒烟（500 点 / bandpower / raw）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from feat_bandpower import raw_to_bandpower
from raw_time import squeeze_raw_2s

HERE = Path(__file__).resolve().parent


def _load_build(name: str):
    path = HERE / f"baseline_{name}.py"
    spec = importlib.util.spec_from_file_location(f"smoke_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.build_model


def main() -> None:
    x_time = torch.randn(4, 8, 500)
    x_feat = torch.from_numpy(raw_to_bandpower(np.random.randn(4, 1, 8, 500).astype(np.float32)))
    x_raw = torch.from_numpy(squeeze_raw_2s(np.random.randn(4, 1, 8, 500).astype(np.float32)))

    time_models = ("eegnet", "shallow", "deep", "eegtcnet", "conformer")
    feat_models = ("dbn", "gcbnet", "dgcnn")
    raw_models = ("dbn_raw", "gcbnet_raw", "dgcnn_raw")

    for name in time_models:
        build = _load_build(name)
        try:
            m = build(8, 500, 2, 0.5)
            y = m(x_time)
            if y.ndim > 2:
                y = y.reshape(y.shape[0], -1)
            assert y.shape == (4, 2), (name, y.shape)
            print("OK", name, y.shape)
        except Exception as e:
            print("FAIL", name, type(e).__name__, e)

    for name in feat_models:
        build = _load_build(name)
        try:
            m = build(8, 2, 2, 0.5)
            y = m(x_feat)
            if y.ndim > 2:
                y = y.reshape(y.shape[0], -1)
            assert y.shape == (4, 2), (name, y.shape)
            print("OK", name, y.shape)
        except Exception as e:
            print("FAIL", name, type(e).__name__, e)

    for name in raw_models:
        build = _load_build(name)
        try:
            m = build(8, 500, 2, 0.5)
            y = m(x_raw)
            if y.ndim > 2:
                y = y.reshape(y.shape[0], -1)
            assert y.shape == (4, 2), (name, y.shape)
            print("OK", name, y.shape)
        except Exception as e:
            print("FAIL", name, type(e).__name__, e)


if __name__ == "__main__":
    main()
