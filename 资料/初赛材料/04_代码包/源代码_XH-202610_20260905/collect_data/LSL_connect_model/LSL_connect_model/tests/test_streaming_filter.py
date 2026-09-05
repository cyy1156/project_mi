"""回归测试：采集端因果流式滤波（StreamingEegFilter）。

针对「按批（≤25 样本）零相位滤波产生块边界伪迹」的修复：
1. 任意变批长（1–25）流式输出与整段一次性因果滤波逐位相等（状态跨批衔接）；
2. 通带正弦包络平坦，无批边界幅度调制（旧分块路径作对照）；
3. reset 后行为可复现（重连后重新起算）；
4. from_config 生效：50Hz 陷波抑制、10Hz 通带通过、DC 被高通滤除；
5. 边界形状（空批、一维）安全返回。
"""

import unittest

import numpy as np
from scipy.signal import hilbert, sosfilt

from lsl_connect.preprocessing import (
    PreprocessConfig,
    StreamingEegFilter,
    apply_eeg_filters,
)

FS = 250
N_CH = 8


def _make_signal(seconds: float = 20.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    t = np.arange(n) / FS
    sig = np.empty((N_CH, n), dtype=np.float64)
    for c in range(N_CH):
        sig[c] = (
            10.0 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            + 4.0 * np.sin(2 * np.pi * 21 * t)
            + 3.0 * np.sin(2 * np.pi * 50 * t)  # 工频，应被陷波抑制
            + rng.normal(0.0, 5.0, n)
        )
    return sig


def _whole_causal_reference(sig: np.ndarray) -> np.ndarray:
    """整段一次性因果滤波（与流式相同的 sos 与首样本播种），作为金标准。"""
    f = StreamingEegFilter()
    out = np.empty_like(sig, dtype=np.float64)
    for c in range(sig.shape[0]):
        zi = f._zi_unit * sig[c, 0]
        out[c], _ = sosfilt(f._sos, sig[c], zi=zi)
    return out


def _feed_streaming(
    f: StreamingEegFilter, sig: np.ndarray, rng: np.random.Generator, max_chunk: int = 25
) -> np.ndarray:
    """用随机批长（1..max_chunk）喂数据，模拟采集循环的变长 fetch。"""
    outs = []
    i, n = 0, sig.shape[1]
    while i < n:
        k = int(rng.integers(1, max_chunk + 1))
        outs.append(f.process(sig[:, i : i + k]))
        i += k
    return np.concatenate(outs, axis=1)


class TestStreamingEegFilter(unittest.TestCase):
    def test_streaming_equals_whole_signal(self):
        """变批长流式输出必须与整段因果滤波逐位相等。"""
        sig = _make_signal()
        ref = _whole_causal_reference(sig)
        f = StreamingEegFilter()
        out = _feed_streaming(f, sig, np.random.default_rng(42))
        np.testing.assert_array_equal(out, ref)

    def test_single_sample_batches(self):
        """逐样本（批长=1）喂数也必须与整段一致。"""
        sig = _make_signal(seconds=5.0)
        ref = _whole_causal_reference(sig)
        f = StreamingEegFilter()
        out = np.concatenate([f.process(sig[:, i : i + 1]) for i in range(sig.shape[1])], axis=1)
        np.testing.assert_array_equal(out, ref)

    def test_no_chunk_boundary_artifact(self):
        """通带 10Hz 正弦：流式输出包络应平坦，无 25 样本周期的幅度调制。"""
        n = FS * 30
        t = np.arange(n) / FS
        sig = np.tile(10.0 * np.sin(2 * np.pi * 10 * t), (N_CH, 1))

        f = StreamingEegFilter()
        out = _feed_streaming(f, sig, np.random.default_rng(7))
        settled = np.abs(hilbert(out[0]))[2 * FS :]
        stream_rel = settled.std() / settled.mean()
        self.assertLess(
            stream_rel,
            0.02,
            f"流式输出包络波动过大（相对std={stream_rel:.4f}），疑似边界伪迹",
        )

        # 旧分块零相位路径对照：同样的批切分应产生明显包络调制
        old = sig.copy()
        for i in range(0, n, 25):
            apply_eeg_filters(old[:, i : i + 25], PreprocessConfig())
        old_env = np.abs(hilbert(old[0]))[2 * FS :]
        old_rel = old_env.std() / max(1e-9, old_env.mean())
        self.assertGreater(
            old_rel,
            5 * stream_rel,
            "旧分块路径未表现出边界伪迹——对照前提可能不成立，需人工复核",
        )

    def test_reset_reproduces_output(self):
        """reset 后相同输入、相同切分应复现完全一致的输出（重连语义）。"""
        sig = _make_signal(seconds=4.0)
        f = StreamingEegFilter()
        out1 = _feed_streaming(f, sig, np.random.default_rng(1))
        f.reset()
        out2 = _feed_streaming(f, sig, np.random.default_rng(1))
        np.testing.assert_array_equal(out1, out2)

    def test_from_config_frequency_response(self):
        """from_config：10Hz 通带近无损、50Hz 被陷波抑制、DC 被高通滤除。"""
        f = StreamingEegFilter.from_config(PreprocessConfig(sample_rate=FS))
        n = FS * 20
        t = np.arange(n) / FS

        sig10 = np.ones((1, n)) * (10.0 * np.sin(2 * np.pi * 10 * t))
        gain10 = np.abs(hilbert(f.process(sig10)[0]))[5 * FS :].mean() / 10.0
        self.assertGreater(gain10, 0.85)
        self.assertLess(gain10, 1.15)

        f.reset()
        sig50 = np.ones((1, n)) * (10.0 * np.sin(2 * np.pi * 50 * t))
        gain50 = np.abs(hilbert(f.process(sig50)[0]))[5 * FS :].mean() / 10.0
        self.assertLess(gain50, 0.15)

        f.reset()
        dc = np.full((1, n), 50.0)
        self.assertLess(np.abs(f.process(dc)).max(), 1.0, "DC 未被高通滤除")

    def test_edge_shapes(self):
        """空批与一维输入安全返回、不误建状态。"""
        f = StreamingEegFilter()
        out = f.process(np.empty((N_CH, 0)))
        self.assertEqual(out.shape, (N_CH, 0))
        out = f.process(np.ones(10))
        self.assertEqual(out.ndim, 1)
        self.assertIsNone(f._zi, "一维/空输入不应创建滤波状态")


if __name__ == "__main__":
    unittest.main()
