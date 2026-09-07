"""RingBuffer push/取窗回归：1-D 单样本不得被广播成 C 行（旧 bug：_n 按 8 倍增长、87.5% 重复行）。"""

from __future__ import annotations

import unittest

import numpy as np

from experiment_game.experiment.inference_v2 import RingBuffer


class TestRingBufferPush(unittest.TestCase):
    def test_push_1d_writes_one_row(self) -> None:
        # 旧 _pull_loop 的调用形态：逐样本 1-D (8,)
        buf = RingBuffer()
        for i in range(100):
            buf.push(np.full(8, float(i)))
        self.assertEqual(buf._n, 100)
        tail = buf.snapshot_tail(0.4, t_now_lsl=1000.0)  # 100 行
        self.assertIsNotNone(tail)
        self.assertEqual(tail.shape, (100, 8))
        self.assertEqual(np.unique(tail[:, 0]).size, 100)  # 无重复行
        np.testing.assert_array_equal(tail[:, 0], np.arange(100, dtype=np.float64))

    def test_push_pull_loop_path(self) -> None:
        # 与 _pull_loop 一致：设备序直接 push，不做 reorder
        buf = RingBuffer()
        for i in range(50):
            row = np.full(8, float(i))
            buf.push(row.reshape(1, -1))
        self.assertEqual(buf._n, 50)

    def test_push_2d_block(self) -> None:
        buf = RingBuffer()
        block = np.arange(50 * 8, dtype=np.float64).reshape(50, 8)
        buf.push(block)
        self.assertEqual(buf._n, 50)
        tail = buf.snapshot_tail(0.2, t_now_lsl=1000.0)
        np.testing.assert_array_equal(tail, block)

    def test_push_wraparound_1d(self) -> None:
        # 1-D 逐样本写满并环绕一圈
        buf = RingBuffer(capacity_s=0.04)  # cap = 10 行
        for i in range(25):
            buf.push(np.full(8, float(i)))
        self.assertEqual(buf._n, 25)
        tail = buf.snapshot_tail(0.04, t_now_lsl=1000.0)  # 10 行
        np.testing.assert_array_equal(tail[:, 0], np.arange(15, 25, dtype=np.float64))

    def test_window_time_alignment(self) -> None:
        buf = RingBuffer()
        n = 2500
        buf.push(np.stack([np.arange(n, dtype=np.float64)] * 8, axis=1))
        # now == t_end → 尾部 750 行
        w = buf.window_ending_at(100.0, 750, t_now_lsl=100.0)
        self.assertEqual(w.shape, (750, 8))
        self.assertEqual(w[0, 0], n - 750)
        self.assertEqual(w[-1, 0], n - 1)
        # t_end 早 1s → 窗口回退 250 行
        w2 = buf.window_ending_at(99.0, 750, t_now_lsl=100.0)
        self.assertEqual(w2[0, 0], n - 750 - 250)
        self.assertEqual(w2[-1, 0], n - 251)
        # 判定时刻未到 → None
        self.assertIsNone(buf.window_ending_at(101.0, 750, t_now_lsl=100.0))
        # 数据不足 → None
        self.assertIsNone(buf.window_ending_at(100.0, 750, t_now_lsl=200.0))

    def test_stale_watchdog(self) -> None:
        import time

        from experiment_game.experiment.inference_v2 import EEG_STALE_TIMEOUT_S

        buf = RingBuffer()
        # 冷启动：尚无样本时也开始计时（不再返回 None）
        self.assertIsNotNone(buf.last_push_age_s())
        self.assertFalse(buf.is_stale(timeout_s=10.0))
        time.sleep(0.08)
        self.assertTrue(buf.is_stale(timeout_s=0.05))
        st0 = buf.stale_status(timeout_s=0.05)
        self.assertIsNotNone(st0)
        self.assertEqual(st0["n_samples"], 0)

        buf = RingBuffer()
        buf.push(np.zeros(8))
        self.assertIsNotNone(buf.last_push_age_s())
        self.assertFalse(buf.is_stale(timeout_s=10.0))
        time.sleep(0.08)
        self.assertTrue(buf.is_stale(timeout_s=0.05))
        st = buf.stale_status(timeout_s=0.05)
        self.assertIsNotNone(st)
        self.assertGreaterEqual(st["age_s"], 0.05)
        self.assertEqual(st["n_samples"], 1)
        # 新样本后恢复
        buf.push(np.ones(8))
        self.assertFalse(buf.is_stale(timeout_s=0.05))
        self.assertGreaterEqual(EEG_STALE_TIMEOUT_S, 1.0)


if __name__ == "__main__":
    unittest.main()
