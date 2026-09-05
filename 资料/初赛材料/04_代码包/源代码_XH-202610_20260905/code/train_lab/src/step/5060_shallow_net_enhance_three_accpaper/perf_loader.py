"""OpenBMI 训练性能：DataLoader 多进程 + GPU 传输 / AMP 开关。

针对本机常见配置（如 RTX 5060 8GB + 16GB 内存 + 多核 CPU）的默认值；
可用 SharedTrainHP / CLI 覆盖。
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def apply_runtime_threads(torch_num_threads: int) -> None:
    """限制主进程 BLAS/CPU 算子线程，给 DataLoader worker 留核。"""
    n = int(torch_num_threads)
    if n > 0:
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(max(1, min(4, n // 2)))
        except RuntimeError:
            # 已初始化后不可改 interop
            pass
    # 减轻 OpenMP 过度订阅
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, n) if n > 0 else "1"))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, n) if n > 0 else "1"))


def configure_cuda_backends(*, cudnn_benchmark: bool, deterministic: bool) -> None:
    if not torch.cuda.is_available():
        return
    # benchmark 与 deterministic 互斥偏好：要吞吐时开 benchmark
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic) and not cudnn_benchmark
    # Fast：TF32 加速；Repro：关 TF32 降低非确定性
    allow_tf32 = not bool(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    sampler: Sampler | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory) and torch.cuda.is_available(),
        drop_last=bool(drop_last),
    )
    nw = int(num_workers)
    if nw > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
        # Windows spawn：避免 fork 相关默认
        kwargs["multiprocessing_context"] = None
    return DataLoader(**kwargs)
