"""Re-export official perf helpers with explicit names (IDE-friendly)."""
from __future__ import annotations

from typing import Any, Callable

from _official_load import load_official

_m = load_official("perf_loader")

apply_runtime_threads: Callable[[int], None] = _m.apply_runtime_threads
configure_cuda_backends: Callable[..., None] = _m.configure_cuda_backends
make_loader: Callable[..., Any] = _m.make_loader

__all__ = ["apply_runtime_threads", "configure_cuda_backends", "make_loader"]
