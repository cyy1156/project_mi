"""可插拔基线 / 自研模型注册表。"""

from .registry import (
    BUILTIN_NAMES,
    ModelSpec,
    build_model,
    get_spec,
    list_models,
    register,
)

__all__ = [
    "BUILTIN_NAMES",
    "ModelSpec",
    "build_model",
    "get_spec",
    "list_models",
    "register",
]
