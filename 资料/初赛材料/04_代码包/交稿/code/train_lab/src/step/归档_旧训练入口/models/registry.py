"""
模型注册表：基线 + 自研统一 build_model(name, n_chans, n_times, n_outputs, **hp)。

契约：输入 (B, C, T)，输出 logits (B, n_outputs)；使用各模型原生分类头。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn as nn
from braindecode.models import (
    Deep4Net,
    EEGConformer,
    EEGNet,
    EEGTCNet,
    ShallowFBCSPNet,
)

BuilderFn = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # cnn | tcn | transformer | custom
    builder: BuilderFn
    input_layout: str = "bct"
    needs_n_times: bool = True
    default_hparams: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {}


def register(
    name: str,
    family: str,
    builder: BuilderFn,
    *,
    input_layout: str = "bct",
    needs_n_times: bool = True,
    default_hparams: dict[str, Any] | None = None,
    search_space: dict[str, list[Any]] | None = None,
    notes: str = "",
) -> None:
    key = name.strip().lower()
    if key in REGISTRY:
        raise ValueError(f"模型已注册: {key}")
    REGISTRY[key] = ModelSpec(
        name=key,
        family=family,
        builder=builder,
        input_layout=input_layout,
        needs_n_times=needs_n_times,
        default_hparams=dict(default_hparams or {}),
        search_space=dict(search_space or {}),
        notes=notes,
    )


def get_spec(name: str) -> ModelSpec:
    key = name.strip().lower()
    if key not in REGISTRY:
        raise KeyError(f"未知模型: {name!r}；可选: {list_models()}")
    return REGISTRY[key]


def list_models() -> list[str]:
    return sorted(REGISTRY.keys())


def build_model(
    name: str,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    **hp: Any,
) -> nn.Module:
    spec = get_spec(name)
    if spec.input_layout != "bct":
        raise ValueError(f"{spec.name}: 仅支持 input_layout=bct，收到 {spec.input_layout}")
    merged = {**spec.default_hparams, **{k: v for k, v in hp.items() if v is not None}}
    kwargs: dict[str, Any] = {
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        **merged,
    }
    if spec.needs_n_times:
        kwargs["n_times"] = n_times
    return spec.builder(**kwargs)


# ----- builders（只取本模型关心的键） -----


def _pop_shared(hp: dict[str, Any]) -> dict[str, Any]:
    """训练壳传入的优化器键不进网络构造。"""
    out = dict(hp)
    for k in (
        "lr",
        "weight_decay",
        "patience",
        "max_epochs",
        "batch_train",
        "batch_eval",
        "n_folds",
        "val_ratio",
        "seed",
        "out_dir",
        "task_kfold_dir",
        "model_name",
        "data_tag",
        "data_prefix",
        "init_from_task",
        "freeze_backbone",
        "classifier",
        "weight_transfer",
    ):
        out.pop(k, None)
    return out


def build_eegnet(*, n_chans: int, n_outputs: int, n_times: int, **hp: Any) -> nn.Module:
    hp = _pop_shared(hp)
    return EEGNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        F1=int(hp.pop("F1", hp.pop("f1", 8))),
        D=int(hp.pop("D", hp.pop("d", 2))),
        F2=int(hp.pop("F2", hp.pop("f2", 16))),
        drop_prob=float(hp.pop("drop_prob", 0.5)),
        **{k: v for k, v in hp.items() if k in ("kernel_length",)},
    )


def build_shallow(*, n_chans: int, n_outputs: int, n_times: int, **hp: Any) -> nn.Module:
    hp = _pop_shared(hp)
    # 丢弃 EEGNet 专有键，避免误传
    for k in ("F1", "f1", "D", "d", "F2", "f2"):
        hp.pop(k, None)
    kwargs = {
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "n_times": n_times,
        "drop_prob": float(hp.pop("drop_prob", 0.5)),
    }
    for k in (
        "n_filters_time",
        "filter_time_length",
        "n_filters_spat",
        "pool_time_length",
        "pool_time_stride",
    ):
        if k in hp:
            kwargs[k] = hp.pop(k)
    return ShallowFBCSPNet(**kwargs)


def build_deep(*, n_chans: int, n_outputs: int, n_times: int, **hp: Any) -> nn.Module:
    hp = _pop_shared(hp)
    for k in ("F1", "f1", "D", "d", "F2", "f2"):
        hp.pop(k, None)
    kwargs = {
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "n_times": n_times,
        "drop_prob": float(hp.pop("drop_prob", 0.5)),
    }
    for k in (
        "n_filters_time",
        "n_filters_spat",
        "filter_time_length",
        "n_filters_2",
        "n_filters_3",
        "n_filters_4",
    ):
        if k in hp:
            kwargs[k] = hp.pop(k)
    return Deep4Net(**kwargs)


def build_eegtcnet(*, n_chans: int, n_outputs: int, n_times: int, **hp: Any) -> nn.Module:
    hp = _pop_shared(hp)
    for k in ("F1", "f1", "D", "d", "F2", "f2"):
        hp.pop(k, None)
    kwargs = {
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "n_times": n_times,
        "drop_prob": float(hp.pop("drop_prob", 0.5)),
    }
    for k in ("depth", "filters", "kernel_size", "kern_length", "filter_1", "depth_multiplier"):
        if k in hp:
            kwargs[k] = hp.pop(k)
    return EEGTCNet(**kwargs)


def build_conformer(*, n_chans: int, n_outputs: int, n_times: int, **hp: Any) -> nn.Module:
    hp = _pop_shared(hp)
    for k in ("F1", "f1", "D", "d", "F2", "f2"):
        hp.pop(k, None)
    kwargs: dict[str, Any] = {
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "n_times": n_times,
        "final_fc_length": hp.pop("final_fc_length", "auto"),
        "drop_prob": float(hp.pop("drop_prob", 0.5)),
    }
    for k in (
        "n_filters_time",
        "filter_time_length",
        "pool_time_length",
        "pool_time_stride",
        "num_layers",
        "num_heads",
        "att_drop_prob",
    ):
        if k in hp:
            kwargs[k] = hp.pop(k)
    return EEGConformer(**kwargs)


# ----- 内置登记 -----

_SHARED_SEARCH = {
    "lr": [5e-4, 7e-4, 1e-3, 1.5e-3],
    "weight_decay": [5e-5, 1e-4, 2e-4],
    "drop_prob": [0.40, 0.50, 0.55, 0.60],
    "patience": [15, 18, 20],
}

register(
    name="eegnet",
    family="cnn",
    builder=build_eegnet,
    default_hparams={"F1": 8, "D": 2, "F2": 16, "drop_prob": 0.5},
    search_space=dict(_SHARED_SEARCH),
    notes="braindecode EEGNet；原生 Conv 头",
)
register(
    name="shallow",
    family="cnn",
    builder=build_shallow,
    default_hparams={"drop_prob": 0.5},
    search_space=dict(_SHARED_SEARCH),
    notes="ShallowFBCSPNet；勿套用 F1/D/F2",
)
register(
    name="deep",
    family="cnn",
    builder=build_deep,
    default_hparams={"drop_prob": 0.5},
    search_space=dict(_SHARED_SEARCH),
    notes="Deep4Net",
)
register(
    name="eegtcnet",
    family="tcn",
    builder=build_eegtcnet,
    default_hparams={"drop_prob": 0.5},
    search_space={
        **_SHARED_SEARCH,
        "depth": [2, 3],
        "filters": [8, 12],
    },
    notes="EEGTCNet（计划中的 EEGNet+TCN）；禁止裸 TCN",
)
register(
    name="conformer",
    family="transformer",
    builder=build_conformer,
    default_hparams={
        "drop_prob": 0.5,
        "final_fc_length": "auto",
        "num_layers": 2,
        "num_heads": 10,
        "att_drop_prob": 0.5,
    },
    search_space={
        **_SHARED_SEARCH,
        "num_layers": [2, 4, 6],
        "att_drop_prob": [0.3, 0.5],
    },
    notes="EEGConformer；用本族超参网格",
)

BUILTIN_NAMES = ("eegnet", "shallow", "deep", "eegtcnet", "conformer")

# 触发自研登记
try:
    from . import custom as _custom  # noqa: F401
except Exception:
    pass
