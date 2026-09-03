"""OpenBMI 8ch E1f 权重加载；标签轴 OpenBMI(Rest,L,R) → 挑战杯(L,R,Rest)。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

# challenge class i <- openbmi class PERM[i]
OPENBMI_TO_CHALLENGE_PERM = (1, 2, 0)

REPO = Path(__file__).resolve().parents[5]
DEFAULT_STACK = REPO / "experiment_game" / "config" / "e1f_four_member.json"

# Exp34 成员名 → OpenBMI stack 名
NAME_MAP = {
    "shallow": "shallow",
    "shallow_b": "t_shallow",
    "t_shallow": "t_shallow",
    "eegnet": "eegnet",
    "conformer": "conformer",
}


def load_e1f_stack(path: Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_STACK
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_three_ckpt(member_name: str, stack: dict | None = None) -> Path:
    stack = stack or load_e1f_stack()
    key = NAME_MAP.get(member_name, member_name)
    for m in stack["members"]:
        if m["name"] == key:
            rel = m["three_ckpt"]
            path = REPO / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
    raise KeyError(f"stack 中无成员 {member_name}/{key}")


def remap_classifier_out3(state: dict) -> dict:
    """把 out=3 的分类头行顺序从 OpenBMI 换到挑战杯。"""
    perm = list(OPENBMI_TO_CHALLENGE_PERM)
    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        kl = k.lower()
        is_head = any(
            s in kl for s in ("final", "classif", "fc", "conv_classifier", "linear")
        )
        if is_head and v.ndim >= 1 and int(v.shape[0]) == 3:
            out[k] = v[perm].clone()
        else:
            out[k] = v
    return out


def load_init_state(
    member_name: str,
    *,
    stack_path: Path | None = None,
    remap_labels: bool = True,
) -> dict:
    ckpt = resolve_three_ckpt(member_name, load_e1f_stack(stack_path) if stack_path else None)
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if remap_labels:
        state = remap_classifier_out3(state)
    return state


def apply_init_weights(model: nn.Module, state: dict, *, strict: bool = False) -> list[str]:
    missing, unexpected = model.load_state_dict(state, strict=strict)
    msgs = []
    if missing:
        msgs.append(f"missing={list(missing)[:8]}")
    if unexpected:
        msgs.append(f"unexpected={list(unexpected)[:8]}")
    return msgs
