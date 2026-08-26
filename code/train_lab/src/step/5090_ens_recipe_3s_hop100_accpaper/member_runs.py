"""成员 run 路径解析（方案 26/28 共用）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from s26_config import DEFAULT_MEMBERS, MemberRuns

_MEMBER_ALIASES: dict[str, str] = {
    "s": "shallow",
    "shallow": "shallow",
    "t": "t_shallow",
    "t_shallow": "t_shallow",
    "t-shallow": "t_shallow",
    "e": "eegnet",
    "eegnet": "eegnet",
    "c": "conformer",
    "conformer": "conformer",
}


def normalize_member_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key not in _MEMBER_ALIASES:
        allowed = sorted(set(_MEMBER_ALIASES.values()))
        raise ValueError(f"unknown member {name!r}; allowed: {allowed}")
    return _MEMBER_ALIASES[key]


def parse_member_names(members: str | None) -> list[str] | None:
    if members is None:
        return None
    names = [normalize_member_name(x) for x in members.split(",") if x.strip()]
    if not names:
        raise ValueError("--members must list at least one member")
    return names


def resolve_run_dir(
    member: str,
    *,
    pool: MemberRuns = DEFAULT_MEMBERS,
    overrides: dict[str, Path | None] | None = None,
) -> Path:
    overrides = overrides or {}
    if member in overrides and overrides[member] is not None:
        return Path(overrides[member])
    try:
        return pool.as_dict()[member]
    except KeyError as exc:
        raise KeyError(f"unknown member {member!r}") from exc


def member_run_dirs(
    names: list[str],
    *,
    pool: MemberRuns = DEFAULT_MEMBERS,
    overrides: dict[str, Path | None] | None = None,
) -> list[Path]:
    return [resolve_run_dir(n, pool=pool, overrides=overrides) for n in names]


def default_e1_member_names(*, four_member: bool) -> list[str]:
    if four_member:
        return ["shallow", "t_shallow", "eegnet", "conformer"]
    return ["shallow", "eegnet", "conformer"]


def parse_runs_from_args(args: argparse.Namespace) -> list[Path]:
    """replay_e1.py：--members 优先，否则沿用 E1 三/四成员默认。"""
    overrides = {
        "shallow": getattr(args, "shallow_run", None),
        "eegnet": getattr(args, "eegnet_run", None),
        "conformer": getattr(args, "conformer_run", None),
        "t_shallow": getattr(args, "t_shallow_run", None),
    }
    if args.members:
        names = parse_member_names(args.members)
        assert names is not None
        return member_run_dirs(names, overrides=overrides)
    four = bool(getattr(args, "four_member", False))
    return member_run_dirs(default_e1_member_names(four_member=four), overrides=overrides)
