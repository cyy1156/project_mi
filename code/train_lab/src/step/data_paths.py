"""数据目录解析：merged_2s / bci2a_* / stieger_*（含 1s 滑窗）。"""

from __future__ import annotations

from pathlib import Path

# step → src → train_lab → code
CODE_ROOT = Path(__file__).resolve().parents[3]
PRE_ROOT = CODE_ROOT / "preprocess_lab"

# data_tag -> (相对 out 的子目录, npy 前缀)
DATA_TAGS: dict[str, tuple[str, str]] = {
    "merged_2s": ("merged_2s", "merged"),
    "bci2a_2s": ("bci2a_2s", "bci2a"),
    "bci2a_4s": ("bci2a_4s", "bci2a"),
    "bci2a_1s": ("bci2a_1s", "bci2a"),
    "bci2a_2s_hop100": ("bci2a_2s_hop100", "bci2a"),
    "openbmi_2s_hop100": ("openbmi_2s_hop100", "openbmi"),
    "openbmi_2s_hop100_noz": ("openbmi_2s_hop100_noz", "openbmi"),
    "openbmi_2s_hop100_mu813": ("openbmi_2s_hop100_mu813", "openbmi"),
    "openbmi_2s_hop100_beta1330": ("openbmi_2s_hop100_beta1330", "openbmi"),
    "openbmi_2s_hop100_pf1000": ("openbmi_2s_hop100_pf1000", "openbmi"),
    "openbmi_2s_fixed_cue2to4_noz": ("openbmi_2s_fixed_cue2to4_noz", "openbmi"),
    "stieger_2s": ("stieger_2s", "stieger"),
    "stieger_4s": ("stieger_4s", "stieger"),
    "stieger_1s": ("stieger_1s", "stieger"),
    "stieger_2s_hop100": ("stieger_2s_hop100", "stieger"),
}


def resolve_data(data_tag: str) -> tuple[Path, str]:
    tag = data_tag.strip().lower()
    if tag not in DATA_TAGS:
        raise KeyError(f"未知 data_tag={data_tag!r}；可选: {list(DATA_TAGS)}")
    sub, prefix = DATA_TAGS[tag]
    return PRE_ROOT / "out" / sub, prefix
