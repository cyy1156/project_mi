"""官方 59 EEG ∩ OpenBMI 62 导 → 45 通道（官方顺序）。

实测名称交集为 **45**（非方案草稿里的 51）。CPz 不在交集（官方用 Pz；OpenBMI 无 CPz）。
"""

from __future__ import annotations

# 官方 59 EEG 名序（challenge_mi select_eeg59 / pkl 前 59）
OFFICIAL_59 = (
    "Fpz",
    "Fp1",
    "Fp2",
    "AF3",
    "AF4",
    "AF7",
    "AF8",
    "Fz",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "FCz",
    "FC1",
    "FC2",
    "FC3",
    "FC4",
    "FC5",
    "FC6",
    "FT7",
    "FT8",
    "Cz",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "T7",
    "T8",
    "CP1",
    "CP2",
    "CP3",
    "CP4",
    "CP5",
    "CP6",
    "TP7",
    "TP8",
    "Pz",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "POz",
    "PO3",
    "PO4",
    "PO5",
    "PO6",
    "PO7",
    "PO8",
    "Oz",
    "O1",
    "O2",
)

# 官方序 ∩ OpenBMI 实测 62 导（大小写不敏感匹配后固定此序）
INTERSECT_45 = (
    "Fp1",
    "Fp2",
    "AF3",
    "AF4",
    "AF7",
    "AF8",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC3",
    "FC4",
    "FC5",
    "FC6",
    "Cz",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "T7",
    "T8",
    "CP1",
    "CP2",
    "CP3",
    "CP4",
    "CP5",
    "CP6",
    "TP7",
    "TP8",
    "Pz",
    "P3",
    "P4",
    "P7",
    "P8",
    "POz",
    "PO3",
    "PO4",
    "Oz",
    "O1",
    "O2",
)

PROTOCOL_45 = "challenge_mi_3s_45ch"
PROTOCOL_OPENBMI_45 = "openbmi_3s_fixed_45ch"


def _norm(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def indices_in_names(ch_names: list[str] | tuple[str, ...], order: tuple[str, ...] = INTERSECT_45) -> list[int]:
    idx_map = {_norm(n): i for i, n in enumerate(ch_names)}
    out: list[int] = []
    for slot in order:
        key = _norm(slot)
        if key not in idx_map:
            raise KeyError(f"缺少通道 {slot} in {list(ch_names)[:8]}…")
        out.append(int(idx_map[key]))
    return out


def indices_59_to_45() -> list[int]:
    return indices_in_names(OFFICIAL_59, INTERSECT_45)
