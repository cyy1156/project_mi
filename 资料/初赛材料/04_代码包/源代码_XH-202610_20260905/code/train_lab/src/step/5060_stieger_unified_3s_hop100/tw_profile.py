"""窗长配置：2s / 3s 联动数据、权重、结果目录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TwProfile:
    tw: str
    win_sec: float
    n_times: int
    hop_sec: float
    data_tag: str
    data_dir: Path
    openbmi_weight_root: Path
    openbmi_model_pkg: str
    openbmi_dataset_key: str
    openbmi_shallow_run: str
    ft_weight_root: Path
    arm_zeroshot: str
    arm_ft_half: str
    zeroshot_subdir: str
    ft_half_subdir: str


def build_profiles(
    *,
    code_root: Path,
    train_lab: Path,
    docs_10: Path,
) -> dict[str, TwProfile]:
  preprocess_out = code_root / "preprocess_lab" / "out"
  results = docs_10 / "results"
  return {
      "3s": TwProfile(
          tw="3s",
          win_sec=3.0,
          n_times=750,
          hop_sec=0.1,
          data_tag="stieger_3s_hop100",
          data_dir=preprocess_out / "stieger_3s_hop100",
          openbmi_weight_root=train_lab
          / "out"
          / "5060_baseline_openbmi_3s_hop100_accpaper",
          openbmi_model_pkg="shallow_openbmi_3s_hop100_balbatch_accpaper",
          openbmi_dataset_key="openbmi_3s_hop100",
          openbmi_shallow_run="run_20260821_190504",
          ft_weight_root=train_lab / "out" / "stieger_ft_3s_hop100_accpaper",
          arm_zeroshot="S10-01",
          arm_ft_half="S10-02",
          zeroshot_subdir="S10-01_zeroshot",
          ft_half_subdir="S10-02_ft_half",
      ),
      "2s": TwProfile(
          tw="2s",
          win_sec=2.0,
          n_times=500,
          hop_sec=0.1,
          data_tag="stieger_2s_hop100",
          data_dir=preprocess_out / "stieger_2s_hop100",
          openbmi_weight_root=train_lab
          / "out"
          / "5060_baseline_openbmi_2s_hop100_accpaper",
          openbmi_model_pkg="shallow_openbmi_2s_hop100_balbatch_accpaper",
          openbmi_dataset_key="openbmi_2s_hop100",
          openbmi_shallow_run="run_20260807_135828",
          ft_weight_root=train_lab / "out" / "stieger_ft_2s_hop100_accpaper",
          arm_zeroshot="S10-01b",
          arm_ft_half="S10-02b",
          zeroshot_subdir="S10-01b_zeroshot_2s",
          ft_half_subdir="S10-02b_ft_half_2s",
      ),
  }
