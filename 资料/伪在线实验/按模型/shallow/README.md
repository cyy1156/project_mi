# shallow · 伪在线实验归类

> 模型：`braindecode.ShallowFBCSPNet`  
> 当前部署（2s）：[`../../04_旁路_OpenBMI权重_游戏零样本与门控/`](../../04_旁路_OpenBMI权重_游戏零样本与门控/)  
> **新臂（3s · Stieger）**：[07 · 5060](../../07_旁路_OpenBMI_3s滑窗_Stieger零样本/) · [08 · 5070](../../08_旁路_OpenBMI_3s滑窗_Stieger零样本_5070/)

## 按协议臂

| 臂 | Tw | 权重域 | 角色 | 本模型结果入口 |
|----|-----|--------|------|----------------|
| [01 零样本](../../01_不微调_零样本/) | 2s | BCI2a | 历史冻结 | `results/20260805_005221_shallow_pseudo_online/` |
| [02 前半FT](../../02_微调_前半训后半评/) | 2s | BCI2a→游戏 | 历史汇总 | `results/20260805_013452_shallow_game_ft_half/` |
| [03 质量门控](../../03_旁路_teachable质量门控/) | 2s | BCI2a / OpenBMI | P1/P2 | `results/*shallow*P*` |
| [04 OpenBMI 零样本+门控](../../04_旁路_OpenBMI权重_游戏零样本与门控/) | 2s | OpenBMI 正式 | **现行部署** | `results/20260809_173723_shallow_openbmi_Q0Q1/` |
| [05 OpenBMI 前半FT](../../05_旁路_OpenBMI_前半微调后半评/) | 2s | OpenBMI→游戏 | 已结案 | `results/20260809_174914_shallow_openbmi_game_ft_half/` |
| [06 FT+后半门控](../../06_旁路_OpenBMI_前半FT后半门控/) | 2s | 05 FT | 已结案 | `results/20260809_180505_shallow_ft_gated/` |
| [**07 Stieger 复现 01–06**](../../07_旁路_OpenBMI_3s滑窗_Stieger零样本/) | **3s** | **S3 · 5060** | **新开** | S07-01…06 见该臂方案 |
| [**08 Stieger · 5070**](../../08_旁路_OpenBMI_3s滑窗_Stieger零样本_5070/) | **3s** | **S3 · 5070** | **新开** | S08-01…06；协议同 07 |

## 离线 Acc_paper 锚点（OpenBMI · 训练侧）

| Tw | Task | Three | run |
|----|------|-------|-----|
| 2s（正式） | 0.6941±0.0349 | 0.5404±0.0256 | `run_20260807_135828` |
| **3s（S3）** | **0.7415±0.0306** | **0.5876±0.0296** | `run_20260821_190504` |

训练方案：[20 · 5060](../../../模型训练/20_旁路_shallow_3s滑窗100ms_openbmi_accpaper/) · [21 · 5070](../../../模型训练/21_5070_旁路_shallow_3s滑窗100ms_openbmi_accpaper/)
