# 13 · CIACNet 想法复现（旁路）

| 项 | 路径 |
|----|------|
| **完整方案** | [`方案.md`](./方案.md) |
| 结果登记 | [`总结/结果登记表.md`](./总结/结果登记表.md) |
| 论文 PDF | [`../../论文/pdf.pdf`](../../论文/pdf.pdf) |
| 代码 | `code/train_lab/src/step/5060_ciacnet_mi_accpaper/` |
| P 数据缓存 | `code/preprocess_lab/out/bci2a_p_8ch_cue05to5/` |
| out | `code/train_lab/out/5060_ciacnet_mi_accpaper/` |

- **P 轨（已完成主报）**：8ch 论文配置下 EEGNet **68.36%** / CIACNet **71.41%**（Δ **+3.05pp**，实现合格；未达 80% 接近论文趋势）。详见登记表。  
- **L 轨**：同结构 + **本室个性化**（OpenBMI 滑窗、被试独立、Acc_paper、balbatch 等）。  
禁止 P/L 与论文 85% / shallow 0.540 绝对值混比。

### P / L 快速命令

```bash
cd code/train_lab/src/step/5060_ciacnet_mi_accpaper
# P 轨
python run_p_track.py --arm P1
python run_p_track.py --arm P2
# L 轨（本室 Acc_paper；L0=fold0）
python run_l_track.py --arm L0e    # EEGNet fold0
python run_l_track.py --arm L0c    # CIACNet fold0
python chain_l0.py                 # L0e → L0c
python run_l_track.py --arm L1c    # 五折（门控后）
```
