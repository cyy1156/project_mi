# B1 子集评估 · eegnet

- 时间：`20260809_140743`
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\eegnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_172218`
- 清单：`D:\cyy\MI\find_best_trail\out\teachable_trials_v1.json`
- mask：`D:\cyy\MI\find_best_trail\out\teachable_window_masks_v1.npz`

## task

| 行 | Acc_paper | vs R0 Δ | 试次数/折(均) |
|----|-----------|--------|---------------|
| R0 | 0.6869±0.0334 | +0.0000 | 3240.0 |
| R1 | 0.6606±0.1003 | -0.0263 | 480.0 |
| R2 | 0.5726±0.1557 | -0.1143 | 244.2 |
| R3 | 0.6662±0.0995 | -0.0207 | 527.2 |

## three

| 行 | Acc_paper | vs R0 Δ | 试次数/折(均) |
|----|-----------|--------|---------------|
| R0 | 0.5322±0.0291 | +0.0000 | 3240.0 |
| R1 | 0.6725±0.1016 | +0.1403 | 480.0 |
| R2 | 0.7447±0.0803 | +0.2125 | 244.2 |
| R3 | 0.7071±0.0570 | +0.1750 | 527.2 |

**Three R2 决策**：Δ=+0.2125 → 建议开 B2（R2−R0 ≥ +0.03）

