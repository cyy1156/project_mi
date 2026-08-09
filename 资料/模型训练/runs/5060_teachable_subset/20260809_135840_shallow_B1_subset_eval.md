# B1 子集评估 · shallow

- 时间：`20260809_135840`
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\shallow_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260807_135828`
- 清单：`D:\cyy\MI\find_best_trail\out\teachable_trials_v1.json`
- mask：`D:\cyy\MI\find_best_trail\out\teachable_window_masks_v1.npz`

## task

| 行 | Acc_paper | vs R0 Δ | 试次数/折(均) |
|----|-----------|--------|---------------|
| R0 | 0.6941±0.0349 | +0.0000 | 3240.0 |
| R1 | 0.7143±0.0625 | +0.0203 | 480.0 |
| R2 | 0.6842±0.0963 | -0.0099 | 244.2 |
| R3 | 0.7148±0.0593 | +0.0207 | 527.2 |

## three

| 行 | Acc_paper | vs R0 Δ | 试次数/折(均) |
|----|-----------|--------|---------------|
| R0 | 0.5401±0.0257 | +0.0000 | 3240.0 |
| R1 | 0.6674±0.0738 | +0.1273 | 480.0 |
| R2 | 0.7513±0.0686 | +0.2112 | 244.2 |
| R3 | 0.6825±0.0579 | +0.1424 | 527.2 |

**Three R2 决策**：Δ=+0.2112 → 建议开 B2（R2−R0 ≥ +0.03）

