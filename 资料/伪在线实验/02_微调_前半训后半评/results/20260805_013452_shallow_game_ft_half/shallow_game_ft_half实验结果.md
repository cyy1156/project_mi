# 游戏微调实验记录（20260805_013452 / shallow_game_ft_half）

- protocol：`game_pseudo_online_ft Tw=2s hop=100ms arm=02_finetune_first_half_train_second_half_eval finetune_mode=full_model freeze_backbone=false head_only=false split=trial_order_half init=bci2a_2s_hop100_balbatch_accpaper train=game_first_half_windows eval=game_second_half_pseudo_online early_stop=val_acc_paper max_epochs=300 patience=20 lr=1e-4 balbatch no_rap no_otta metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）；full_model FT
- **finetune_mode=full_model**（全参数可训；非 head-only）
- init：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\shallow_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_165940`
- weights：`D:\cyy\MI\code\train_lab\out\baseline_game_ft_hop100_accpaper\shallow_game_ft_half_balbatch_accpaper\20260805_013452`
- early_stop=**Val Acc_paper** | max_epochs=300 | patience=20
- 主报：后半 trial **段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| shallow | 0.6000 ± 0.0222 | 0.6014 ± 0.0218 | 0.5689 ± 0.0097 | 0.4754 ± 0.0311 |

\*同后半测试集、同 init、未微调对照。

### three

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| shallow | 0.4656 ± 0.0525 | 0.3953 ± 0.0470 | 0.3998 ± 0.0243 | 0.2492 ± 0.0318 |

\*同后半测试集、同 init、未微调对照。

## sub03 / sub03_ses01_20260723_185153

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| shallow | 0.7839 ± 0.0194 | 0.7839 ± 0.0194 | 0.7415 ± 0.0086 | 0.4903 ± 0.0428 |

\*同后半测试集、同 init、未微调对照。

### three

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| shallow | 0.5548 ± 0.0219 | 0.5216 ± 0.0169 | 0.5134 ± 0.0175 | 0.2871 ± 0.0991 |

\*同后半测试集、同 init、未微调对照。

