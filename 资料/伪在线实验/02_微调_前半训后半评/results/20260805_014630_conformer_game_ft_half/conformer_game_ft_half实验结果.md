# 游戏微调实验记录（20260805_014630 / conformer_game_ft_half）

- protocol：`game_pseudo_online_ft Tw=2s hop=100ms arm=02_finetune_first_half_train_second_half_eval finetune_mode=full_model freeze_backbone=false head_only=false split=trial_order_half init=bci2a_2s_hop100_balbatch_accpaper train=game_first_half_windows eval=game_second_half_pseudo_online early_stop=val_acc_paper max_epochs=300 patience=20 lr=1e-4 balbatch no_rap no_otta metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux`
- model：`conformer` | EEGConformer num_layers=2, num_heads=10; full_model FT
- **finetune_mode=full_model**（全参数可训；非 head-only）
- init：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\conformer_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_195303`
- weights：`D:\cyy\MI\code\train_lab\out\baseline_game_ft_hop100_accpaper\conformer_game_ft_half_balbatch_accpaper\20260805_014630`
- early_stop=**Val Acc_paper** | max_epochs=300 | patience=20
- 主报：后半 trial **段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| conformer | 0.5311 ± 0.0245 | 0.5335 ± 0.0240 | 0.5545 ± 0.0172 | 0.4918 ± 0.0293 |

\*同后半测试集、同 init、未微调对照。

### three

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| conformer | 0.4885 ± 0.0282 | 0.3850 ± 0.0182 | 0.3960 ± 0.0146 | 0.3082 ± 0.0953 |

\*同后半测试集、同 init、未微调对照。

## sub03 / sub03_ses01_20260723_185153

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| conformer | 0.7613 ± 0.0400 | 0.7613 ± 0.0400 | 0.7204 ± 0.0311 | 0.4419 ± 0.1018 |

\*同后半测试集、同 init、未微调对照。

### three

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| conformer | 0.5677 ± 0.0237 | 0.4912 ± 0.0331 | 0.4897 ± 0.0328 | 0.2710 ± 0.1037 |

\*同后半测试集、同 init、未微调对照。

