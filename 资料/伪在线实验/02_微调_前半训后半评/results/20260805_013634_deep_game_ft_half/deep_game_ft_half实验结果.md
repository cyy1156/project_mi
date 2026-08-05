# 游戏微调实验记录（20260805_013634 / deep_game_ft_half）

- protocol：`game_pseudo_online_ft Tw=2s hop=100ms arm=02_finetune_first_half_train_second_half_eval finetune_mode=full_model freeze_backbone=false head_only=false split=trial_order_half init=bci2a_2s_hop100_balbatch_accpaper train=game_first_half_windows eval=game_second_half_pseudo_online early_stop=val_acc_paper max_epochs=300 patience=20 lr=1e-4 balbatch no_rap no_otta metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux`
- model：`deep` | Deep4Net-compat（pool=1/1）；full_model FT
- **finetune_mode=full_model**（全参数可训；非 head-only）
- init：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\deep_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_171252`
- weights：`D:\cyy\MI\code\train_lab\out\baseline_game_ft_hop100_accpaper\deep_game_ft_half_balbatch_accpaper\20260805_013634`
- early_stop=**Val Acc_paper** | max_epochs=300 | patience=20
- 主报：后半 trial **段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| deep | 0.5344 ± 0.0266 | 0.5337 ± 0.0290 | 0.5101 ± 0.0157 | 0.4918 ± 0.0402 |

\*同后半测试集、同 init、未微调对照。

### three

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| deep | 0.2918 ± 0.0826 | 0.3081 ± 0.0391 | 0.3164 ± 0.0250 | 0.2754 ± 0.1147 |

\*同后半测试集、同 init、未微调对照。

## sub03 / sub03_ses01_20260723_185153

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| deep | 0.6516 ± 0.0416 | 0.6516 ± 0.0416 | 0.6252 ± 0.0479 | 0.5419 ± 0.0485 |

\*同后半测试集、同 init、未微调对照。

### three

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| deep | 0.4419 ± 0.1043 | 0.4109 ± 0.0729 | 0.4154 ± 0.0373 | 0.2968 ± 0.0506 |

\*同后半测试集、同 init、未微调对照。

