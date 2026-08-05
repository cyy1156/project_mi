# 游戏微调实验记录（20260805_015555 / eegtcnet_game_ft_half）

- protocol：`game_pseudo_online_ft Tw=2s hop=100ms arm=02_finetune_first_half_train_second_half_eval finetune_mode=full_model freeze_backbone=false head_only=false split=trial_order_half init=bci2a_2s_hop100_balbatch_accpaper train=game_first_half_windows eval=game_second_half_pseudo_online early_stop=val_acc_paper max_epochs=300 patience=20 lr=1e-4 balbatch no_rap no_otta metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux`
- model：`eegtcnet` | EEGTCNet（braindecode 默认）；full_model FT
- **finetune_mode=full_model**（全参数可训；非 head-only）
- init：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\eegtcnet_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_190821`
- weights：`D:\cyy\MI\code\train_lab\out\baseline_game_ft_hop100_accpaper\eegtcnet_game_ft_half_balbatch_accpaper\20260805_015555`
- early_stop=**Val Acc_paper** | max_epochs=300 | patience=20
- 主报：后半 trial **段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| eegtcnet | 0.5115 ± 0.0380 | 0.5111 ± 0.0397 | 0.5106 ± 0.0303 | 0.5115 ± 0.0066 |

\*同后半测试集、同 init、未微调对照。

### three

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| eegtcnet | 0.3279 ± 0.0933 | 0.3119 ± 0.0293 | 0.3180 ± 0.0191 | 0.3016 ± 0.0554 |

\*同后半测试集、同 init、未微调对照。

## sub03 / sub03_ses01_20260723_185153

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| eegtcnet | 0.6452 ± 0.1015 | 0.6452 ± 0.1015 | 0.6269 ± 0.0730 | 0.5000 ± 0.0250 |

\*同后半测试集、同 init、未微调对照。

### three

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| eegtcnet | 0.3548 ± 0.0907 | 0.3630 ± 0.0750 | 0.3431 ± 0.0779 | 0.2968 ± 0.0982 |

\*同后半测试集、同 init、未微调对照。

