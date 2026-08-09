# 05 OpenBMI 游戏微调实验记录（20260809_174855 / shallow_openbmi_game_ft_half）

- protocol：`game_pseudo_online_ft Tw=2s hop=100ms arm=05_openbmi_finetune_first_half_train_second_half_eval finetune_mode=full_model freeze_backbone=false head_only=false split=trial_order_half init=openbmi_2s_hop100_balbatch_accpaper channel_remap=game_to_openbmi train=game_first_half_windows eval=game_second_half_pseudo_online early_stop=val_acc_paper max_epochs=300 patience=20 lr=1e-4 balbatch no_rap no_otta model=shallow_only metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux`
- model：`shallow` | ShallowFBCSPNet；OpenBMI Acc_paper init；full_model FT；channel remap
- **finetune_mode=full_model**（全参数可训；非 head-only）
- init_domain：`openbmi`
- init：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\shallow_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260807_135828`
- channel_remap：游戏序 → `Cz, C3, C4, CP3, FC4, FC3, CP4, CPz`
- weights：`D:\cyy\MI\code\train_lab\out\openbmi_game_ft_hop100_accpaper\shallow_openbmi_game_ft_half_balbatch_accpaper\20260809_174855`
- early_stop=**Val Acc_paper** | max_epochs=2 | patience=2
- 主报：后半 trial **段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

- split：train_trials=31 eval_trials=31 | ft_train=25 ft_val=6
- remap：`C3, C4, Cz, CP3, CP4, CPz, FC3, FC4` → `Cz, C3, C4, CP3, FC4, FC3, CP4, CPz`

### task

| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | 零样本后半 Acc_paper* |
|------|----------------|----------------|-----------|----------------------|
| shallow | 0.5738 ± 0.0000 | 0.5785 ± 0.0000 | 0.5426 ± 0.0000 | 0.4754 ± 0.0000 |

\*同后半测试集、同 init、未微调对照。

### three

- （本次跳过）

