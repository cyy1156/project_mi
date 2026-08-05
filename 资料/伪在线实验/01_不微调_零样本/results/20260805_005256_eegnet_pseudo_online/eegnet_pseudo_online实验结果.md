# 伪在线实验记录（20260805_005256 / eegnet_pseudo_online）

- protocol：`game_pseudo_online Tw=2s hop=100ms seg=acquire_mi|rest concat=per_subject_trial_order win=in_segment_only no_cross_boundary models=shallow,deep,conformer,eegnet,eegtcnet select_list=trialmaj_task_acc_paper_top5 weights=bci2a_2s_hop100_balbatch_accpaper weight_pkg=baselines_2s_hop100_accpaper metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux no_game_retrain no_rap no_otta`
- model：`eegnet` | EEGNet F1=8, D=2, F2=16
- weights：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\eegnet_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_160649`
- preprocess：`game_phase4_like`
- 主指标：**段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| eegnet | 0.4829 ± 0.0404 | 0.4835 ± 0.0386 | 0.4921 ± 0.0197 | 2583 | `run_20260804_160649` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| eegnet | 0.3122 ± 0.0697 | 0.3351 ± 0.0479 | 0.3403 ± 0.0207 | 2583 | `run_20260804_160649` |

## sub03 / sub03_ses01_20260723_185153

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| eegnet | 0.4419 ± 0.0412 | 0.4419 ± 0.0412 | 0.4663 ± 0.0194 | 2604 | `run_20260804_160649` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| eegnet | 0.3145 ± 0.0502 | 0.3183 ± 0.0399 | 0.3187 ± 0.0264 | 2604 | `run_20260804_160649` |

