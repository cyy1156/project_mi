# 伪在线实验记录（20260805_005247 / conformer_pseudo_online）

- protocol：`game_pseudo_online Tw=2s hop=100ms seg=acquire_mi|rest concat=per_subject_trial_order win=in_segment_only no_cross_boundary models=shallow,deep,conformer,eegnet,eegtcnet select_list=trialmaj_task_acc_paper_top5 weights=bci2a_2s_hop100_balbatch_accpaper weight_pkg=baselines_2s_hop100_accpaper metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux no_game_retrain no_rap no_otta`
- model：`conformer` | EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- weights：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\conformer_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_195303`
- preprocess：`game_phase4_like`
- 主指标：**段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| conformer | 0.4813 ± 0.0390 | 0.4806 ± 0.0365 | 0.4914 ± 0.0304 | 2583 | `run_20260804_195303` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| conformer | 0.3220 ± 0.0979 | 0.3192 ± 0.0125 | 0.3222 ± 0.0133 | 2583 | `run_20260804_195303` |

## sub03 / sub03_ses01_20260723_185153

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| conformer | 0.4532 ± 0.0904 | 0.4532 ± 0.0904 | 0.4548 ± 0.0578 | 2604 | `run_20260804_195303` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| conformer | 0.2694 ± 0.0959 | 0.3022 ± 0.0344 | 0.3052 ± 0.0231 | 2604 | `run_20260804_195303` |

