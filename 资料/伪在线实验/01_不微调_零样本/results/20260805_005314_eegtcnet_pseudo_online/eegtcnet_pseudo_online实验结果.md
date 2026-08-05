# 伪在线实验记录（20260805_005314 / eegtcnet_pseudo_online）

- protocol：`game_pseudo_online Tw=2s hop=100ms seg=acquire_mi|rest concat=per_subject_trial_order win=in_segment_only no_cross_boundary models=shallow,deep,conformer,eegnet,eegtcnet select_list=trialmaj_task_acc_paper_top5 weights=bci2a_2s_hop100_balbatch_accpaper weight_pkg=baselines_2s_hop100_accpaper metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux no_game_retrain no_rap no_otta`
- model：`eegtcnet` | EEGTCNet（braindecode 默认）
- weights：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\eegtcnet_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_190821`
- preprocess：`game_phase4_like`
- 主指标：**段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| eegtcnet | 0.5057 ± 0.0033 | 0.5028 ± 0.0035 | 0.5071 ± 0.0055 | 2583 | `run_20260804_190821` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| eegtcnet | 0.2959 ± 0.0728 | 0.3419 ± 0.0075 | 0.3365 ± 0.0070 | 2583 | `run_20260804_190821` |

## sub03 / sub03_ses01_20260723_185153

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| eegtcnet | 0.5016 ± 0.0200 | 0.5016 ± 0.0200 | 0.5058 ± 0.0261 | 2604 | `run_20260804_190821` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| eegtcnet | 0.2935 ± 0.1002 | 0.3269 ± 0.0164 | 0.3233 ± 0.0144 | 2604 | `run_20260804_190821` |

