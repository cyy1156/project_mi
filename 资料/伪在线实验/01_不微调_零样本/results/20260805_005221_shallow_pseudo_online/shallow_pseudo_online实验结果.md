# 伪在线实验记录（20260805_005221 / shallow_pseudo_online）

- protocol：`game_pseudo_online Tw=2s hop=100ms seg=acquire_mi|rest concat=per_subject_trial_order win=in_segment_only no_cross_boundary models=shallow,deep,conformer,eegnet,eegtcnet select_list=trialmaj_task_acc_paper_top5 weights=bci2a_2s_hop100_balbatch_accpaper weight_pkg=baselines_2s_hop100_accpaper metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux no_game_retrain no_rap no_otta`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）
- weights：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\shallow_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_165940`
- preprocess：`game_phase4_like`
- 主指标：**段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| shallow | 0.4813 ± 0.0310 | 0.4805 ± 0.0306 | 0.4823 ± 0.0237 | 2583 | `run_20260804_165940` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| shallow | 0.2472 ± 0.0132 | 0.3112 ± 0.0382 | 0.3268 ± 0.0214 | 2583 | `run_20260804_165940` |

## sub03 / sub03_ses01_20260723_185153

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| shallow | 0.4839 ± 0.0378 | 0.4839 ± 0.0378 | 0.4790 ± 0.0266 | 2604 | `run_20260804_165940` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| shallow | 0.2839 ± 0.0846 | 0.3215 ± 0.0181 | 0.3231 ± 0.0156 | 2604 | `run_20260804_165940` |

