# 伪在线实验记录（20260805_005231 / deep_pseudo_online）

- protocol：`game_pseudo_online Tw=2s hop=100ms seg=acquire_mi|rest concat=per_subject_trial_order win=in_segment_only no_cross_boundary models=shallow,deep,conformer,eegnet,eegtcnet select_list=trialmaj_task_acc_paper_top5 weights=bci2a_2s_hop100_balbatch_accpaper weight_pkg=baselines_2s_hop100_accpaper metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux no_game_retrain no_rap no_otta`
- model：`deep` | Deep4Net-compat（pool=1/1）
- weights：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\deep_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_171252`
- preprocess：`game_phase4_like`
- 主指标：**段级 Acc_paper**

## sub02 / sub02_ses01_20260723_180607

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| deep | 0.5154 ± 0.0366 | 0.5158 ± 0.0346 | 0.5043 ± 0.0281 | 2583 | `run_20260804_171252` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| deep | 0.3024 ± 0.1066 | 0.3120 ± 0.0347 | 0.3236 ± 0.0116 | 2583 | `run_20260804_171252` |

## sub03 / sub03_ses01_20260723_185153

### task

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| deep | 0.5210 ± 0.0333 | 0.5210 ± 0.0333 | 0.5008 ± 0.0239 | 2604 | `run_20260804_171252` |

### three

| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |
|------|----------------|-----------------|-------------|---------|----------|
| deep | 0.2952 ± 0.0529 | 0.3151 ± 0.0292 | 0.3168 ± 0.0210 | 2604 | `run_20260804_171252` |

