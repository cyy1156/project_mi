# 被试独立五折实验记录（20260810_005109 / dbn_raw）

- 开始：`2026-08-10T00:51:09`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`dbn_raw`（单脚本；无 registry）
- 结构：TemporalEncoder(D=64) + DBN(hidden 300/400)；raw 时域输入，无 RBM 预训练
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\dbn_raw\openbmi_balanced_train_2s\run_20260810_005109`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5858 ± 0.0231`
- Test F1：`0.5290 ± 0.1081`
- Test Acc：`0.5824 ± 0.0341`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`9`
- Val F1（最优）：`0.5303`
- Val loss（最优时）：`0.6633`

**Test（overall）**
- Accuracy：`0.5843`
- Recall：`0.3941`
- Specificity：`0.7745`
- Precision：`0.6361`
- F1：`0.4867`
- Balanced Acc：`0.5843`
- 混淆矩阵：TP=`867` TN=`1704` FP=`496` FN=`1333`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`107`
- 验证最优轮次（best_epoch）：`87`
- Val F1（最优）：`0.6594`
- Val loss（最优时）：`0.6802`

**Test（overall）**
- Accuracy：`0.6311`
- Recall：`0.6695`
- Specificity：`0.5927`
- Precision：`0.6218`
- F1：`0.6448`
- Balanced Acc：`0.6311`
- 混淆矩阵：TP=`1473` TN=`1304` FP=`896` FN=`727`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`18`
- Val F1（最优）：`0.4131`
- Val loss（最优时）：`0.7096`

**Test（overall）**
- Accuracy：`0.5536`
- Recall：`0.3309`
- Specificity：`0.7764`
- Precision：`0.5967`
- F1：`0.4257`
- Balanced Acc：`0.5536`
- 混淆矩阵：TP=`728` TN=`1708` FP=`492` FN=`1472`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`73`
- 验证最优轮次（best_epoch）：`53`
- Val F1（最优）：`0.6618`
- Val loss（最优时）：`0.6717`

**Test（overall）**
- Accuracy：`0.6057`
- Recall：`0.8027`
- Specificity：`0.4086`
- Precision：`0.5758`
- F1：`0.6706`
- Balanced Acc：`0.6057`
- 混淆矩阵：TP=`1766` TN=`899` FP=`1301` FN=`434`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`69`
- 验证最优轮次（best_epoch）：`49`
- Val F1（最优）：`0.4252`
- Val loss（最优时）：`0.7811`

**Test（overall）**
- Accuracy：`0.5370`
- Recall：`0.3315`
- Specificity：`0.7425`
- Precision：`0.5628`
- F1：`0.4172`
- Balanced Acc：`0.5370`
- 混淆矩阵：TP=`663` TN=`1485` FP=`515` FN=`1337`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.4812 ± 0.0308`
- Test F1-macro：`0.3110 ± 0.0891`
- Test Acc：`0.4934 ± 0.0158`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`46`
- 验证最优轮次（best_epoch）：`26`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0036`

**Test（overall）**
- Accuracy：`0.5148`
- F1-macro：`0.3202`
- Recall-macro：`0.3724`
- Recall idle/left/right：`0.9418` / `0.0591` / `0.1164`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2072     44     84
  true1    957     65     78
  true2    937     35    128
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`93`
- 验证最优轮次（best_epoch）：`73`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0986`

**Test（overall）**
- Accuracy：`0.4691`
- F1-macro：`0.4648`
- Recall-macro：`0.4909`
- Recall idle/left/right：`0.4036` / `0.5136` / `0.5555`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    888    557    755
  true1    177    565    358
  true2    164    325    611
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0828`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2222`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2200      0      0
  true1   1100      0      0
  true2   1100      0      0
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`4`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0421`

**Test（overall）**
- Accuracy：`0.4830`
- F1-macro：`0.3255`
- Recall-macro：`0.3797`
- Recall idle/left/right：`0.7927` / `0.0000` / `0.3464`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1744      0    456
  true1    704      0    396
  true2    719      0    381
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0760`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2222`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2000      0      0
  true1   1000      0      0
  true2   1000      0      0
```

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 20,
  "batch_train": 128,
  "batch_eval": 256,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "prefetch_factor": 2,
  "non_blocking": true,
  "torch_num_threads": 6,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true
}
```

- 结束：`2026-08-10T01:01:28`
