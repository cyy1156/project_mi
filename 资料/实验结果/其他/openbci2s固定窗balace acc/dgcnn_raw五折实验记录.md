# 被试独立五折实验记录（20260810_010132 / dgcnn_raw）

- 开始：`2026-08-10T01:01:32`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`dgcnn_raw`（单脚本；无 registry）
- 结构：TemporalEncoder(D=64) + DGCNN(k=2, layers=[128])；raw 时域输入
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\dgcnn_raw\openbmi_balanced_train_2s\run_20260810_010132`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5464 ± 0.0311`
- Test F1：`0.4419 ± 0.2249`
- Test Acc：`0.5366 ± 0.0370`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.5811`
- Val loss（最优时）：`0.9020`

**Test（overall）**
- Accuracy：`0.4891`
- Recall：`0.5223`
- Specificity：`0.4559`
- Precision：`0.4898`
- F1：`0.5055`
- Balanced Acc：`0.4891`
- 混淆矩阵：TP=`1149` TN=`1003` FP=`1197` FN=`1051`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.0000`
- Val loss（最优时）：`3.0088`

**Test（overall）**
- Accuracy：`0.5000`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2200` FP=`0` FN=`2200`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.4992`
- Val loss（最优时）：`0.6869`

**Test（overall）**
- Accuracy：`0.5430`
- Recall：`0.5177`
- Specificity：`0.5682`
- Precision：`0.5452`
- F1：`0.5311`
- Balanced Acc：`0.5430`
- 混淆矩阵：TP=`1139` TN=`1250` FP=`950` FN=`1061`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.6227`
- Val loss（最优时）：`0.6863`

**Test（overall）**
- Accuracy：`0.5855`
- Recall：`0.7077`
- Specificity：`0.4632`
- Precision：`0.5687`
- F1：`0.6306`
- Balanced Acc：`0.5855`
- 混淆矩阵：TP=`1557` TN=`1019` FP=`1181` FN=`643`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`73`
- 验证最优轮次（best_epoch）：`53`
- Val F1（最优）：`0.5047`
- Val loss（最优时）：`0.8257`

**Test（overall）**
- Accuracy：`0.5655`
- Recall：`0.5145`
- Specificity：`0.6165`
- Precision：`0.5729`
- F1：`0.5421`
- Balanced Acc：`0.5655`
- 混淆矩阵：TP=`1029` TN=`1233` FP=`767` FN=`971`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.5073 ± 0.0050`
- Test F1-macro：`0.2866 ± 0.0716`
- Test Acc：`0.4964 ± 0.0129`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`11`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0176`

**Test（overall）**
- Accuracy：`0.4995`
- F1-macro：`0.2250`
- Recall-macro：`0.3338`
- Recall idle/left/right：`0.9968` / `0.0045` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2193      5      2
  true1   1094      5      1
  true2   1099      1      0
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`24`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0214`

**Test（overall）**
- Accuracy：`0.5050`
- F1-macro：`0.2437`
- Recall-macro：`0.3420`
- Recall idle/left/right：`0.9941` / `0.0273` / `0.0045`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2187      8      5
  true1   1062     30      8
  true2   1076     19      5
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`30`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`0.9907`

**Test（overall）**
- Accuracy：`0.4711`
- F1-macro：`0.3410`
- Recall-macro：`0.3621`
- Recall idle/left/right：`0.7982` / `0.1418` / `0.1464`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1756    234    210
  true1    773    156    171
  true2    841     98    161
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`113`
- 验证最优轮次（best_epoch）：`93`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`0.9633`

**Test（overall）**
- Accuracy：`0.5061`
- F1-macro：`0.4004`
- Recall-macro：`0.4083`
- Recall idle/left/right：`0.7995` / `0.2255` / `0.2000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1759    234    207
  true1    714    248    138
  true2    717    163    220
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`4`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0623`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2229`
- Recall-macro：`0.3335`
- Recall idle/left/right：`0.9995` / `0.0010` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1999      1      0
  true1    999      1      0
  true2    998      2      0
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

- 结束：`2026-08-10T01:11:03`
