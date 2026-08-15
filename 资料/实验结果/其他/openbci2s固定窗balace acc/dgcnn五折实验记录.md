# 被试独立五折实验记录（20260810_004525 / dgcnn）

- 开始：`2026-08-10T00:45:25`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`dgcnn`（单脚本；无 registry）
- 结构：DGCNN(k=2, layers=[128], dropout=shared drop_prob)
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\dgcnn\openbmi_balanced_train_2s\run_20260810_004525`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5375 ± 0.0138`
- Test F1：`0.5234 ± 0.0426`
- Test Acc：`0.5438 ± 0.0067`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`9`
- Val F1（最优）：`0.5180`
- Val loss（最优时）：`0.6859`

**Test（overall）**
- Accuracy：`0.5377`
- Recall：`0.4841`
- Specificity：`0.5914`
- Precision：`0.5423`
- F1：`0.5115`
- Balanced Acc：`0.5377`
- 混淆矩阵：TP=`1065` TN=`1301` FP=`899` FN=`1135`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`6`
- Val F1（最优）：`0.4866`
- Val loss（最优时）：`0.6880`

**Test（overall）**
- Accuracy：`0.5500`
- Recall：`0.4668`
- Specificity：`0.6332`
- Precision：`0.5600`
- F1：`0.5092`
- Balanced Acc：`0.5500`
- 混淆矩阵：TP=`1027` TN=`1393` FP=`807` FN=`1173`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`62`
- 验证最优轮次（best_epoch）：`42`
- Val F1（最优）：`0.5072`
- Val loss（最优时）：`0.6948`

**Test（overall）**
- Accuracy：`0.5339`
- Recall：`0.4623`
- Specificity：`0.6055`
- Precision：`0.5395`
- F1：`0.4979`
- Balanced Acc：`0.5339`
- 混淆矩阵：TP=`1017` TN=`1332` FP=`868` FN=`1183`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.5717`
- Val loss（最优时）：`0.6886`

**Test（overall）**
- Accuracy：`0.5480`
- Recall：`0.6991`
- Specificity：`0.3968`
- Precision：`0.5368`
- F1：`0.6073`
- Balanced Acc：`0.5480`
- 混淆矩阵：TP=`1538` TN=`873` FP=`1327` FN=`662`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`5`
- Val F1（最优）：`0.4720`
- Val loss（最优时）：`0.7010`

**Test（overall）**
- Accuracy：`0.5495`
- Recall：`0.4350`
- Specificity：`0.6640`
- Precision：`0.5642`
- F1：`0.4912`
- Balanced Acc：`0.5495`
- 混淆矩阵：TP=`870` TN=`1328` FP=`672` FN=`1130`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.4256 ± 0.0155`
- Test F1-macro：`0.3706 ± 0.0116`
- Test Acc：`0.4308 ± 0.0229`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`13`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0646`

**Test（overall）**
- Accuracy：`0.4218`
- F1-macro：`0.3821`
- Recall-macro：`0.3821`
- Recall idle/left/right：`0.5409` / `0.3200` / `0.2855`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1190    560    450
  true1    496    352    252
  true2    539    247    314
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`8`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0668`

**Test（overall）**
- Accuracy：`0.4064`
- F1-macro：`0.3805`
- Recall-macro：`0.3853`
- Recall idle/left/right：`0.4695` / `0.4000` / `0.2864`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1033    679    488
  true1    396    440    264
  true2    415    370    315
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`17`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0669`

**Test（overall）**
- Accuracy：`0.4152`
- F1-macro：`0.3559`
- Recall-macro：`0.3570`
- Recall idle/left/right：`0.5900` / `0.2609` / `0.2200`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1298    475    427
  true1    573    287    240
  true2    632    226    242
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`4`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0624`

**Test（overall）**
- Accuracy：`0.4395`
- F1-macro：`0.3775`
- Recall-macro：`0.3823`
- Recall idle/left/right：`0.6114` / `0.1836` / `0.3518`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1345    295    560
  true1    595    202    303
  true2    550    163    387
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0838`

**Test（overall）**
- Accuracy：`0.4713`
- F1-macro：`0.3572`
- Recall-macro：`0.3785`
- Recall idle/left/right：`0.7495` / `0.3010` / `0.0850`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1499    369    132
  true1    629    301     70
  true2    680    235     85
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

- 结束：`2026-08-10T00:50:31`
