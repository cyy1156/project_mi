# 被试独立五折实验记录（20260810_003334 / dbn）

- 开始：`2026-08-10T00:33:34`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`dbn`（单脚本；无 registry）
- 结构：DBN(hidden 300/400)；监督 forward，无 RBM 预训练；drop_prob 忽略
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\dbn\openbmi_balanced_train_2s\run_20260810_003334`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5323 ± 0.0156`
- Test F1：`0.4186 ± 0.1937`
- Test Acc：`0.5199 ± 0.0217`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`12`
- Val F1（最优）：`0.3748`
- Val loss（最优时）：`0.6919`

**Test（overall）**
- Accuracy：`0.5014`
- Recall：`0.1377`
- Specificity：`0.8650`
- Precision：`0.5050`
- F1：`0.2164`
- Balanced Acc：`0.5014`
- 混淆矩阵：TP=`303` TN=`1903` FP=`297` FN=`1897`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`3`
- Val F1（最优）：`0.3478`
- Val loss（最优时）：`0.6931`

**Test（overall）**
- Accuracy：`0.5143`
- Recall：`0.0891`
- Specificity：`0.9395`
- Precision：`0.5957`
- F1：`0.1550`
- Balanced Acc：`0.5143`
- 混淆矩阵：TP=`196` TN=`2067` FP=`133` FN=`2004`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`9`
- Val F1（最优）：`0.6199`
- Val loss（最优时）：`0.6928`

**Test（overall）**
- Accuracy：`0.4934`
- Recall：`0.8445`
- Specificity：`0.1423`
- Precision：`0.4961`
- F1：`0.6251`
- Balanced Acc：`0.4934`
- 混淆矩阵：TP=`1858` TN=`313` FP=`1887` FN=`342`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`46`
- Val F1（最优）：`0.5334`
- Val loss（最优时）：`0.6870`

**Test（overall）**
- Accuracy：`0.5450`
- Recall：`0.6077`
- Specificity：`0.4823`
- Precision：`0.5400`
- F1：`0.5719`
- Balanced Acc：`0.5450`
- 混淆矩阵：TP=`1337` TN=`1061` FP=`1139` FN=`863`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`7`
- Val F1（最优）：`0.5160`
- Val loss（最优时）：`0.6931`

**Test（overall）**
- Accuracy：`0.5453`
- Recall：`0.5015`
- Specificity：`0.5890`
- Precision：`0.5496`
- F1：`0.5244`
- Balanced Acc：`0.5453`
- 混淆矩阵：TP=`1003` TN=`1178` FP=`822` FN=`997`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.5000 ± 0.0000`
- Test F1-macro：`0.2222 ± 0.0000`
- Test Acc：`0.5000 ± 0.0000`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`6`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0890`

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

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`8`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0937`

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

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0829`

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
- Val loss（最优时）：`1.0831`

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

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0767`

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

- 结束：`2026-08-10T00:37:59`
