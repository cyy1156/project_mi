# 被试独立五折实验记录（20260805_232302 / eegnet）

- 开始：`2026-08-05T23:23:02`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`eegnet`（单脚本；无 registry）
- 结构：F1=8, D=2, F2=16
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\eegnet\openbmi_balanced_train_2s\run_20260805_232302`

---
## 最终结论

### Task（静息/任务）
- Val Balanced Acc：`0.5966 ± 0.0290`
- Test F1：`0.5840 ± 0.0588`
- Test Acc：`0.5816 ± 0.0291`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`88`
- 验证最优轮次（best_epoch）：`70`
- Val F1（最优）：`0.5486`
- Val loss（最优时）：`0.6659`

**Test（overall）**
- Accuracy：`0.5882`
- Recall：`0.3895`
- Specificity：`0.7868`
- Precision：`0.6463`
- F1：`0.4861`
- Balanced Acc：`0.5882`
- 混淆矩阵：TP=`857` TN=`1731` FP=`469` FN=`1343`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`60`
- 验证最优轮次（best_epoch）：`42`
- Val F1（最优）：`0.6516`
- Val loss（最优时）：`0.6579`

**Test（overall）**
- Accuracy：`0.5830`
- Recall：`0.6286`
- Specificity：`0.5373`
- Precision：`0.5760`
- F1：`0.6012`
- Balanced Acc：`0.5830`
- 混淆矩阵：TP=`1383` TN=`1182` FP=`1018` FN=`817`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`60`
- 验证最优轮次（best_epoch）：`42`
- Val F1（最优）：`0.6030`
- Val loss（最优时）：`0.6776`

**Test（overall）**
- Accuracy：`0.5714`
- Recall：`0.7136`
- Specificity：`0.4291`
- Precision：`0.5556`
- F1：`0.6248`
- Balanced Acc：`0.5714`
- 混淆矩阵：TP=`1570` TN=`944` FP=`1256` FN=`630`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`111`
- 验证最优轮次（best_epoch）：`93`
- Val F1（最优）：`0.6187`
- Val loss（最优时）：`0.6450`

**Test（overall）**
- Accuracy：`0.6280`
- Recall：`0.7027`
- Specificity：`0.5532`
- Precision：`0.6113`
- F1：`0.6538`
- Balanced Acc：`0.6280`
- 混淆矩阵：TP=`1546` TN=`1217` FP=`983` FN=`654`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val F1（最优）：`0.5518`
- Val loss（最优时）：`0.6976`

**Test（overall）**
- Accuracy：`0.5377`
- Recall：`0.5745`
- Specificity：`0.5010`
- Precision：`0.5352`
- F1：`0.5541`
- Balanced Acc：`0.5377`
- 混淆矩阵：TP=`1149` TN=`1002` FP=`998` FN=`851`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4607 ± 0.0343`
- Test F1-macro：`0.4585 ± 0.0300`
- Test Acc：`0.5041 ± 0.0204`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val F1-macro（最优）：`0.4492`
- Val loss（最优时）：`0.9966`

**Test（overall）**
- Accuracy：`0.5195`
- F1-macro：`0.4076`
- Recall-macro：`0.4152`
- Recall idle/left/right：`0.8327` / `0.2018` / `0.2109`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1832    153    215
  true1    787    222     91
  true2    797     71    232
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`160`
- 验证最优轮次（best_epoch）：`142`
- Val F1-macro（最优）：`0.4873`
- Val loss（最优时）：`0.9766`

**Test（overall）**
- Accuracy：`0.5211`
- F1-macro：`0.4930`
- Recall-macro：`0.4935`
- Recall idle/left/right：`0.6041` / `0.4027` / `0.4736`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1329    378    493
  true1    431    443    226
  true2    431    148    521
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`118`
- 验证最优轮次（best_epoch）：`100`
- Val F1-macro（最优）：`0.4559`
- Val loss（最优时）：`1.0035`

**Test（overall）**
- Accuracy：`0.4786`
- F1-macro：`0.4582`
- Recall-macro：`0.4632`
- Recall idle/left/right：`0.5250` / `0.4473` / `0.4173`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1155    534    511
  true1    368    492    240
  true2    429    212    459
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`122`
- 验证最优轮次（best_epoch）：`104`
- Val F1-macro（最优）：`0.5053`
- Val loss（最优时）：`0.9339`

**Test（overall）**
- Accuracy：`0.5216`
- F1-macro：`0.4841`
- Recall-macro：`0.4867`
- Recall idle/left/right：`0.6264` / `0.5064` / `0.3273`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1378    476    346
  true1    397    557    146
  true2    467    273    360
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val F1-macro（最优）：`0.4057`
- Val loss（最优时）：`1.0636`

**Test（overall）**
- Accuracy：`0.4798`
- F1-macro：`0.4497`
- Recall-macro：`0.4563`
- Recall idle/left/right：`0.5500` / `0.5170` / `0.3020`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100    561    339
  true1    344    517    139
  true2    455    243    302
```

### 共用超参
```json
{
  "data_tag": "openbmi_balanced_train_2s",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 18,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5
}
```

- 结束：`2026-08-06T00:07:56`
