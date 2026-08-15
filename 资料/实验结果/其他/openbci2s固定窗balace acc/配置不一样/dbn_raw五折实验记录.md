# 被试独立五折实验记录（20260806_061355 / dbn_raw）

- 开始：`2026-08-06T06:13:55`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`dbn_raw`（单脚本；无 registry）
- 输入：原始时域 `(21600, 8, 500)` → TemporalEncoder(D=64) → DBN
- 结构：Encoder + DBN(in=64, hidden 300/400)；监督 forward，无 RBM 预训练
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\dbn_raw\openbmi_balanced_train_2s\run_20260806_061355`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.6265 ± 0.0731`
- Test F1：`0.6373 ± 0.0601`
- Test Acc：`0.5793 ± 0.0154`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`99`
- 验证最优轮次（best_epoch）：`81`
- Val F1（最优）：`0.6556`
- Val loss（最优时）：`0.6844`

**Test（overall）**
- Accuracy：`0.6055`
- Recall：`0.7659`
- Specificity：`0.4450`
- Precision：`0.5798`
- F1：`0.6600`
- Balanced Acc：`0.6055`
- 混淆矩阵：TP=`1685` TN=`979` FP=`1221` FN=`515`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val F1（最优）：`0.6828`
- Val loss（最优时）：`0.7399`

**Test（overall）**
- Accuracy：`0.5834`
- Recall：`0.8727`
- Specificity：`0.2941`
- Precision：`0.5528`
- F1：`0.6769`
- Balanced Acc：`0.5834`
- 混淆矩阵：TP=`1920` TN=`647` FP=`1553` FN=`280`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val F1（最优）：`0.4914`
- Val loss（最优时）：`0.6788`

**Test（overall）**
- Accuracy：`0.5636`
- Recall：`0.4768`
- Specificity：`0.6505`
- Precision：`0.5770`
- F1：`0.5222`
- Balanced Acc：`0.5636`
- 混淆矩阵：TP=`1049` TN=`1431` FP=`769` FN=`1151`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`83`
- 验证最优轮次（best_epoch）：`65`
- Val F1（最优）：`0.6912`
- Val loss（最优时）：`0.7212`

**Test（overall）**
- Accuracy：`0.5798`
- Recall：`0.9327`
- Specificity：`0.2268`
- Precision：`0.5468`
- F1：`0.6894`
- Balanced Acc：`0.5798`
- 混淆矩阵：TP=`2052` TN=`499` FP=`1701` FN=`148`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val F1（最优）：`0.6112`
- Val loss（最优时）：`0.7555`

**Test（overall）**
- Accuracy：`0.5640`
- Recall：`0.7685`
- Specificity：`0.3595`
- Precision：`0.5454`
- F1：`0.6380`
- Balanced Acc：`0.5640`
- 混淆矩阵：TP=`1537` TN=`719` FP=`1281` FN=`463`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4236 ± 0.0473`
- Test F1-macro：`0.4195 ± 0.0619`
- Test Acc：`0.4923 ± 0.0117`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`132`
- 验证最优轮次（best_epoch）：`114`
- Val F1-macro（最优）：`0.4465`
- Val loss（最优时）：`1.0135`

**Test（overall）**
- Accuracy：`0.4968`
- F1-macro：`0.4695`
- Recall-macro：`0.4732`
- Recall idle/left/right：`0.5677` / `0.3918` / `0.4600`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1249    409    542
  true1    371    431    298
  true2    378    216    506
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`84`
- 验证最优轮次（best_epoch）：`66`
- Val F1-macro（最优）：`0.4557`
- Val loss（最优时）：`1.0259`

**Test（overall）**
- Accuracy：`0.5045`
- F1-macro：`0.4696`
- Recall-macro：`0.4715`
- Recall idle/left/right：`0.6036` / `0.4382` / `0.3727`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1328    411    461
  true1    381    482    237
  true2    365    325    410
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`52`
- 验证最优轮次（best_epoch）：`34`
- Val F1-macro（最优）：`0.3475`
- Val loss（最优时）：`1.0113`

**Test（overall）**
- Accuracy：`0.5009`
- F1-macro：`0.3082`
- Recall-macro：`0.3614`
- Recall idle/left/right：`0.9195` / `0.0500` / `0.1145`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2023     48    129
  true1    952     55     93
  true2    943     31    126
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`124`
- 验证最优轮次（best_epoch）：`106`
- Val F1-macro（最优）：`0.4768`
- Val loss（最优时）：`0.9690`

**Test（overall）**
- Accuracy：`0.4875`
- F1-macro：`0.4541`
- Recall-macro：`0.4562`
- Recall idle/left/right：`0.5814` / `0.3764` / `0.4109`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1279    427    494
  true1    394    414    292
  true2    361    287    452
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`103`
- 验证最优轮次（best_epoch）：`85`
- Val F1-macro（最优）：`0.3916`
- Val loss（最优时）：`1.0978`

**Test（overall）**
- Accuracy：`0.4718`
- F1-macro：`0.3961`
- Recall-macro：`0.3977`
- Recall idle/left/right：`0.6940` / `0.2260` / `0.2730`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1388    226    386
  true1    567    226    207
  true2    598    129    273
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

- 结束：`2026-08-06T06:34:42`
