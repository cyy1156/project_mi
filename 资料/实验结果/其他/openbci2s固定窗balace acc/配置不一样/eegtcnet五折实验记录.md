# 被试独立五折实验记录（20260806_003410 / eegtcnet）

- 开始：`2026-08-06T00:34:10`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`eegtcnet`（单脚本；无 registry）
- 结构：EEGTCNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\eegtcnet\openbmi_balanced_train_2s\run_20260806_003410`

---
## 最终结论

### Task（静息/任务）
- Val Balanced Acc：`0.5723 ± 0.0503`
- Test F1：`0.6281 ± 0.0390`
- Test Acc：`0.5717 ± 0.0483`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.6678`
- Val loss（最优时）：`0.7135`

**Test（overall）**
- Accuracy：`0.5016`
- Recall：`0.9914`
- Specificity：`0.0118`
- Precision：`0.5008`
- F1：`0.6654`
- Balanced Acc：`0.5016`
- 混淆矩阵：TP=`2181` TN=`26` FP=`2174` FN=`19`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`115`
- 验证最优轮次（best_epoch）：`97`
- Val F1（最优）：`0.6790`
- Val loss（最优时）：`0.6477`

**Test（overall）**
- Accuracy：`0.6120`
- Recall：`0.6823`
- Specificity：`0.5418`
- Precision：`0.5982`
- F1：`0.6375`
- Balanced Acc：`0.6120`
- 混淆矩阵：TP=`1501` TN=`1192` FP=`1008` FN=`699`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val F1（最优）：`0.5337`
- Val loss（最优时）：`0.6920`

**Test（overall）**
- Accuracy：`0.5698`
- Recall：`0.5568`
- Specificity：`0.5827`
- Precision：`0.5716`
- F1：`0.5641`
- Balanced Acc：`0.5698`
- 混淆矩阵：TP=`1225` TN=`1282` FP=`918` FN=`975`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`163`
- 验证最优轮次（best_epoch）：`145`
- Val F1（最优）：`0.6486`
- Val loss（最优时）：`0.6337`

**Test（overall）**
- Accuracy：`0.6355`
- Recall：`0.7318`
- Specificity：`0.5391`
- Precision：`0.6136`
- F1：`0.6675`
- Balanced Acc：`0.6355`
- 混淆矩阵：TP=`1610` TN=`1186` FP=`1014` FN=`590`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val F1（最优）：`0.6135`
- Val loss（最优时）：`0.6982`

**Test（overall）**
- Accuracy：`0.5395`
- Recall：`0.7080`
- Specificity：`0.3710`
- Precision：`0.5295`
- F1：`0.6059`
- Balanced Acc：`0.5395`
- 混淆矩阵：TP=`1416` TN=`742` FP=`1258` FN=`584`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.2677 ± 0.0483`
- Test F1-macro：`0.2663 ± 0.0433`
- Test Acc：`0.4484 ± 0.0561`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1-macro（最优）：`0.2315`
- Val loss（最优时）：`1.0547`

**Test（overall）**
- Accuracy：`0.4964`
- F1-macro：`0.2329`
- Recall-macro：`0.3341`
- Recall idle/left/right：`0.9832` / `0.0191` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2163     37      0
  true1   1079     21      0
  true2   1083     17      0
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`0.2223`
- Val loss（最优时）：`1.0464`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2223`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2200      0      0
  true1   1100      0      0
  true2   1099      1      0
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`0.2435`
- Val loss（最优时）：`1.0638`

**Test（overall）**
- Accuracy：`0.4811`
- F1-macro：`0.2479`
- Recall-macro：`0.3298`
- Recall idle/left/right：`0.9350` / `0.0409` / `0.0136`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2057    113     30
  true1   1045     45     10
  true2   1039     46     15
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val F1-macro（最优）：`0.2884`
- Val loss（最优时）：`1.0824`

**Test（overall）**
- Accuracy：`0.4036`
- F1-macro：`0.2880`
- Recall-macro：`0.3380`
- Recall idle/left/right：`0.6005` / `0.4136` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1321    879      0
  true1    645    455      0
  true2    679    421      0
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`0.3529`
- Val loss（最优时）：`1.0931`

**Test（overall）**
- Accuracy：`0.3608`
- F1-macro：`0.3406`
- Recall-macro：`0.3465`
- Recall idle/left/right：`0.4035` / `0.3760` / `0.2600`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    807    679    514
  true1    384    376    240
  true2    403    337    260
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

- 结束：`2026-08-06T00:51:36`
