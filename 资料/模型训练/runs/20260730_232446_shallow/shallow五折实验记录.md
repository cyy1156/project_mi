# 被试独立五折实验记录（20260730_232446 / shallow）

- 开始：`2026-07-30T23:24:46`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`shallow`（单脚本；无 registry）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow\merged_2s\run_20260730_232446`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8274 ± 0.0104`
- Test F1：`0.8237 ± 0.0198`
- Test Acc：`0.7108 ± 0.0279`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val F1（最优）：`0.8355`
- Val loss（最优时）：`0.5607`

**Test（overall）**
- Accuracy：`0.6981`
- Recall：`0.9251`
- Specificity：`0.1375`
- Precision：`0.7259`
- F1：`0.8135`
- Balanced Acc：`0.5313`
- 混淆矩阵：TP=`4670` TN=`281` FP=`1763` FN=`378`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6566` F1=`0.7748` BalAcc=`0.5390`
- `stieger_only`：Acc=`0.7006` F1=`0.8157` BalAcc=`0.5305`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val F1（最优）：`0.8108`
- Val loss（最优时）：`0.7069`

**Test（overall）**
- Accuracy：`0.7034`
- Recall：`0.8759`
- Specificity：`0.2637`
- Precision：`0.7520`
- F1：`0.8092`
- Balanced Acc：`0.5698`
- 混淆矩阵：TP=`3818` TN=`451` FP=`1259` FN=`541`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6709` F1=`0.7579` BalAcc=`0.6214`
- `stieger_only`：Acc=`0.7057` F1=`0.8124` BalAcc=`0.5647`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val F1（最优）：`0.8262`
- Val loss（最优时）：`0.5966`

**Test（overall）**
- Accuracy：`0.7646`
- Recall：`0.9813`
- Specificity：`0.1279`
- Precision：`0.7678`
- F1：`0.8615`
- Balanced Acc：`0.5546`
- 混淆矩阵：TP=`4666` TN=`207` FP=`1411` FN=`89`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6744` F1=`0.8049` BalAcc=`0.5039`
- `stieger_only`：Acc=`0.7705` F1=`0.8651` BalAcc=`0.5592`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.8233`
- Val loss（最优时）：`0.5969`

**Test（overall）**
- Accuracy：`0.6834`
- Recall：`0.9682`
- Specificity：`0.0440`
- Precision：`0.6945`
- F1：`0.8089`
- Balanced Acc：`0.5061`
- 混淆矩阵：TP=`5489` TN=`111` FP=`2414` FN=`180`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6984` F1=`0.8153` BalAcc=`0.5424`
- `stieger_only`：Acc=`0.6827` F1=`0.8086` BalAcc=`0.5043`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val F1（最优）：`0.8411`
- Val loss（最优时）：`0.5754`

**Test（overall）**
- Accuracy：`0.7043`
- Recall：`0.9864`
- Specificity：`0.0210`
- Precision：`0.7093`
- F1：`0.8252`
- Balanced Acc：`0.5037`
- 混淆矩阵：TP=`5814` TN=`51` FP=`2383` FN=`80`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6826` F1=`0.7905` BalAcc=`0.5683`
- `stieger_only`：Acc=`0.7047` F1=`0.8258` BalAcc=`0.5022`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4513 ± 0.0299`
- Test F1-macro：`0.4202 ± 0.0428`
- Test Acc：`0.4379 ± 0.0412`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val F1-macro（最优）：`0.5034`
- Val loss（最优时）：`1.0144`

**Test（overall）**
- Accuracy：`0.4003`
- F1-macro：`0.3996`
- Recall-macro：`0.4009`
- Recall idle/left/right：`0.4080` / `0.4154` / `0.3792`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    834    628    582
  true1    754   1041    711
  true2    685    893    964
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5328` F1m=`0.5260`
- `stieger_only`：Acc=`0.3925` F1m=`0.3918`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val F1-macro（最优）：`0.4173`
- Val loss（最优时）：`1.0884`

**Test（overall）**
- Accuracy：`0.4449`
- F1-macro：`0.4451`
- Recall-macro：`0.4517`
- Recall idle/left/right：`0.5246` / `0.4544` / `0.3763`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    897    521    292
  true1    792    947    345
  true2    729    690    856
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5779` F1m=`0.5727`
- `stieger_only`：Acc=`0.4355` F1m=`0.4355`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val F1-macro（最优）：`0.4285`
- Val loss（最优时）：`1.0876`

**Test（overall）**
- Accuracy：`0.5093`
- F1-macro：`0.4885`
- Recall-macro：`0.4957`
- Recall idle/left/right：`0.3597` / `0.3805` / `0.7470`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    582    274    762
  true1    330    922   1171
  true2    223    367   1742
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4769` F1m=`0.4326`
- `stieger_only`：Acc=`0.5114` F1m=`0.4912`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val F1-macro（最优）：`0.4478`
- Val loss（最优时）：`1.0776`

**Test（overall）**
- Accuracy：`0.3939`
- F1-macro：`0.3634`
- Recall-macro：`0.3855`
- Recall idle/left/right：`0.1446` / `0.4129` / `0.5991`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    365    992   1168
  true1    305   1183   1377
  true2    277    847   1680
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4076` F1m=`0.3667`
- `stieger_only`：Acc=`0.3933` F1m=`0.3622`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val F1-macro（最优）：`0.4596`
- Val loss（最优时）：`1.0422`

**Test（overall）**
- Accuracy：`0.4408`
- F1-macro：`0.4045`
- Recall-macro：`0.4241`
- Recall idle/left/right：`0.1528` / `0.5965` / `0.5231`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    372   1202    860
  true1    275   1756    913
  true2    259   1148   1543
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5928` F1m=`0.6006`
- `stieger_only`：Acc=`0.4377` F1m=`0.3987`

### 共用超参
```json
{
  "data_tag": "merged_2s",
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

- 结束：`2026-07-31T10:12:15`
