# 被试独立五折实验记录（20260731_105045 / eegtcnet）

- 开始：`2026-07-31T10:50:45`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`eegtcnet`（单脚本；无 registry）
- 结构：EEGTCNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegtcnet\merged_2s\run_20260731_105045`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8317 ± 0.0130`
- Test F1：`0.8325 ± 0.0135`
- Test Acc：`0.7170 ± 0.0199`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val F1（最优）：`0.8459`
- Val loss（最优时）：`0.5767`

**Test（overall）**
- Accuracy：`0.7116`
- Recall：`0.9988`
- Specificity：`0.0024`
- Precision：`0.7120`
- F1：`0.8314`
- Balanced Acc：`0.5006`
- 混淆矩阵：TP=`5042` TN=`5` FP=`2039` FN=`6`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6742` F1=`0.8054` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.7139` F1=`0.8329` BalAcc=`0.5007`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`36`
- 验证最优轮次（best_epoch）：`18`
- Val F1（最优）：`0.8097`
- Val loss（最优时）：`0.6254`

**Test（overall）**
- Accuracy：`0.7243`
- Recall：`0.9530`
- Specificity：`0.1415`
- Precision：`0.7389`
- F1：`0.8324`
- Balanced Acc：`0.5472`
- 混淆矩阵：TP=`4154` TN=`242` FP=`1468` FN=`205`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6759` F1=`0.7717` BalAcc=`0.6029`
- `stieger_only`：Acc=`0.7277` F1=`0.8360` BalAcc=`0.5416`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val F1（最优）：`0.8265`
- Val loss（最优时）：`0.6084`

**Test（overall）**
- Accuracy：`0.7515`
- Recall：`0.9975`
- Specificity：`0.0284`
- Precision：`0.7511`
- F1：`0.8569`
- Balanced Acc：`0.5130`
- 混淆矩阵：TP=`4743` TN=`46` FP=`1572` FN=`12`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6718` F1=`0.8037` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.7566` F1=`0.8602` BalAcc=`0.5141`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val F1（最优）：`0.8333`
- Val loss（最优时）：`0.5797`

**Test（overall）**
- Accuracy：`0.6932`
- Recall：`0.9861`
- Specificity：`0.0356`
- Precision：`0.6966`
- F1：`0.8164`
- Balanced Acc：`0.5109`
- 混淆矩阵：TP=`5590` TN=`90` FP=`2435` FN=`79`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6848` F1=`0.8073` BalAcc=`0.5258`
- `stieger_only`：Acc=`0.6936` F1=`0.8168` BalAcc=`0.5101`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`40`
- 验证最优轮次（best_epoch）：`22`
- Val F1（最优）：`0.8432`
- Val loss（最优时）：`0.5659`

**Test（overall）**
- Accuracy：`0.7046`
- Recall：`0.9859`
- Specificity：`0.0234`
- Precision：`0.7097`
- F1：`0.8253`
- Balanced Acc：`0.5047`
- 混淆矩阵：TP=`5811` TN=`57` FP=`2377` FN=`83`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5389` F1=`0.6452` BalAcc=`0.4978`
- `stieger_only`：Acc=`0.7080` F1=`0.8281` BalAcc=`0.5046`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4220 ± 0.0396`
- Test F1-macro：`0.3988 ± 0.0681`
- Test Acc：`0.4223 ± 0.0633`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`115`
- 验证最优轮次（best_epoch）：`97`
- Val F1-macro（最优）：`0.4975`
- Val loss（最优时）：`1.0373`

**Test（overall）**
- Accuracy：`0.4027`
- F1-macro：`0.3761`
- Recall-macro：`0.3902`
- Recall idle/left/right：`0.1952` / `0.6293` / `0.3462`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    399   1104    541
  true1    245   1577    684
  true2    248   1414    880
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5404` F1m=`0.5197`
- `stieger_only`：Acc=`0.3946` F1m=`0.3657`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`107`
- 验证最优轮次（best_epoch）：`89`
- Val F1-macro（最优）：`0.4108`
- Val loss（最优时）：`1.0597`

**Test（overall）**
- Accuracy：`0.4464`
- F1-macro：`0.4448`
- Recall-macro：`0.4523`
- Recall idle/left/right：`0.4959` / `0.5134` / `0.3477`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    848    609    253
  true1    683   1070    331
  true2    705    779    791
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5628` F1m=`0.5549`
- `stieger_only`：Acc=`0.4382` F1m=`0.4356`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`139`
- 验证最优轮次（best_epoch）：`121`
- Val F1-macro（最优）：`0.4211`
- Val loss（最优时）：`1.0756`

**Test（overall）**
- Accuracy：`0.5332`
- F1-macro：`0.5073`
- Recall-macro：`0.5104`
- Recall idle/left/right：`0.3010` / `0.4651` / `0.7650`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    487    375    756
  true1    154   1127   1142
  true2     93    455   1784
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4359` F1m=`0.3618`
- `stieger_only`：Acc=`0.5395` F1m=`0.5163`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`70`
- 验证最优轮次（best_epoch）：`52`
- Val F1-macro（最优）：`0.3931`
- Val loss（最优时）：`1.0697`

**Test（overall）**
- Accuracy：`0.3571`
- F1-macro：`0.3263`
- Recall-macro：`0.3503`
- Recall idle/left/right：`0.1406` / `0.3012` / `0.6091`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    355    732   1438
  true1    261    863   1741
  true2    273    823   1708
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3261` F1m=`0.3092`
- `stieger_only`：Acc=`0.3585` F1m=`0.3269`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`47`
- 验证最优轮次（best_epoch）：`29`
- Val F1-macro（最优）：`0.3875`
- Val loss（最优时）：`1.0687`

**Test（overall）**
- Accuracy：`0.3722`
- F1-macro：`0.3397`
- Recall-macro：`0.3598`
- Recall idle/left/right：`0.1598` / `0.2948` / `0.6247`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    389    815   1230
  true1    393    868   1683
  true2    311    796   1843
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.2575` F1m=`0.1867`
- `stieger_only`：Acc=`0.3746` F1m=`0.3388`

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

- 结束：`2026-07-31T11:42:26`
