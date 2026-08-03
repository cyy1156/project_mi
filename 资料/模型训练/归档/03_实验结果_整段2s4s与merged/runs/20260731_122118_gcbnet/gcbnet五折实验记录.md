# 被试独立五折实验记录（20260731_122118 / gcbnet）

- 开始：`2026-07-31T12:21:18`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`gcbnet`（单脚本；无 registry）
- 输入：bandpower 立方体 `(36056, 8, 2)`（非时域 500）
- 结构：GCBNet(k=2, layers=[128], dropout=shared drop_prob)；8 导联偶数
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\gcbnet\merged_2s\run_20260731_122118`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8302 ± 0.0126`
- Test F1：`0.8324 ± 0.0129`
- Test Acc：`0.7142 ± 0.0190`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val F1（最优）：`0.8459`
- Val loss（最优时）：`0.5829`

**Test（overall）**
- Accuracy：`0.7052`
- Recall：`0.9822`
- Specificity：`0.0210`
- Precision：`0.7125`
- F1：`0.8259`
- Balanced Acc：`0.5016`
- 混淆矩阵：TP=`4958` TN=`43` FP=`2001` FN=`90`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6742` F1=`0.8054` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.7070` F1=`0.8270` BalAcc=`0.5018`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.8096`
- Val loss（最优时）：`0.6253`

**Test（overall）**
- Accuracy：`0.7207`
- Recall：`0.9945`
- Specificity：`0.0228`
- Precision：`0.7218`
- F1：`0.8365`
- Balanced Acc：`0.5087`
- 混淆矩阵：TP=`4335` TN=`39` FP=`1671` FN=`24`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6960` F1=`0.8076` BalAcc=`0.5613`
- `stieger_only`：Acc=`0.7224` F1=`0.8383` BalAcc=`0.5040`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val F1（最优）：`0.8262`
- Val loss（最优时）：`0.6090`

**Test（overall）**
- Accuracy：`0.7475`
- Recall：`1.0000`
- Specificity：`0.0056`
- Precision：`0.7472`
- F1：`0.8553`
- Balanced Acc：`0.5028`
- 混淆矩阵：TP=`4755` TN=`9` FP=`1609` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6718` F1=`0.8037` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.7525` F1=`0.8585` BalAcc=`0.5030`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.8286`
- Val loss（最优时）：`0.6106`

**Test（overall）**
- Accuracy：`0.6921`
- Recall：`1.0000`
- Specificity：`0.0008`
- Precision：`0.6920`
- F1：`0.8180`
- Balanced Acc：`0.5004`
- 混淆矩阵：TP=`5669` TN=`2` FP=`2523` FN=`0`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6821` F1=`0.8098` BalAcc=`0.5084`
- `stieger_only`：Acc=`0.6926` F1=`0.8184` BalAcc=`0.5000`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val F1（最优）：`0.8407`
- Val loss（最优时）：`0.5820`

**Test（overall）**
- Accuracy：`0.7053`
- Recall：`0.9898`
- Specificity：`0.0164`
- Precision：`0.7090`
- F1：`0.8262`
- Balanced Acc：`0.5031`
- 混淆矩阵：TP=`5834` TN=`40` FP=`2394` FN=`60`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7485` F1=`0.8306` BalAcc=`0.6597`
- `stieger_only`：Acc=`0.7044` F1=`0.8261` BalAcc=`0.4997`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.3769 ± 0.0138`
- Test F1-macro：`0.3696 ± 0.0255`
- Test Acc：`0.3783 ± 0.0256`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`40`
- Val F1-macro（最优）：`0.4008`
- Val loss（最优时）：`1.0799`

**Test（overall）**
- Accuracy：`0.3703`
- F1-macro：`0.3648`
- Recall-macro：`0.3714`
- Recall idle/left/right：`0.3796` / `0.4713` / `0.2632`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    776    896    372
  true1    793   1181    532
  true2    825   1048    669
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4217` F1m=`0.4095`
- `stieger_only`：Acc=`0.3672` F1m=`0.3620`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val F1-macro（最优）：`0.3708`
- Val loss（最优时）：`1.0907`

**Test（overall）**
- Accuracy：`0.3704`
- F1-macro：`0.3697`
- Recall-macro：`0.3803`
- Recall idle/left/right：`0.5012` / `0.3369` / `0.3029`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    857    453    400
  true1    974    702    408
  true2    928    658    689
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4749` F1m=`0.4652`
- `stieger_only`：Acc=`0.3631` F1m=`0.3625`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val F1-macro（最优）：`0.3702`
- Val loss（最优时）：`1.0925`

**Test（overall）**
- Accuracy：`0.4288`
- F1-macro：`0.4169`
- Recall-macro：`0.4201`
- Recall idle/left/right：`0.3319` / `0.3409` / `0.5875`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    537    380    701
  true1    339    826   1258
  true2    364    598   1370
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3846` F1m=`0.3816`
- `stieger_only`：Acc=`0.4317` F1m=`0.4186`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val F1-macro（最优）：`0.3819`
- Val loss（最优时）：`1.0843`

**Test（overall）**
- Accuracy：`0.3601`
- F1-macro：`0.3419`
- Recall-macro：`0.3536`
- Recall idle/left/right：`0.1711` / `0.3923` / `0.4975`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    432    971   1122
  true1    463   1124   1278
  true2    420    989   1395
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4565` F1m=`0.4526`
- `stieger_only`：Acc=`0.3556` F1m=`0.3338`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val F1-macro（最优）：`0.3604`
- Val loss（最优时）：`1.0829`

**Test（overall）**
- Accuracy：`0.3620`
- F1-macro：`0.3546`
- Recall-macro：`0.3562`
- Recall idle/left/right：`0.2617` / `0.3645` / `0.4424`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    637    791   1006
  true1    755   1073   1116
  true2    656    989   1305
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4192` F1m=`0.3659`
- `stieger_only`：Acc=`0.3609` F1m=`0.3521`

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

- 结束：`2026-07-31T12:38:04`
