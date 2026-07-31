# 被试独立五折实验记录（20260730_223118 / eegnet）

- 开始：`2026-07-30T22:31:18`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`eegnet`（单脚本；无 registry）
- 结构：F1=8, D=2, F2=16
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegnet\merged_2s\run_20260730_223118`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8316 ± 0.0121`
- Test F1：`0.8289 ± 0.0179`
- Test Acc：`0.7153 ± 0.0226`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val F1（最优）：`0.8469`
- Val loss（最优时）：`0.6086`

**Test（overall）**
- Accuracy：`0.7125`
- Recall：`0.9950`
- Specificity：`0.0147`
- Precision：`0.7138`
- F1：`0.8313`
- Balanced Acc：`0.5049`
- 混淆矩阵：TP=`5023` TN=`30` FP=`2014` FN=`25`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6768` F1=`0.8061` BalAcc=`0.5059`
- `stieger_only`：Acc=`0.7146` F1=`0.8327` BalAcc=`0.5048`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val F1（最优）：`0.8118`
- Val loss（最优时）：`0.6172`

**Test（overall）**
- Accuracy：`0.7047`
- Recall：`0.8557`
- Specificity：`0.3199`
- Precision：`0.7623`
- F1：`0.8063`
- Balanced Acc：`0.5878`
- 混淆矩阵：TP=`3730` TN=`547` FP=`1163` FN=`629`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6005` F1=`0.6294` BalAcc=`0.6540`
- `stieger_only`：Acc=`0.7120` F1=`0.8149` BalAcc=`0.5796`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val F1（最优）：`0.8284`
- Val loss（最优时）：`0.6190`

**Test（overall）**
- Accuracy：`0.7585`
- Recall：`0.9958`
- Specificity：`0.0612`
- Precision：`0.7571`
- F1：`0.8602`
- Balanced Acc：`0.5285`
- 混淆矩阵：TP=`4735` TN=`99` FP=`1519` FN=`20`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6744` F1=`0.8049` BalAcc=`0.5039`
- `stieger_only`：Acc=`0.7640` F1=`0.8637` BalAcc=`0.5307`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val F1（最优）：`0.8299`
- Val loss（最优时）：`0.6044`

**Test（overall）**
- Accuracy：`0.6929`
- Recall：`0.9996`
- Specificity：`0.0044`
- Precision：`0.6927`
- F1：`0.8183`
- Balanced Acc：`0.5020`
- 混淆矩阵：TP=`5667` TN=`11` FP=`2514` FN=`2`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7011` F1=`0.8185` BalAcc=`0.5400`
- `stieger_only`：Acc=`0.6926` F1=`0.8183` BalAcc=`0.5001`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val F1（最优）：`0.8410`
- Val loss（最优时）：`0.5903`

**Test（overall）**
- Accuracy：`0.7076`
- Recall：`0.9983`
- Specificity：`0.0037`
- Precision：`0.7081`
- F1：`0.8286`
- Balanced Acc：`0.5010`
- 混淆矩阵：TP=`5884` TN=`9` FP=`2425` FN=`10`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6946` F1=`0.8145` BalAcc=`0.5220`
- `stieger_only`：Acc=`0.7079` F1=`0.8288` BalAcc=`0.5005`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4375 ± 0.0193`
- Test F1-macro：`0.4118 ± 0.0502`
- Test Acc：`0.4321 ± 0.0519`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val F1-macro（最优）：`0.4328`
- Val loss（最优时）：`1.0759`

**Test（overall）**
- Accuracy：`0.3982`
- F1-macro：`0.3869`
- Recall-macro：`0.3913`
- Recall idle/left/right：`0.2808` / `0.5519` / `0.3411`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    574    936    534
  true1    422   1383    701
  true2    396   1279    867
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5126` F1m=`0.4906`
- `stieger_only`：Acc=`0.3914` F1m=`0.3786`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`46`
- 验证最优轮次（best_epoch）：`28`
- Val F1-macro（最优）：`0.4435`
- Val loss（最优时）：`1.0464`

**Test（overall）**
- Accuracy：`0.3880`
- F1-macro：`0.3798`
- Recall-macro：`0.4088`
- Recall idle/left/right：`0.6626` / `0.3186` / `0.2453`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1133    425    152
  true1   1211    664    209
  true2   1095    622    558
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4296` F1m=`0.3562`
- `stieger_only`：Acc=`0.3851` F1m=`0.3785`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`59`
- 验证最优轮次（best_epoch）：`41`
- Val F1-macro（最优）：`0.4063`
- Val loss（最优时）：`1.0813`

**Test（overall）**
- Accuracy：`0.5285`
- F1-macro：`0.5063`
- Recall-macro：`0.5086`
- Recall idle/left/right：`0.3239` / `0.4503` / `0.7517`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    524    357    737
  true1    205   1091   1127
  true2    104    475   1753
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4282` F1m=`0.3824`
- `stieger_only`：Acc=`0.5350` F1m=`0.5141`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`36`
- 验证最优轮次（best_epoch）：`18`
- Val F1-macro（最优）：`0.4663`
- Val loss（最优时）：`1.0351`

**Test（overall）**
- Accuracy：`0.4016`
- F1-macro：`0.3672`
- Recall-macro：`0.3934`
- Recall idle/left/right：`0.1465` / `0.3749` / `0.6587`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    370    881   1274
  true1    283   1074   1508
  true2    246    711   1847
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4348` F1m=`0.3859`
- `stieger_only`：Acc=`0.4001` F1m=`0.3651`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val F1-macro（最优）：`0.4386`
- Val loss（最优时）：`1.0412`

**Test（overall）**
- Accuracy：`0.4440`
- F1-macro：`0.4187`
- Recall-macro：`0.4312`
- Recall idle/left/right：`0.2243` / `0.4168` / `0.6525`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    546    869   1019
  true1    425   1227   1292
  true2    342    683   1925
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5569` F1m=`0.5406`
- `stieger_only`：Acc=`0.4417` F1m=`0.4154`

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

- 结束：`2026-07-30T23:24:25`
