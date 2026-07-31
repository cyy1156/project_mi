# 被试独立五折实验记录（20260731_114237 / conformer）

- 开始：`2026-07-31T11:42:37`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`conformer`（单脚本；无 registry）
- 结构：EEGConformer（num_layers=2, num_heads=10, att_drop=0.5 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\conformer\merged_2s\run_20260731_114237`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.8319 ± 0.0133`
- Test F1：`0.8272 ± 0.0177`
- Test Acc：`0.7137 ± 0.0228`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val F1（最优）：`0.8504`
- Val loss（最优时）：`0.5405`

**Test（overall）**
- Accuracy：`0.7149`
- Recall：`0.9713`
- Specificity：`0.0817`
- Precision：`0.7232`
- F1：`0.8290`
- Balanced Acc：`0.5265`
- 混淆矩阵：TP=`4903` TN=`167` FP=`1877` FN=`145`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7020` F1=`0.8156` BalAcc=`0.5547`
- `stieger_only`：Acc=`0.7157` F1=`0.8298` BalAcc=`0.5246`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.8106`
- Val loss（最优时）：`0.7217`

**Test（overall）**
- Accuracy：`0.6995`
- Recall：`0.8635`
- Specificity：`0.2813`
- Precision：`0.7539`
- F1：`0.8050`
- Balanced Acc：`0.5724`
- 混淆矩阵：TP=`3764` TN=`481` FP=`1229` FN=`595`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6658` F1=`0.7335` BalAcc=`0.6580`
- `stieger_only`：Acc=`0.7018` F1=`0.8090` BalAcc=`0.5640`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.8279`
- Val loss（最优时）：`0.5940`

**Test（overall）**
- Accuracy：`0.7566`
- Recall：`0.9882`
- Specificity：`0.0760`
- Precision：`0.7586`
- F1：`0.8583`
- Balanced Acc：`0.5321`
- 混淆矩阵：TP=`4699` TN=`123` FP=`1495` FN=`56`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6744` F1=`0.8049` BalAcc=`0.5039`
- `stieger_only`：Acc=`0.7620` F1=`0.8617` BalAcc=`0.5347`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val F1（最优）：`0.8299`
- Val loss（最优时）：`0.5919`

**Test（overall）**
- Accuracy：`0.6915`
- Recall：`0.9966`
- Specificity：`0.0063`
- Precision：`0.6925`
- F1：`0.8172`
- Balanced Acc：`0.5015`
- 混淆矩阵：TP=`5650` TN=`16` FP=`2509` FN=`19`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6766` F1=`0.8071` BalAcc=`0.5000`
- `stieger_only`：Acc=`0.6922` F1=`0.8177` BalAcc=`0.5016`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.8406`
- Val loss（最优时）：`0.6159`

**Test（overall）**
- Accuracy：`0.7058`
- Recall：`0.9895`
- Specificity：`0.0189`
- Precision：`0.7095`
- F1：`0.8264`
- Balanced Acc：`0.5042`
- 混淆矩阵：TP=`5832` TN=`46` FP=`2388` FN=`62`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6826` F1=`0.7954` BalAcc=`0.5518`
- `stieger_only`：Acc=`0.7063` F1=`0.8270` BalAcc=`0.5031`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4478 ± 0.0437`
- Test F1-macro：`0.4146 ± 0.0482`
- Test Acc：`0.4442 ± 0.0511`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val F1-macro（最优）：`0.5245`
- Val loss（最优时）：`1.0145`

**Test（overall）**
- Accuracy：`0.4179`
- F1-macro：`0.4047`
- Recall-macro：`0.4077`
- Recall idle/left/right：`0.2564` / `0.4852` / `0.4815`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    524    779    741
  true1    418   1216    872
  true2    336    982   1224
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5429` F1m=`0.5224`
- `stieger_only`：Acc=`0.4105` F1m=`0.3968`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1-macro（最优）：`0.3896`
- Val loss（最优时）：`1.1393`

**Test（overall）**
- Accuracy：`0.4220`
- F1-macro：`0.4191`
- Recall-macro：`0.4241`
- Recall idle/left/right：`0.4784` / `0.3301` / `0.4637`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    818    467    425
  true1    784    688    612
  true2    643    577   1055
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5302` F1m=`0.4796`
- `stieger_only`：Acc=`0.4144` F1m=`0.4122`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`71`
- 验证最优轮次（best_epoch）：`53`
- Val F1-macro（最优）：`0.4436`
- Val loss（最优时）：`1.0944`

**Test（overall）**
- Accuracy：`0.5393`
- F1-macro：`0.5035`
- Recall-macro：`0.5112`
- Recall idle/left/right：`0.2602` / `0.5101` / `0.7633`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    421    337    860
  true1    233   1236    954
  true2    141    411   1780
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5051` F1m=`0.4634`
- `stieger_only`：Acc=`0.5415` F1m=`0.5060`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1-macro（最优）：`0.4323`
- Val loss（最优时）：`1.0832`

**Test（overall）**
- Accuracy：`0.3915`
- F1-macro：`0.3656`
- Recall-macro：`0.3851`
- Recall idle/left/right：`0.1861` / `0.3326` / `0.6366`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    470    763   1292
  true1    417    953   1495
  true2    324    695   1785
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4484` F1m=`0.4280`
- `stieger_only`：Acc=`0.3888` F1m=`0.3612`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`56`
- 验证最优轮次（best_epoch）：`38`
- Val F1-macro（最优）：`0.4491`
- Val loss（最优时）：`1.0949`

**Test（overall）**
- Accuracy：`0.4500`
- F1-macro：`0.3801`
- Recall-macro：`0.4275`
- Recall idle/left/right：`0.0608` / `0.7215` / `0.5003`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    148   1541    745
  true1    106   2124    714
  true2    115   1359   1476
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5988` F1m=`0.5802`
- `stieger_only`：Acc=`0.4470` F1m=`0.3730`

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

- 结束：`2026-07-31T12:13:22`
