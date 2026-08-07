# 被试独立五折实验记录（20260805_191656 / dgcnn_raw）

- 开始：`2026-08-05T19:16:56`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`dgcnn_raw`（TemporalEncoder + DGCNN）
- 输入：raw `(21600, 8, 500)` → Encoder → 节点特征 D=64（非 bandpower）
- 结构：DGCNNRaw(k=2, layers=[128], in_channels=64, relu_is=1)
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\dgcnn_raw\openbmi_balanced_train_2s\run_20260805_191656`

---
## 最终结论

### Task（静息/任务）
- Val F1：`0.6075 ± 0.0345`
- Test F1：`0.5910 ± 0.0376`
- Test Acc：`0.5655 ± 0.0355`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.5914`
- Val loss（最优时）：`0.8205`

**Test（overall）**
- Accuracy：`0.5109`
- Recall：`0.6105`
- Specificity：`0.4114`
- Precision：`0.5091`
- F1：`0.5552`
- Balanced Acc：`0.5109`
- 混淆矩阵：TP=`1343` TN=`905` FP=`1295` FN=`857`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`122`
- 验证最优轮次（best_epoch）：`104`
- Val F1（最优）：`0.6304`
- Val loss（最优时）：`0.6706`

**Test（overall）**
- Accuracy：`0.6207`
- Recall：`0.5768`
- Specificity：`0.6645`
- Precision：`0.6323`
- F1：`0.6033`
- Balanced Acc：`0.6207`
- 混淆矩阵：TP=`1269` TN=`1462` FP=`738` FN=`931`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.6570`
- Val loss（最优时）：`0.6860`

**Test（overall）**
- Accuracy：`0.5534`
- Recall：`0.8368`
- Specificity：`0.2700`
- Precision：`0.5341`
- F1：`0.6520`
- Balanced Acc：`0.5534`
- 混淆矩阵：TP=`1841` TN=`594` FP=`1606` FN=`359`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.5555`
- Val loss（最优时）：`0.6857`

**Test（overall）**
- Accuracy：`0.5770`
- Recall：`0.5118`
- Specificity：`0.6423`
- Precision：`0.5886`
- F1：`0.5475`
- Balanced Acc：`0.5770`
- 混淆矩阵：TP=`1126` TN=`1413` FP=`787` FN=`1074`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`91`
- 验证最优轮次（best_epoch）：`73`
- Val F1（最优）：`0.6034`
- Val loss（最优时）：`0.7637`

**Test（overall）**
- Accuracy：`0.5655`
- Recall：`0.6440`
- Specificity：`0.4870`
- Precision：`0.5566`
- F1：`0.5971`
- Balanced Acc：`0.5655`
- 混淆矩阵：TP=`1288` TN=`974` FP=`1026` FN=`712`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4359 ± 0.0181`
- Test F1-macro：`0.4358 ± 0.0072`
- Test Acc：`0.4996 ± 0.0297`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`139`
- 验证最优轮次（best_epoch）：`121`
- Val F1-macro（最优）：`0.4334`
- Val loss（最优时）：`1.0076`

**Test（overall）**
- Accuracy：`0.5202`
- F1-macro：`0.4388`
- Recall-macro：`0.4380`
- Recall idle/left/right：`0.7668` / `0.2827` / `0.2645`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1687    243    270
  true1    630    311    159
  true2    659    150    291
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`144`
- 验证最优轮次（best_epoch）：`126`
- Val F1-macro（最优）：`0.4486`
- Val loss（最优时）：`1.0057`

**Test（overall）**
- Accuracy：`0.5345`
- F1-macro：`0.4440`
- Recall-macro：`0.4453`
- Recall idle/left/right：`0.8023` / `0.2745` / `0.2591`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1765    175    260
  true1    610    302    188
  true2    629    186    285
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`114`
- 验证最优轮次（best_epoch）：`96`
- Val F1-macro（最优）：`0.4484`
- Val loss（最优时）：`1.0176`

**Test（overall）**
- Accuracy：`0.4536`
- F1-macro：`0.4325`
- Recall-macro：`0.4400`
- Recall idle/left/right：`0.4945` / `0.3618` / `0.4636`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1088    520    592
  true1    344    398    358
  true2    361    229    510
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`151`
- 验证最优轮次（best_epoch）：`133`
- Val F1-macro（最优）：`0.4477`
- Val loss（最优时）：`0.9476`

**Test（overall）**
- Accuracy：`0.5123`
- F1-macro：`0.4401`
- Recall-macro：`0.4391`
- Recall idle/left/right：`0.7318` / `0.2900` / `0.2955`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1610    302    288
  true1    586    319    195
  true2    566    209    325
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`116`
- 验证最优轮次（best_epoch）：`98`
- Val F1-macro（最优）：`0.4015`
- Val loss（最优时）：`1.1247`

**Test（overall）**
- Accuracy：`0.4775`
- F1-macro：`0.4234`
- Recall-macro：`0.4218`
- Recall idle/left/right：`0.6445` / `0.3450` / `0.2760`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1289    375    336
  true1    501    345    154
  true2    538    186    276
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

- 结束：`2026-08-05T19:46:00`
