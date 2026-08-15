# 被试独立五折实验记录（20260806_012329 / gcbnet）

- 开始：`2026-08-06T01:23:29`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`gcbnet`（单脚本；无 registry）
- 输入：bandpower 立方体 `(21600, 8, 2)`（非时域 500）
- 结构：GCBNet(k=2, layers=[128], dropout=shared drop_prob)；8 导联偶数
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\gcbnet\openbmi_balanced_train_2s\run_20260806_012329`

---
## 最终结论

### Task（静息/任务）
- Val Balanced Acc：`0.5376 ± 0.0176`
- Test F1：`0.5502 ± 0.0488`
- Test Acc：`0.5439 ± 0.0135`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val F1（最优）：`0.5567`
- Val loss（最优时）：`0.6864`

**Test（overall）**
- Accuracy：`0.5423`
- Recall：`0.5714`
- Specificity：`0.5132`
- Precision：`0.5399`
- F1：`0.5552`
- Balanced Acc：`0.5423`
- 混淆矩阵：TP=`1257` TN=`1129` FP=`1071` FN=`943`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`74`
- 验证最优轮次（best_epoch）：`56`
- Val F1（最优）：`0.5210`
- Val loss（最优时）：`0.6930`

**Test（overall）**
- Accuracy：`0.5684`
- Recall：`0.6077`
- Specificity：`0.5291`
- Precision：`0.5634`
- F1：`0.5847`
- Balanced Acc：`0.5684`
- 混淆矩阵：TP=`1337` TN=`1164` FP=`1036` FN=`863`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val F1（最优）：`0.5126`
- Val loss（最优时）：`0.6973`

**Test（overall）**
- Accuracy：`0.5425`
- Recall：`0.4414`
- Specificity：`0.6436`
- Precision：`0.5533`
- F1：`0.4910`
- Balanced Acc：`0.5425`
- 混淆矩阵：TP=`971` TN=`1416` FP=`784` FN=`1229`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.6136`
- Val loss（最优时）：`0.6852`

**Test（overall）**
- Accuracy：`0.5393`
- Recall：`0.7491`
- Specificity：`0.3295`
- Precision：`0.5277`
- F1：`0.6192`
- Balanced Acc：`0.5393`
- 混淆矩阵：TP=`1648` TN=`725` FP=`1475` FN=`552`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`39`
- 验证最优轮次（best_epoch）：`21`
- Val F1（最优）：`0.5042`
- Val loss（最优时）：`0.7036`

**Test（overall）**
- Accuracy：`0.5270`
- Recall：`0.4745`
- Specificity：`0.5795`
- Precision：`0.5302`
- F1：`0.5008`
- Balanced Acc：`0.5270`
- 混淆矩阵：TP=`949` TN=`1159` FP=`841` FN=`1051`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.3398 ± 0.0194`
- Test F1-macro：`0.3450 ± 0.0153`
- Test Acc：`0.4745 ± 0.0110`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`150`
- 验证最优轮次（best_epoch）：`132`
- Val F1-macro（最优）：`0.3070`
- Val loss（最优时）：`1.0617`

**Test（overall）**
- Accuracy：`0.4830`
- F1-macro：`0.3374`
- Recall-macro：`0.3650`
- Recall idle/left/right：`0.8368` / `0.1082` / `0.1500`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1841    191    168
  true1    877    119    104
  true2    857     78    165
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`122`
- 验证最优轮次（best_epoch）：`104`
- Val F1-macro（最优）：`0.3649`
- Val loss（最优时）：`1.0413`

**Test（overall）**
- Accuracy：`0.4757`
- F1-macro：`0.3689`
- Recall-macro：`0.3833`
- Recall idle/left/right：`0.7527` / `0.1227` / `0.2745`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1656    180    364
  true1    725    135    240
  true2    723     75    302
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`142`
- 验证最优轮次（best_epoch）：`124`
- Val F1-macro（最优）：`0.3524`
- Val loss（最优时）：`1.0880`

**Test（overall）**
- Accuracy：`0.4577`
- F1-macro：`0.3274`
- Recall-macro：`0.3503`
- Recall idle/left/right：`0.7800` / `0.1091` / `0.1618`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1716    199    285
  true1    814    120    166
  true2    822    100    178
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`148`
- 验证最优轮次（best_epoch）：`130`
- Val F1-macro（最优）：`0.3371`
- Val loss（最优时）：`1.0348`

**Test（overall）**
- Accuracy：`0.4886`
- F1-macro：`0.3563`
- Recall-macro：`0.3768`
- Recall idle/left/right：`0.8241` / `0.1545` / `0.1518`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1813    210    177
  true1    816    170    114
  true2    774    159    167
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`98`
- 验证最优轮次（best_epoch）：`80`
- Val F1-macro（最优）：`0.3378`
- Val loss（最优时）：`1.0718`

**Test（overall）**
- Accuracy：`0.4672`
- F1-macro：`0.3348`
- Recall-macro：`0.3575`
- Recall idle/left/right：`0.7965` / `0.1620` / `0.1140`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1593    192    215
  true1    745    162     93
  true2    768    118    114
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

- 结束：`2026-08-06T01:40:20`
