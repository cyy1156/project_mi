# 被试独立五折实验记录（20260806_012025 / dbn）

- 开始：`2026-08-06T01:20:25`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`dbn`（单脚本；无 registry）
- 输入：bandpower 立方体 `(21600, 8, 2)`（非时域 500）
- 结构：DBN(hidden 300/400)；监督 forward，无 RBM 预训练；drop_prob 忽略
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\dbn\openbmi_balanced_train_2s\run_20260806_012025`

---
## 最终结论

### Task（静息/任务）
- Val Balanced Acc：`0.5363 ± 0.0126`
- Test F1：`0.5114 ± 0.1181`
- Test Acc：`0.5266 ± 0.0139`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val F1（最优）：`0.4289`
- Val loss（最优时）：`0.6895`

**Test（overall）**
- Accuracy：`0.5118`
- Recall：`0.2482`
- Specificity：`0.7755`
- Precision：`0.5250`
- F1：`0.3370`
- Balanced Acc：`0.5118`
- 混淆矩阵：TP=`546` TN=`1706` FP=`494` FN=`1654`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`53`
- 验证最优轮次（best_epoch）：`35`
- Val F1（最优）：`0.4091`
- Val loss（最优时）：`0.6888`

**Test（overall）**
- Accuracy：`0.5336`
- Recall：`0.3173`
- Specificity：`0.7500`
- Precision：`0.5593`
- F1：`0.4049`
- Balanced Acc：`0.5336`
- 混淆矩阵：TP=`698` TN=`1650` FP=`550` FN=`1502`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val F1（最优）：`0.6054`
- Val loss（最优时）：`0.6921`

**Test（overall）**
- Accuracy：`0.5084`
- Recall：`0.7109`
- Specificity：`0.3059`
- Precision：`0.5060`
- F1：`0.5912`
- Balanced Acc：`0.5084`
- 混淆矩阵：TP=`1564` TN=`673` FP=`1527` FN=`636`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val F1（最优）：`0.5558`
- Val loss（最优时）：`0.6882`

**Test（overall）**
- Accuracy：`0.5434`
- Recall：`0.6436`
- Specificity：`0.4432`
- Precision：`0.5362`
- F1：`0.5850`
- Balanced Acc：`0.5434`
- 混淆矩阵：TP=`1416` TN=`975` FP=`1225` FN=`784`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.6311`
- Val loss（最优时）：`0.6931`

**Test（overall）**
- Accuracy：`0.5355`
- Recall：`0.8210`
- Specificity：`0.2500`
- Precision：`0.5226`
- F1：`0.6387`
- Balanced Acc：`0.5355`
- 混淆矩阵：TP=`1642` TN=`500` FP=`1500` FN=`358`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.2222 ± 0.0000`
- Test F1-macro：`0.2222 ± 0.0000`
- Test Acc：`0.5000 ± 0.0000`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`0.2222`
- Val loss（最优时）：`1.0410`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2222`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2200      0      0
  true1   1100      0      0
  true2   1100      0      0
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`0.2222`
- Val loss（最优时）：`1.0412`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2222`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2200      0      0
  true1   1100      0      0
  true2   1100      0      0
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`0.2222`
- Val loss（最优时）：`1.0422`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2222`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2200      0      0
  true1   1100      0      0
  true2   1100      0      0
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`0.2222`
- Val loss（最优时）：`1.0400`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2222`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2200      0      0
  true1   1100      0      0
  true2   1100      0      0
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`0.2222`
- Val loss（最优时）：`1.0418`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2222`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2000      0      0
  true1   1000      0      0
  true2   1000      0      0
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

- 结束：`2026-08-06T01:23:08`
