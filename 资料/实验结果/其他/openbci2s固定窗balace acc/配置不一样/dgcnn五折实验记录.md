# 被试独立五折实验记录（20260806_014041 / dgcnn）

- 开始：`2026-08-06T01:40:41`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`dgcnn`（单脚本；无 registry）
- 输入：bandpower 立方体 `(21600, 8, 2)`（非时域 500）
- 结构：DGCNN(k=2, layers=[128], dropout=shared drop_prob)
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\dgcnn\openbmi_balanced_train_2s\run_20260806_014041`

---
## 最终结论

### Task（静息/任务）
- Val Balanced Acc：`0.5396 ± 0.0139`
- Test F1：`0.5403 ± 0.0404`
- Test Acc：`0.5441 ± 0.0087`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val F1（最优）：`0.5657`
- Val loss（最优时）：`0.6847`

**Test（overall）**
- Accuracy：`0.5418`
- Recall：`0.5923`
- Specificity：`0.4914`
- Precision：`0.5380`
- F1：`0.5638`
- Balanced Acc：`0.5418`
- 混淆矩阵：TP=`1303` TN=`1081` FP=`1119` FN=`897`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val F1（最优）：`0.4937`
- Val loss（最优时）：`0.6880`

**Test（overall）**
- Accuracy：`0.5593`
- Recall：`0.5655`
- Specificity：`0.5532`
- Precision：`0.5586`
- F1：`0.5620`
- Balanced Acc：`0.5593`
- 混淆矩阵：TP=`1244` TN=`1217` FP=`983` FN=`956`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val F1（最优）：`0.4985`
- Val loss（最优时）：`0.6936`

**Test（overall）**
- Accuracy：`0.5359`
- Recall：`0.3991`
- Specificity：`0.6727`
- Precision：`0.5494`
- F1：`0.4623`
- Balanced Acc：`0.5359`
- 混淆矩阵：TP=`878` TN=`1480` FP=`720` FN=`1322`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.5413`
- Val loss（最优时）：`0.6884`

**Test（overall）**
- Accuracy：`0.5475`
- Recall：`0.6068`
- Specificity：`0.4882`
- Precision：`0.5425`
- F1：`0.5728`
- Balanced Acc：`0.5475`
- 混淆矩阵：TP=`1335` TN=`1074` FP=`1126` FN=`865`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val F1（最优）：`0.5368`
- Val loss（最优时）：`0.6989`

**Test（overall）**
- Accuracy：`0.5360`
- Recall：`0.5460`
- Specificity：`0.5260`
- Precision：`0.5353`
- F1：`0.5406`
- Balanced Acc：`0.5360`
- 混淆矩阵：TP=`1092` TN=`1052` FP=`948` FN=`908`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.2550 ± 0.0052`
- Test F1-macro：`0.2598 ± 0.0259`
- Test Acc：`0.4995 ± 0.0041`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val F1-macro（最优）：`0.2600`
- Val loss（最优时）：`1.0317`

**Test（overall）**
- Accuracy：`0.4943`
- F1-macro：`0.2338`
- Recall-macro：`0.3330`
- Recall idle/left/right：`0.9782` / `0.0118` / `0.0091`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2152     19     29
  true1   1076     13     11
  true2   1083      7     10
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`36`
- 验证最优轮次（best_epoch）：`18`
- Val F1-macro（最优）：`0.2626`
- Val loss（最优时）：`1.0283`

**Test（overall）**
- Accuracy：`0.5020`
- F1-macro：`0.2365`
- Recall-macro：`0.3383`
- Recall idle/left/right：`0.9932` / `0.0155` / `0.0064`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2185      8      7
  true1   1078     17      5
  true2   1090      3      7
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val F1-macro（最优）：`0.2505`
- Val loss（最优时）：`1.0419`

**Test（overall）**
- Accuracy：`0.4968`
- F1-macro：`0.2767`
- Recall-macro：`0.3476`
- Recall idle/left/right：`0.9445` / `0.0664` / `0.0318`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2078     92     30
  true1   1008     73     19
  true2   1004     61     35
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val F1-macro（最优）：`0.2515`
- Val loss（最优时）：`1.0308`

**Test（overall）**
- Accuracy：`0.4982`
- F1-macro：`0.2502`
- Recall-macro：`0.3398`
- Recall idle/left/right：`0.9732` / `0.0245` / `0.0218`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2141     24     35
  true1   1060     27     13
  true2   1049     27     24
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val F1-macro（最优）：`0.2504`
- Val loss（最优时）：`1.0567`

**Test（overall）**
- Accuracy：`0.5060`
- F1-macro：`0.3017`
- Recall-macro：`0.3607`
- Recall idle/left/right：`0.9420` / `0.0760` / `0.0640`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1884     58     58
  true1    878     76     46
  true2    884     52     64
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

- 结束：`2026-08-06T01:46:03`
