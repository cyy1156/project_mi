# 被试独立五折实验记录（20260806_001615 / deep）

- 开始：`2026-08-06T00:16:15`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`deep`（单脚本；无 registry）
- 结构：Deep4Net（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\deep\openbmi_balanced_train_2s\run_20260806_001615`

---
## 最终结论

### Task（静息/任务）
- Val Balanced Acc：`0.5833 ± 0.0314`
- Test F1：`0.6117 ± 0.0449`
- Test Acc：`0.5824 ± 0.0264`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`74`
- 验证最优轮次（best_epoch）：`56`
- Val F1（最优）：`0.6282`
- Val loss（最优时）：`0.6552`

**Test（overall）**
- Accuracy：`0.6061`
- Recall：`0.5605`
- Specificity：`0.6518`
- Precision：`0.6168`
- F1：`0.5873`
- Balanced Acc：`0.6061`
- 混淆矩阵：TP=`1233` TN=`1434` FP=`766` FN=`967`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`126`
- 验证最优轮次（best_epoch）：`108`
- Val F1（最优）：`0.6594`
- Val loss（最优时）：`0.6890`

**Test（overall）**
- Accuracy：`0.6198`
- Recall：`0.7118`
- Specificity：`0.5277`
- Precision：`0.6012`
- F1：`0.6518`
- Balanced Acc：`0.6198`
- 混淆矩阵：TP=`1566` TN=`1161` FP=`1039` FN=`634`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`87`
- 验证最优轮次（best_epoch）：`69`
- Val F1（最优）：`0.4540`
- Val loss（最优时）：`0.7249`

**Test（overall）**
- Accuracy：`0.5727`
- Recall：`0.5073`
- Specificity：`0.6382`
- Precision：`0.5837`
- F1：`0.5428`
- Balanced Acc：`0.5727`
- 混淆矩阵：TP=`1116` TN=`1404` FP=`796` FN=`1084`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val F1（最优）：`0.6671`
- Val loss（最优时）：`0.6791`

**Test（overall）**
- Accuracy：`0.5495`
- Recall：`0.9041`
- Specificity：`0.1950`
- Precision：`0.5290`
- F1：`0.6674`
- Balanced Acc：`0.5495`
- 混淆矩阵：TP=`1989` TN=`429` FP=`1771` FN=`211`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val F1（最优）：`0.6145`
- Val loss（最优时）：`0.6990`

**Test（overall）**
- Accuracy：`0.5640`
- Recall：`0.6790`
- Specificity：`0.4490`
- Precision：`0.5520`
- F1：`0.6090`
- Balanced Acc：`0.5640`
- 混淆矩阵：TP=`1358` TN=`898` FP=`1102` FN=`642`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4290 ± 0.0668`
- Test F1-macro：`0.4305 ± 0.0555`
- Test Acc：`0.4707 ± 0.0261`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val F1-macro（最优）：`0.3139`
- Val loss（最优时）：`1.1139`

**Test（overall）**
- Accuracy：`0.4559`
- F1-macro：`0.3267`
- Recall-macro：`0.3833`
- Recall idle/left/right：`0.6736` / `0.0009` / `0.4755`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1482      0    718
  true1    604      1    495
  true2    577      0    523
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`117`
- 验证最优轮次（best_epoch）：`99`
- Val F1-macro（最优）：`0.4729`
- Val loss（最优时）：`1.0114`

**Test（overall）**
- Accuracy：`0.5189`
- F1-macro：`0.4866`
- Recall-macro：`0.5008`
- Recall idle/left/right：`0.5732` / `0.3018` / `0.6273`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1261    253    686
  true1    375    332    393
  true2    283    127    690
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`72`
- 验证最优轮次（best_epoch）：`54`
- Val F1-macro（最优）：`0.4592`
- Val loss（最优时）：`1.0231`

**Test（overall）**
- Accuracy：`0.4459`
- F1-macro：`0.4377`
- Recall-macro：`0.4565`
- Recall idle/left/right：`0.4141` / `0.4118` / `0.5436`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    911    522    767
  true1    250    453    397
  true2    293    209    598
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`103`
- 验证最优轮次（best_epoch）：`85`
- Val F1-macro（最优）：`0.5014`
- Val loss（最优时）：`0.9610`

**Test（overall）**
- Accuracy：`0.4768`
- F1-macro：`0.4684`
- Recall-macro：`0.5015`
- Recall idle/left/right：`0.4027` / `0.7118` / `0.3900`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    886    900    414
  true1    184    783    133
  true2    218    453    429
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val F1-macro（最优）：`0.3975`
- Val loss（最优时）：`1.0638`

**Test（overall）**
- Accuracy：`0.4557`
- F1-macro：`0.4332`
- Recall-macro：`0.4362`
- Recall idle/left/right：`0.5145` / `0.4050` / `0.3890`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1029    464    507
  true1    401    405    194
  true2    410    201    389
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

- 结束：`2026-08-06T00:34:04`
