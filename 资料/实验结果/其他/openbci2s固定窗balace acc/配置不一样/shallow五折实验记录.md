# 被试独立五折实验记录（20260806_000811 / shallow）

- 开始：`2026-08-06T00:08:11`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`shallow`（单脚本；无 registry）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\shallow\openbmi_balanced_train_2s\run_20260806_000811`

---
## 最终结论

### Task（静息/任务）
- Val Balanced Acc：`0.5996 ± 0.0231`
- Test F1：`0.6083 ± 0.0306`
- Test Acc：`0.6039 ± 0.0127`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val F1（最优）：`0.6027`
- Val loss（最优时）：`0.6950`

**Test（overall）**
- Accuracy：`0.6127`
- Recall：`0.5141`
- Specificity：`0.7114`
- Precision：`0.6404`
- F1：`0.5703`
- Balanced Acc：`0.6127`
- 混淆矩阵：TP=`1131` TN=`1565` FP=`635` FN=`1069`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val F1（最优）：`0.6390`
- Val loss（最优时）：`0.6810`

**Test（overall）**
- Accuracy：`0.6023`
- Recall：`0.6409`
- Specificity：`0.5636`
- Precision：`0.5949`
- F1：`0.6171`
- Balanced Acc：`0.6023`
- 混淆矩阵：TP=`1410` TN=`1240` FP=`960` FN=`790`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val F1（最优）：`0.5401`
- Val loss（最优时）：`0.7429`

**Test（overall）**
- Accuracy：`0.5925`
- Recall：`0.5895`
- Specificity：`0.5955`
- Precision：`0.5930`
- F1：`0.5913`
- Balanced Acc：`0.5925`
- 混淆矩阵：TP=`1297` TN=`1310` FP=`890` FN=`903`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`56`
- 验证最优轮次（best_epoch）：`38`
- Val F1（最优）：`0.6238`
- Val loss（最优时）：`0.6580`

**Test（overall）**
- Accuracy：`0.6232`
- Recall：`0.7359`
- Specificity：`0.5105`
- Precision：`0.6005`
- F1：`0.6614`
- Balanced Acc：`0.6232`
- 混淆矩阵：TP=`1619` TN=`1123` FP=`1077` FN=`581`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`59`
- 验证最优轮次（best_epoch）：`41`
- Val F1（最优）：`0.6070`
- Val loss（最优时）：`0.7675`

**Test（overall）**
- Accuracy：`0.5890`
- Recall：`0.6200`
- Specificity：`0.5580`
- Precision：`0.5838`
- F1：`0.6014`
- Balanced Acc：`0.5890`
- 混淆矩阵：TP=`1240` TN=`1116` FP=`884` FN=`760`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4578 ± 0.0264`
- Test F1-macro：`0.4645 ± 0.0184`
- Test Acc：`0.5216 ± 0.0133`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`33`
- Val F1-macro（最优）：`0.4672`
- Val loss（最优时）：`0.9910`

**Test（overall）**
- Accuracy：`0.5359`
- F1-macro：`0.4587`
- Recall-macro：`0.4558`
- Recall idle/left/right：`0.7764` / `0.2918` / `0.2991`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1708    225    267
  true1    638    321    141
  true2    611    160    329
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`46`
- 验证最优轮次（best_epoch）：`28`
- Val F1-macro（最优）：`0.4737`
- Val loss（最优时）：`1.0109`

**Test（overall）**
- Accuracy：`0.5275`
- F1-macro：`0.4745`
- Recall-macro：`0.4721`
- Recall idle/left/right：`0.6936` / `0.3227` / `0.4000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1526    307    367
  true1    500    355    245
  true2    487    173    440
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val F1-macro（最优）：`0.4322`
- Val loss（最优时）：`1.0471`

**Test（overall）**
- Accuracy：`0.5050`
- F1-macro：`0.4424`
- Recall-macro：`0.4395`
- Recall idle/left/right：`0.7014` / `0.2864` / `0.3309`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1543    294    363
  true1    601    315    184
  true2    594    142    364
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`69`
- 验证最优轮次（best_epoch）：`51`
- Val F1-macro（最优）：`0.4932`
- Val loss（最优时）：`0.9306`

**Test（overall）**
- Accuracy：`0.5334`
- F1-macro：`0.4947`
- Recall-macro：`0.4938`
- Recall idle/left/right：`0.6523` / `0.4627` / `0.3664`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1435    419    346
  true1    423    509    168
  true2    459    238    403
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val F1-macro（最优）：`0.4226`
- Val loss（最优时）：`1.1152`

**Test（overall）**
- Accuracy：`0.5062`
- F1-macro：`0.4520`
- Recall-macro：`0.4495`
- Recall idle/left/right：`0.6765` / `0.3880` / `0.2840`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1353    364    283
  true1    485    388    127
  true2    540    176    284
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

- 结束：`2026-08-06T00:16:09`
