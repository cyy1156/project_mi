# 被试独立五折实验记录（20260806_005142 / conformer）

- 开始：`2026-08-06T00:51:42`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`conformer`（单脚本；无 registry）
- 结构：EEGConformer（num_layers=2, num_heads=10, att_drop=0.5 + shared drop_prob）
- shared hp：`{'data_tag': 'openbmi_balanced_train_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\conformer\openbmi_balanced_train_2s\run_20260806_005142`

---
## 最终结论

### Task（静息/任务）
- Val Balanced Acc：`0.6008 ± 0.0359`
- Test F1：`0.6519 ± 0.0247`
- Test Acc：`0.6016 ± 0.0126`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`71`
- 验证最优轮次（best_epoch）：`53`
- Val F1（最优）：`0.6576`
- Val loss（最优时）：`0.7040`

**Test（overall）**
- Accuracy：`0.6105`
- Recall：`0.6905`
- Specificity：`0.5305`
- Precision：`0.5952`
- F1：`0.6393`
- Balanced Acc：`0.6105`
- 混淆矩阵：TP=`1519` TN=`1167` FP=`1033` FN=`681`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val F1（最优）：`0.6646`
- Val loss（最优时）：`0.7180`

**Test（overall）**
- Accuracy：`0.6116`
- Recall：`0.7745`
- Specificity：`0.4486`
- Precision：`0.5842`
- F1：`0.6660`
- Balanced Acc：`0.6116`
- 混淆矩阵：TP=`1704` TN=`987` FP=`1213` FN=`496`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`64`
- 验证最优轮次（best_epoch）：`46`
- Val F1（最优）：`0.5860`
- Val loss（最优时）：`0.7444`

**Test（overall）**
- Accuracy：`0.5925`
- Recall：`0.6786`
- Specificity：`0.5064`
- Precision：`0.5789`
- F1：`0.6248`
- Balanced Acc：`0.5925`
- 混淆矩阵：TP=`1493` TN=`1114` FP=`1086` FN=`707`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`56`
- 验证最优轮次（best_epoch）：`38`
- Val F1（最优）：`0.6971`
- Val loss（最优时）：`0.6719`

**Test（overall）**
- Accuracy：`0.6123`
- Recall：`0.8764`
- Specificity：`0.3482`
- Precision：`0.5735`
- F1：`0.6933`
- Balanced Acc：`0.6123`
- 混淆矩阵：TP=`1928` TN=`766` FP=`1434` FN=`272`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val F1（最优）：`0.6225`
- Val loss（最优时）：`0.7989`

**Test（overall）**
- Accuracy：`0.5813`
- Recall：`0.7315`
- Specificity：`0.4310`
- Precision：`0.5625`
- F1：`0.6359`
- Balanced Acc：`0.5813`
- 混淆矩阵：TP=`1463` TN=`862` FP=`1138` FN=`537`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4738 ± 0.0320`
- Test F1-macro：`0.4791 ± 0.0154`
- Test Acc：`0.5133 ± 0.0137`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val F1-macro（最优）：`0.4840`
- Val loss（最优时）：`0.9887`

**Test（overall）**
- Accuracy：`0.5255`
- F1-macro：`0.4641`
- Recall-macro：`0.4583`
- Recall idle/left/right：`0.7268` / `0.3436` / `0.3045`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1599    273    328
  true1    616    378    106
  true2    620    145    335
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val F1-macro（最优）：`0.4924`
- Val loss（最优时）：`1.0329`

**Test（overall）**
- Accuracy：`0.5252`
- F1-macro：`0.4923`
- Recall-macro：`0.4965`
- Recall idle/left/right：`0.6114` / `0.3627` / `0.5155`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1345    324    531
  true1    355    399    346
  true2    356    177    567
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`54`
- 验证最优轮次（best_epoch）：`36`
- Val F1-macro（最优）：`0.4588`
- Val loss（最优时）：`1.0902`

**Test（overall）**
- Accuracy：`0.4898`
- F1-macro：`0.4613`
- Recall-macro：`0.4650`
- Recall idle/left/right：`0.5641` / `0.3445` / `0.4864`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1241    386    573
  true1    418    379    303
  true2    430    135    535
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val F1-macro（最优）：`0.5135`
- Val loss（最优时）：`0.9649`

**Test（overall）**
- Accuracy：`0.5198`
- F1-macro：`0.5007`
- Recall-macro：`0.5144`
- Recall idle/left/right：`0.5359` / `0.6036` / `0.4036`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1179    585    436
  true1    277    664    159
  true2    299    357    444
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`40`
- Val F1-macro（最优）：`0.4202`
- Val loss（最优时）：`1.1561`

**Test（overall）**
- Accuracy：`0.5062`
- F1-macro：`0.4774`
- Recall-macro：`0.4772`
- Recall idle/left/right：`0.5935` / `0.4230` / `0.4150`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1187    406    407
  true1    416    423    161
  true2    407    178    415
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

- 结束：`2026-08-06T01:20:03`
