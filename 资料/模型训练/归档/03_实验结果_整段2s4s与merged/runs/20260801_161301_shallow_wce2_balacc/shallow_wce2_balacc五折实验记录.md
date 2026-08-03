# 被试独立五折实验记录（20260801_161301 / shallow_wce2_balacc）

- 开始：`2026-08-01T16:13:01`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`shallow_wce2_balacc`（单脚本；无 registry）
- Task 目标：加权CE w0=2.2, w1=1.0, mode=fixed；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balacc\merged_2s\run_20260801_161301`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc（选模）：`0.5869 ± 0.0226`
- Test Spec：`0.4088 ± 0.1626`
- Test Rec：`0.7244 ± 0.1427`
- Test BalAcc：`0.5666 ± 0.0394`
- Test F1（附报）：`0.7309 ± 0.0760`
- Test Acc：`0.6328 ± 0.0670`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优（Balanced Acc）并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc）：`0.6225`
- Val F1（最优 checkpoint 时，附报）：`0.6699`
- Val loss（最优时）：`0.6621`

**Test（overall）**
- Accuracy：`0.5605`
- Recall：`0.5664`
- Specificity：`0.5460`
- Precision：`0.7550`
- F1：`0.6472`
- Balanced Acc：`0.5562`
- 混淆矩阵：TP=`2859` TN=`1116` FP=`928` FN=`2189`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6162` F1=`0.6833` BalAcc=`0.6172`
- `stieger_only`：Acc=`0.5572` F1=`0.6451` BalAcc=`0.5523`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val 选模分数（Balanced Acc）：`0.5533`
- Val F1（最优 checkpoint 时，附报）：`0.7899`
- Val loss（最优时）：`0.7116`

**Test（overall）**
- Accuracy：`0.5540`
- Recall：`0.5348`
- Specificity：`0.6029`
- Precision：`0.7744`
- F1：`0.6327`
- Balanced Acc：`0.5688`
- 混淆矩阵：TP=`2331` TN=`1031` FP=`679` FN=`2028`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5704` F1=`0.5649` BalAcc=`0.6559`
- `stieger_only`：Acc=`0.5528` F1=`0.6365` BalAcc=`0.5608`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5762`
- Val F1（最优 checkpoint 时，附报）：`0.6826`
- Val loss（最优时）：`0.6762`

**Test（overall）**
- Accuracy：`0.7296`
- Recall：`0.8221`
- Specificity：`0.4580`
- Precision：`0.8168`
- F1：`0.8194`
- Balanced Acc：`0.6400`
- 混淆矩阵：TP=`3909` TN=`741` FP=`877` FN=`846`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7179` F1=`0.8220` BalAcc=`0.5863`
- `stieger_only`：Acc=`0.7304` F1=`0.8192` BalAcc=`0.6467`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.5887`
- Val F1（最优 checkpoint 时，附报）：`0.6723`
- Val loss（最优时）：`0.6857`

**Test（overall）**
- Accuracy：`0.6515`
- Recall：`0.8541`
- Specificity：`0.1964`
- Precision：`0.7047`
- F1：`0.7722`
- Balanced Acc：`0.5253`
- 混淆矩阵：TP=`4842` TN=`496` FP=`2029` FN=`827`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5462` F1=`0.6125` BalAcc=`0.5550`
- `stieger_only`：Acc=`0.6564` F1=`0.7779` BalAcc=`0.5232`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5937`
- Val F1（最优 checkpoint 时，附报）：`0.7595`
- Val loss（最优时）：`0.6602`

**Test（overall）**
- Accuracy：`0.6682`
- Recall：`0.8448`
- Specificity：`0.2408`
- Precision：`0.7293`
- F1：`0.7828`
- Balanced Acc：`0.5428`
- 混淆矩阵：TP=`4979` TN=`586` FP=`1848` FN=`915`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5150` F1=`0.5091` BalAcc=`0.6124`
- `stieger_only`：Acc=`0.6714` F1=`0.7864` BalAcc=`0.5409`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val F1-macro：`0.4513 ± 0.0299`
- Test F1-macro：`0.4202 ± 0.0428`
- Test Acc：`0.4379 ± 0.0412`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val F1-macro（最优）：`0.5034`
- Val loss（最优时）：`1.0144`

**Test（overall）**
- Accuracy：`0.4003`
- F1-macro：`0.3996`
- Recall-macro：`0.4009`
- Recall idle/left/right：`0.4080` / `0.4154` / `0.3792`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    834    628    582
  true1    754   1041    711
  true2    685    893    964
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5328` F1m=`0.5260`
- `stieger_only`：Acc=`0.3925` F1m=`0.3918`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val F1-macro（最优）：`0.4173`
- Val loss（最优时）：`1.0884`

**Test（overall）**
- Accuracy：`0.4449`
- F1-macro：`0.4451`
- Recall-macro：`0.4517`
- Recall idle/left/right：`0.5246` / `0.4544` / `0.3763`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    897    521    292
  true1    792    947    345
  true2    729    690    856
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5779` F1m=`0.5727`
- `stieger_only`：Acc=`0.4355` F1m=`0.4355`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val F1-macro（最优）：`0.4285`
- Val loss（最优时）：`1.0876`

**Test（overall）**
- Accuracy：`0.5093`
- F1-macro：`0.4885`
- Recall-macro：`0.4957`
- Recall idle/left/right：`0.3597` / `0.3805` / `0.7470`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    582    274    762
  true1    330    922   1171
  true2    223    367   1742
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4769` F1m=`0.4326`
- `stieger_only`：Acc=`0.5114` F1m=`0.4912`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val F1-macro（最优）：`0.4478`
- Val loss（最优时）：`1.0776`

**Test（overall）**
- Accuracy：`0.3939`
- F1-macro：`0.3634`
- Recall-macro：`0.3855`
- Recall idle/left/right：`0.1446` / `0.4129` / `0.5991`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    365    992   1168
  true1    305   1183   1377
  true2    277    847   1680
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4076` F1m=`0.3667`
- `stieger_only`：Acc=`0.3933` F1m=`0.3622`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val F1-macro（最优）：`0.4596`
- Val loss（最优时）：`1.0422`

**Test（overall）**
- Accuracy：`0.4408`
- F1-macro：`0.4045`
- Recall-macro：`0.4241`
- Recall idle/left/right：`0.1528` / `0.5965` / `0.5231`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    372   1202    860
  true1    275   1756    913
  true2    259   1148   1543
```

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5928` F1m=`0.6006`
- `stieger_only`：Acc=`0.4377` F1m=`0.3987`

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

- 结束：`2026-08-01T16:27:21`
