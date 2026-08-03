# 被试独立五折实验记录（20260801_153617 / shallow_wce2_balacc）

- 开始：`2026-08-01T15:36:17`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`shallow_wce2_balacc`（单脚本；无 registry）
- Task 目标：加权CE w0=2.0, w1=1.0, mode=fixed；早停=Balanced Acc
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balacc\merged_2s\run_20260801_153617`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc（选模）：`0.5863 ± 0.0215`
- Test Spec：`0.3603 ± 0.1337`
- Test Rec：`0.7699 ± 0.1093`
- Test BalAcc：`0.5651 ± 0.0390`
- Test F1（附报）：`0.7558 ± 0.0566`
- Test Acc：`0.6519 ± 0.0553`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优（Balanced Acc）并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.6177`
- Val F1（最优 checkpoint 时，附报）：`0.7092`
- Val loss（最优时）：`0.6569`

**Test（overall）**
- Accuracy：`0.5763`
- Recall：`0.5994`
- Specificity：`0.5191`
- Precision：`0.7548`
- F1：`0.6682`
- Balanced Acc：`0.5593`
- 混淆矩阵：TP=`3026` TN=`1061` FP=`983` FN=`2022`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5934` F1=`0.6582` BalAcc=`0.6003`
- `stieger_only`：Acc=`0.5753` F1=`0.6688` BalAcc=`0.5564`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5510`
- Val F1（最优 checkpoint 时，附报）：`0.7901`
- Val loss（最优时）：`0.7038`

**Test（overall）**
- Accuracy：`0.6172`
- Recall：`0.6820`
- Specificity：`0.4520`
- Precision：`0.7604`
- F1：`0.7191`
- Balanced Acc：`0.5670`
- 混淆矩阵：TP=`2973` TN=`773` FP=`937` FN=`1386`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5477` F1=`0.5431` BalAcc=`0.6291`
- `stieger_only`：Acc=`0.6221` F1=`0.7279` BalAcc=`0.5597`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val 选模分数（Balanced Acc）：`0.5817`
- Val F1（最优 checkpoint 时，附报）：`0.7288`
- Val loss（最优时）：`0.6762`

**Test（overall）**
- Accuracy：`0.7392`
- Recall：`0.8448`
- Specificity：`0.4289`
- Precision：`0.8130`
- F1：`0.8286`
- Balanced Acc：`0.6369`
- 混淆矩阵：TP=`4017` TN=`694` FP=`924` FN=`738`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7205` F1=`0.8256` BalAcc=`0.5822`
- `stieger_only`：Acc=`0.7404` F1=`0.8288` BalAcc=`0.6435`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5871`
- Val F1（最优 checkpoint 时，附报）：`0.7158`
- Val loss（最优时）：`0.6737`

**Test（overall）**
- Accuracy：`0.6474`
- Recall：`0.8460`
- Specificity：`0.2016`
- Precision：`0.7041`
- F1：`0.7685`
- Balanced Acc：`0.5238`
- 混淆矩阵：TP=`4796` TN=`509` FP=`2016` FN=`873`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5870` F1=`0.6898` BalAcc=`0.5368`
- `stieger_only`：Acc=`0.6503` F1=`0.7717` BalAcc=`0.5229`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5940`
- Val F1（最优 checkpoint 时，附报）：`0.7865`
- Val loss（最优时）：`0.6536`

**Test（overall）**
- Accuracy：`0.6792`
- Recall：`0.8772`
- Specificity：`0.1997`
- Precision：`0.7263`
- F1：`0.7947`
- Balanced Acc：`0.5384`
- 混淆矩阵：TP=`5170` TN=`486` FP=`1948` FN=`724`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5329` F1=`0.5517` BalAcc=`0.6089`
- `stieger_only`：Acc=`0.6821` F1=`0.7979` BalAcc=`0.5366`

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

- 结束：`2026-08-01T15:49:19`
