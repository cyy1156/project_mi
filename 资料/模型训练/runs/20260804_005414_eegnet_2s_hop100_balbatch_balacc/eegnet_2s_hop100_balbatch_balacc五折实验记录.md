# 被试独立五折实验记录（20260804_005414 / eegnet_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T00:54:14`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`eegnet`（原结构）
- 结构：EEGNet F1=8, D=2, F2=16（默认池化）
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\eegnet_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_005414`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.6190 ± 0.0567`
- Test BalAcc：`0.5920 ± 0.0239`
- Test Spec：`0.6784 ± 0.1683`
- Test Rec：`0.5056 ± 0.1666`
- Test F1：`0.5923 ± 0.1277`
- Test Acc：`0.5578 ± 0.0683`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.7132`
- Val F1（最优 checkpoint 时，附报）：`0.7374`
- Val loss（最优时）：`0.6014`

**Test（overall）**
- Accuracy：`0.5705`
- Recall：`0.5732`
- Specificity：`0.5646`
- Precision：`0.7372`
- F1：`0.6449`
- Balanced Acc：`0.5689`
- 混淆矩阵：TP=`3214` TN=`1486` FP=`1146` FN=`2393`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5690`
- Val F1（最优 checkpoint 时，附报）：`0.6661`
- Val loss（最优时）：`0.6706`

**Test（overall）**
- Accuracy：`0.5216`
- Recall：`0.4022`
- Specificity：`0.7796`
- Precision：`0.7978`
- F1：`0.5348`
- Balanced Acc：`0.5909`
- 混淆矩阵：TP=`2272` TN=`2038` FP=`576` FN=`3377`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val 选模分数（Balanced Acc）：`0.5573`
- Val F1（最优 checkpoint 时，附报）：`0.5212`
- Val loss（最优时）：`0.7475`

**Test（overall）**
- Accuracy：`0.6497`
- Recall：`0.6761`
- Specificity：`0.5938`
- Precision：`0.7791`
- F1：`0.7239`
- Balanced Acc：`0.6349`
- 混淆矩阵：TP=`3720` TN=`1542` FP=`1055` FN=`1782`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val 选模分数（Balanced Acc）：`0.6085`
- Val F1（最优 checkpoint 时，附报）：`0.6073`
- Val loss（最优时）：`0.7010`

**Test（overall）**
- Accuracy：`0.5980`
- Recall：`0.6449`
- Specificity：`0.4960`
- Precision：`0.7356`
- F1：`0.6873`
- Balanced Acc：`0.5705`
- 混淆矩阵：TP=`3372` TN=`1193` FP=`1212` FN=`1857`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`53`
- 验证最优轮次（best_epoch）：`35`
- Val 选模分数（Balanced Acc）：`0.6470`
- Val F1（最优 checkpoint 时，附报）：`0.7675`
- Val loss（最优时）：`0.6100`

**Test（overall）**
- Accuracy：`0.4491`
- Recall：`0.2315`
- Specificity：`0.9578`
- Precision：`0.9276`
- F1：`0.3706`
- Balanced Acc：`0.5947`
- 混淆矩阵：TP=`564` TN=`998` FP=`44` FN=`1872`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4638 ± 0.0447`
- Val F1-macro：`0.4550 ± 0.0450`
- Test BalAcc：`0.4640 ± 0.0570`
- Test F1-macro：`0.4464 ± 0.0533`
- Test Acc：`0.4632 ± 0.0543`
- Test Precision-macro：`0.4933 ± 0.0696`
- Test Recall-macro：`0.4640 ± 0.0570`
- Test Recall idle/left/right：`0.5165±0.2397` / `0.4145±0.1700` / `0.4610±0.1581`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5528`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5433`
- Val loss（最优时）：`0.9869`

**Test（overall）**
- Accuracy：`0.4301`
- Balanced Acc：`0.4281`
- F1-macro：`0.4264`
- Precision-macro：`0.4322`
- Recall-macro：`0.4281`
- Recall idle/left/right：`0.3750` / `0.3788` / `0.5305`
- Precision idle/left/right：`0.4062` / `0.4668` / `0.4235`
- F1 idle/left/right：`0.3900` / `0.4182` / `0.4710`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    987    481   1164
  true1    811   1042    898
  true2    632    709   1515
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`71`
- 验证最优轮次（best_epoch）：`53`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4406`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4231`
- Val loss（最优时）：`1.0596`

**Test（overall）**
- Accuracy：`0.4704`
- Balanced Acc：`0.4773`
- F1-macro：`0.4581`
- Precision-macro：`0.5292`
- Recall-macro：`0.4773`
- Recall idle/left/right：`0.7533` / `0.3781` / `0.3006`
- Precision idle/left/right：`0.3988` / `0.5204` / `0.6682`
- F1 idle/left/right：`0.5215` / `0.4380` / `0.4147`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1969    396    249
  true1   1592   1072    171
  true2   1376    592    846
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4356`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4298`
- Val loss（最优时）：`1.0623`

**Test（overall）**
- Accuracy：`0.4066`
- Balanced Acc：`0.4045`
- F1-macro：`0.3759`
- Precision-macro：`0.4538`
- Recall-macro：`0.4045`
- Recall idle/left/right：`0.2241` / `0.7436` / `0.2457`
- Precision idle/left/right：`0.5740` / `0.3730` / `0.4145`
- F1 idle/left/right：`0.3223` / `0.4968` / `0.3085`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    582   1583    432
  true1    170   2030    530
  true2    262   1829    681
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`56`
- 验证最优轮次（best_epoch）：`38`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4406`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4296`
- Val loss（最优时）：`1.0678`

**Test（overall）**
- Accuracy：`0.4452`
- Balanced Acc：`0.4423`
- F1-macro：`0.4329`
- Precision-macro：`0.4377`
- Recall-macro：`0.4423`
- Recall idle/left/right：`0.3846` / `0.3043` / `0.6379`
- Precision idle/left/right：`0.4228` / `0.4157` / `0.4748`
- F1 idle/left/right：`0.4028` / `0.3514` / `0.5444`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    925    713    767
  true1    697    786   1100
  true2    566    392   1688
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`40`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4492`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4493`
- Val loss（最优时）：`1.0624`

**Test（overall）**
- Accuracy：`0.5635`
- Balanced Acc：`0.5679`
- F1-macro：`0.5386`
- Precision-macro：`0.6138`
- Recall-macro：`0.5679`
- Recall idle/left/right：`0.8455` / `0.2677` / `0.5903`
- Precision idle/left/right：`0.4711` / `0.7146` / `0.6558`
- F1 idle/left/right：`0.6051` / `0.3895` / `0.6213`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    881     53    108
  true1    513    298    302
  true2    476     66    781
```

### 共用超参
```json
{
  "data_tag": "bci2a_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 18,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-offline-native",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true
}
```

- 结束：`2026-08-04T02:22:40`
