# 被试独立五折实验记录（20260804_074751 / gcbnet_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T07:47:51`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`gcbnet`（原结构）
- 结构：GCBNet(k=2, layers=[128]) + 1s bandpower
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\gcbnet_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_074751`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5502 ± 0.0624`
- Test BalAcc：`0.5352 ± 0.0270`
- Test Spec：`0.4915 ± 0.1619`
- Test Rec：`0.5789 ± 0.1598`
- Test F1：`0.6255 ± 0.0993`
- Test Acc：`0.5496 ± 0.0646`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.6705`
- Val F1（最优 checkpoint 时，附报）：`0.8190`
- Val loss（最优时）：`0.5851`

**Test（overall）**
- Accuracy：`0.5913`
- Recall：`0.7043`
- Specificity：`0.3507`
- Precision：`0.6979`
- F1：`0.7011`
- Balanced Acc：`0.5275`
- 混淆矩阵：TP=`3949` TN=`923` FP=`1709` FN=`1658`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5299`
- Val F1（最优 checkpoint 时，附报）：`0.7477`
- Val loss（最优时）：`0.6687`

**Test（overall）**
- Accuracy：`0.4826`
- Recall：`0.4178`
- Specificity：`0.6228`
- Precision：`0.7053`
- F1：`0.5247`
- Balanced Acc：`0.5203`
- 混淆矩阵：TP=`2360` TN=`1628` FP=`986` FN=`3289`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5408`
- Val F1（最优 checkpoint 时，附报）：`0.7293`
- Val loss（最优时）：`0.6553`

**Test（overall）**
- Accuracy：`0.5477`
- Recall：`0.6365`
- Specificity：`0.3596`
- Precision：`0.6780`
- F1：`0.6566`
- Balanced Acc：`0.4981`
- 混淆矩阵：TP=`3502` TN=`934` FP=`1663` FN=`2000`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.4908`
- Val F1（最优 checkpoint 时，附报）：`0.7306`
- Val loss（最优时）：`0.6784`

**Test（overall）**
- Accuracy：`0.6476`
- Recall：`0.7709`
- Specificity：`0.3796`
- Precision：`0.7299`
- F1：`0.7498`
- Balanced Acc：`0.5753`
- 混淆矩阵：TP=`4031` TN=`913` FP=`1492` FN=`1198`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc）：`0.5193`
- Val F1（最优 checkpoint 时，附报）：`0.6943`
- Val loss（最优时）：`0.6911`

**Test（overall）**
- Accuracy：`0.4787`
- Recall：`0.3649`
- Specificity：`0.7447`
- Precision：`0.7697`
- F1：`0.4951`
- Balanced Acc：`0.5548`
- 混淆矩阵：TP=`889` TN=`776` FP=`266` FN=`1547`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.3799 ± 0.0604`
- Val F1-macro：`0.3430 ± 0.0787`
- Test BalAcc：`0.3605 ± 0.0201`
- Test F1-macro：`0.3176 ± 0.0297`
- Test Acc：`0.3591 ± 0.0131`
- Test Precision-macro：`0.4010 ± 0.0506`
- Test Recall-macro：`0.3605 ± 0.0201`
- Test Recall idle/left/right：`0.3332±0.3174` / `0.2941±0.1791` / `0.4543±0.1840`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4933`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4974`
- Val loss（最优时）：`1.0785`

**Test（overall）**
- Accuracy：`0.3407`
- Balanced Acc：`0.3391`
- F1-macro：`0.3263`
- Precision-macro：`0.3446`
- Recall-macro：`0.3391`
- Recall idle/left/right：`0.1938` / `0.5253` / `0.2983`
- Precision idle/left/right：`0.3682` / `0.3428` / `0.3228`
- F1 idle/left/right：`0.2539` / `0.4149` / `0.3101`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    510   1247    875
  true1    394   1445    912
  true2    481   1523    852
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3480`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2882`
- Val loss（最优时）：`1.0986`

**Test（overall）**
- Accuracy：`0.3771`
- Balanced Acc：`0.3764`
- F1-macro：`0.3609`
- Precision-macro：`0.3850`
- Recall-macro：`0.3764`
- Recall idle/left/right：`0.3313` / `0.2127` / `0.5853`
- Precision idle/left/right：`0.3219` / `0.4418` / `0.3914`
- F1 idle/left/right：`0.3265` / `0.2871` / `0.4691`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    866    357   1391
  true1   1062    603   1170
  true2    762    405   1647
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3885`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3312`
- Val loss（最优时）：`1.0864`

**Test（overall）**
- Accuracy：`0.3572`
- Balanced Acc：`0.3503`
- F1-macro：`0.2910`
- Precision-macro：`0.4157`
- Recall-macro：`0.3503`
- Recall idle/left/right：`0.0481` / `0.2788` / `0.7240`
- Precision idle/left/right：`0.5435` / `0.3520` / `0.3517`
- F1 idle/left/right：`0.0884` / `0.3111` / `0.4734`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    125    694   1778
  true1     47    761   1922
  true2     58    707   2007
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3236`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2925`
- Val loss（最优时）：`1.0996`

**Test（overall）**
- Accuracy：`0.3504`
- Balanced Acc：`0.3451`
- F1-macro：`0.3315`
- Precision-macro：`0.3684`
- Recall-macro：`0.3451`
- Recall idle/left/right：`0.1514` / `0.4402` / `0.4437`
- Precision idle/left/right：`0.4174` / `0.3142` / `0.3735`
- F1 idle/left/right：`0.2222` / `0.3667` / `0.4056`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    364   1261    780
  true1    257   1137   1189
  true2    251   1221   1174
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3463`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3056`
- Val loss（最优时）：`1.0998`

**Test（overall）**
- Accuracy：`0.3700`
- Balanced Acc：`0.3916`
- F1-macro：`0.2782`
- Precision-macro：`0.4910`
- Recall-macro：`0.3916`
- Recall idle/left/right：`0.9415` / `0.0135` / `0.2200`
- Precision idle/left/right：`0.3371` / `0.6000` / `0.5359`
- F1 idle/left/right：`0.4965` / `0.0264` / `0.3119`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    981      0     61
  true1    907     15    191
  true2   1022     10    291
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

- 结束：`2026-08-04T07:59:19`
