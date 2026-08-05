# 被试独立五折实验记录（20260804_083414 / gcbnet_raw_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T08:34:14`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`gcbnet_raw`（原结构）
- 结构：TemporalEncoder(D=64) + GCBNet(k=2)；2s/hop100 原始时域 (B,8,500)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\gcbnet_raw_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_083414`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5915 ± 0.0558`
- Test BalAcc：`0.5271 ± 0.0281`
- Test Spec：`0.7855 ± 0.1260`
- Test Rec：`0.2687 ± 0.1630`
- Test F1：`0.3614 ± 0.1979`
- Test Acc：`0.4295 ± 0.0795`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`54`
- 验证最优轮次（best_epoch）：`36`
- Val 选模分数（Balanced Acc）：`0.6724`
- Val F1（最优 checkpoint 时，附报）：`0.7781`
- Val loss（最优时）：`0.6845`

**Test（overall）**
- Accuracy：`0.4964`
- Recall：`0.3647`
- Specificity：`0.7770`
- Precision：`0.7770`
- F1：`0.4964`
- Balanced Acc：`0.5708`
- 混淆矩阵：TP=`2045` TN=`2045` FP=`587` FN=`3562`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.6426`
- Val F1（最优 checkpoint 时，附报）：`0.6135`
- Val loss（最优时）：`0.6991`

**Test（overall）**
- Accuracy：`0.3997`
- Recall：`0.2105`
- Specificity：`0.8087`
- Precision：`0.7040`
- F1：`0.3241`
- Balanced Acc：`0.5096`
- 混淆矩阵：TP=`1189` TN=`2114` FP=`500` FN=`4460`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`47`
- 验证最优轮次（best_epoch）：`29`
- Val 选模分数（Balanced Acc）：`0.5580`
- Val F1（最优 checkpoint 时，附报）：`0.6521`
- Val loss（最优时）：`0.7041`

**Test（overall）**
- Accuracy：`0.4203`
- Recall：`0.2695`
- Specificity：`0.7397`
- Precision：`0.6869`
- F1：`0.3872`
- Balanced Acc：`0.5046`
- 混淆矩阵：TP=`1483` TN=`1921` FP=`676` FN=`4019`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`63`
- 验证最优轮次（best_epoch）：`45`
- Val 选模分数（Balanced Acc）：`0.5580`
- Val F1（最优 checkpoint 时，附报）：`0.5309`
- Val loss（最优时）：`0.9397`

**Test（overall）**
- Accuracy：`0.5289`
- Recall：`0.4936`
- Specificity：`0.6058`
- Precision：`0.7314`
- F1：`0.5894`
- Balanced Acc：`0.5497`
- 混淆矩阵：TP=`2581` TN=`1457` FP=`948` FN=`2648`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5267`
- Val F1（最优 checkpoint 时，附报）：`0.4975`
- Val loss（最优时）：`0.7354`

**Test（overall）**
- Accuracy：`0.3019`
- Recall：`0.0049`
- Specificity：`0.9962`
- Precision：`0.7500`
- F1：`0.0098`
- Balanced Acc：`0.5005`
- 混淆矩阵：TP=`12` TN=`1038` FP=`4` FN=`2424`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4271 ± 0.0602`
- Val F1-macro：`0.4213 ± 0.0658`
- Test BalAcc：`0.3722 ± 0.0140`
- Test F1-macro：`0.3270 ± 0.0521`
- Test Acc：`0.3625 ± 0.0182`
- Test Precision-macro：`0.4267 ± 0.0727`
- Test Recall-macro：`0.3722 ± 0.0140`
- Test Recall idle/left/right：`0.6816±0.1735` / `0.1843±0.0883` / `0.2507±0.1092`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5433`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5444`
- Val loss（最优时）：`0.9895`

**Test（overall）**
- Accuracy：`0.3681`
- Balanced Acc：`0.3746`
- F1-macro：`0.3390`
- Precision-macro：`0.4039`
- Recall-macro：`0.3746`
- Recall idle/left/right：`0.7059` / `0.1745` / `0.2433`
- Precision idle/left/right：`0.3328` / `0.4170` / `0.4618`
- F1 idle/left/right：`0.4523` / `0.2460` / `0.3187`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1858    333    441
  true1   1902    480    369
  true2   1823    338    695
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4259`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4268`
- Val loss（最优时）：`1.0863`

**Test（overall）**
- Accuracy：`0.3711`
- Balanced Acc：`0.3780`
- F1-macro：`0.3550`
- Precision-macro：`0.4148`
- Recall-macro：`0.3780`
- Recall idle/left/right：`0.6500` / `0.2332` / `0.2509`
- Precision idle/left/right：`0.3232` / `0.3999` / `0.5214`
- F1 idle/left/right：`0.4318` / `0.2946` / `0.3388`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1699    547    368
  true1   1894    661    280
  true2   1663    445    706
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3808`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3542`
- Val loss（最优时）：`1.1133`

**Test（overall）**
- Accuracy：`0.3431`
- Balanced Acc：`0.3478`
- F1-macro：`0.3270`
- Precision-macro：`0.3609`
- Recall-macro：`0.3478`
- Recall idle/left/right：`0.5903` / `0.2359` / `0.2172`
- Precision idle/left/right：`0.3184` / `0.3558` / `0.4084`
- F1 idle/left/right：`0.4137` / `0.2837` / `0.2836`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1533    571    493
  true1   1707    644    379
  true2   1575    595    602
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4004`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3966`
- Val loss（最优时）：`1.0863`

**Test（overall）**
- Accuracy：`0.3892`
- Balanced Acc：`0.3906`
- F1-macro：`0.3841`
- Precision-macro：`0.3866`
- Recall-macro：`0.3906`
- Recall idle/left/right：`0.4703` / `0.2609` / `0.4407`
- Precision idle/left/right：`0.3934` / `0.3679` / `0.3984`
- F1 idle/left/right：`0.4284` / `0.3053` / `0.4184`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1131    598    676
  true1    824    674   1085
  true2    920    560   1166
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3850`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3844`
- Val loss（最优时）：`1.1642`

**Test（overall）**
- Accuracy：`0.3410`
- Balanced Acc：`0.3699`
- F1-macro：`0.2301`
- Precision-macro：`0.5674`
- Recall-macro：`0.3699`
- Recall idle/left/right：`0.9914` / `0.0171` / `0.1013`
- Precision idle/left/right：`0.3190` / `0.7600` / `0.6233`
- F1 idle/left/right：`0.4827` / `0.0334` / `0.1743`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1033      0      9
  true1   1022     19     72
  true2   1183      6    134
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

- 结束：`2026-08-04T09:05:40`
