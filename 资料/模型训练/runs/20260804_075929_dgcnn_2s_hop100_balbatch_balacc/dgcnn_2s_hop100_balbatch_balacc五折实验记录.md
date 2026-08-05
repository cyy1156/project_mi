# 被试独立五折实验记录（20260804_075929 / dgcnn_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T07:59:29`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`dgcnn`（原结构）
- 结构：DGCNN(k=2, layers=[128]) + 1s bandpower
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\dgcnn_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_075929`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5561 ± 0.0765`
- Test BalAcc：`0.5362 ± 0.0203`
- Test Spec：`0.5420 ± 0.2705`
- Test Rec：`0.5305 ± 0.2360`
- Test F1：`0.5768 ± 0.1544`
- Test Acc：`0.5315 ± 0.0792`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.7042`
- Val F1（最优 checkpoint 时，附报）：`0.7621`
- Val loss（最优时）：`0.6637`

**Test（overall）**
- Accuracy：`0.5320`
- Recall：`0.4963`
- Specificity：`0.6079`
- Precision：`0.7295`
- F1：`0.5907`
- Balanced Acc：`0.5521`
- 混淆矩阵：TP=`2783` TN=`1600` FP=`1032` FN=`2824`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5214`
- Val F1（最优 checkpoint 时，附报）：`0.7342`
- Val loss（最优时）：`0.6746`

**Test（overall）**
- Accuracy：`0.4624`
- Recall：`0.3409`
- Specificity：`0.7249`
- Precision：`0.7282`
- F1：`0.4644`
- Balanced Acc：`0.5329`
- 混淆矩阵：TP=`1926` TN=`1895` FP=`719` FN=`3723`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5504`
- Val F1（最优 checkpoint 时，附报）：`0.7712`
- Val loss（最优时）：`0.6553`

**Test（overall）**
- Accuracy：`0.6065`
- Recall：`0.7824`
- Specificity：`0.2337`
- Precision：`0.6839`
- F1：`0.7298`
- Balanced Acc：`0.5081`
- 混淆矩阵：TP=`4305` TN=`607` FP=`1990` FN=`1197`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.4902`
- Val F1（最优 checkpoint 时，附报）：`0.7673`
- Val loss（最优时）：`0.6746`

**Test（overall）**
- Accuracy：`0.6307`
- Recall：`0.8143`
- Specificity：`0.2316`
- Precision：`0.6973`
- F1：`0.7513`
- Balanced Acc：`0.5230`
- 混淆矩阵：TP=`4258` TN=`557` FP=`1848` FN=`971`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.5145`
- Val F1（最优 checkpoint 时，附报）：`0.6796`
- Val loss（最优时）：`0.6858`

**Test（overall）**
- Accuracy：`0.4261`
- Recall：`0.2184`
- Specificity：`0.9117`
- Precision：`0.8526`
- F1：`0.3477`
- Balanced Acc：`0.5650`
- 混淆矩阵：TP=`532` TN=`950` FP=`92` FN=`1904`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.3607 ± 0.0308`
- Val F1-macro：`0.3115 ± 0.0550`
- Test BalAcc：`0.3608 ± 0.0214`
- Test F1-macro：`0.3085 ± 0.0303`
- Test Acc：`0.3594 ± 0.0136`
- Test Precision-macro：`0.3838 ± 0.0357`
- Test Recall-macro：`0.3608 ± 0.0214`
- Test Recall idle/left/right：`0.4254±0.3252` / `0.1946±0.1126` / `0.4625±0.2400`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4148`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4167`
- Val loss（最优时）：`1.0748`

**Test（overall）**
- Accuracy：`0.3567`
- Balanced Acc：`0.3593`
- F1-macro：`0.3544`
- Precision-macro：`0.3546`
- Recall-macro：`0.3593`
- Recall idle/left/right：`0.4536` / `0.3588` / `0.2654`
- Precision idle/left/right：`0.3811` / `0.3445` / `0.3382`
- F1 idle/left/right：`0.4142` / `0.3515` / `0.2974`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1194    773    665
  true1    946    987    818
  true2    993   1105    758
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3299`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2793`
- Val loss（最优时）：`1.1040`

**Test（overall）**
- Accuracy：`0.3612`
- Balanced Acc：`0.3706`
- F1-macro：`0.3271`
- Precision-macro：`0.4033`
- Recall-macro：`0.3706`
- Recall idle/left/right：`0.7349` / `0.1739` / `0.2029`
- Precision idle/left/right：`0.3285` / `0.4429` / `0.4386`
- F1 idle/left/right：`0.4540` / `0.2497` / `0.2775`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1921    291    402
  true1   2013    493    329
  true2   1914    329    571
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3744`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3134`
- Val loss（最优时）：`1.0975`

**Test（overall）**
- Accuracy：`0.3445`
- Balanced Acc：`0.3377`
- F1-macro：`0.2703`
- Precision-macro：`0.3464`
- Recall-macro：`0.3377`
- Recall idle/left/right：`0.0670` / `0.1626` / `0.7835`
- Precision idle/left/right：`0.3750` / `0.3156` / `0.3487`
- F1 idle/left/right：`0.1137` / `0.2146` / `0.4827`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    174    517   1906
  true1    136    444   2150
  true2    154    446   2172
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`46`
- 验证最优轮次（best_epoch）：`28`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3447`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2632`
- Val loss（最优时）：`1.1218`

**Test（overall）**
- Accuracy：`0.3505`
- Balanced Acc：`0.3406`
- F1-macro：`0.2826`
- Precision-macro：`0.3710`
- Recall-macro：`0.3406`
- Recall idle/left/right：`0.0462` / `0.2590` / `0.7166`
- Precision idle/left/right：`0.4458` / `0.2989` / `0.3684`
- F1 idle/left/right：`0.0836` / `0.2775` / `0.4866`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    111    891   1403
  true1     66    669   1848
  true2     72    678   1896
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3397`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2850`
- Val loss（最优时）：`1.1150`

**Test（overall）**
- Accuracy：`0.3841`
- Balanced Acc：`0.3960`
- F1-macro：`0.3083`
- Precision-macro：`0.4435`
- Recall-macro：`0.3960`
- Recall idle/left/right：`0.8253` / `0.0189` / `0.3439`
- Precision idle/left/right：`0.3439` / `0.5000` / `0.4866`
- F1 idle/left/right：`0.4855` / `0.0364` / `0.4030`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    860     10    172
  true1    784     21    308
  true2    857     11    455
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

- 结束：`2026-08-04T08:07:59`
