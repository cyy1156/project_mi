# 被试独立五折实验记录（20260804_103643 / gcbnet_raw_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:36:43`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`gcbnet_raw`（原结构）
- 结构：TemporalEncoder(D=64) + GCBNet(k=2)；fixed2s 原始时域 (B,8,500)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\gcbnet_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103643`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5376 ± 0.0395`
- Test BalAcc：`0.5096 ± 0.0304`
- Test Spec：`0.8637 ± 0.1414`
- Test Rec：`0.1556 ± 0.1489`
- Test F1：`0.2207 ± 0.1974`
- Test Acc：`0.3817 ± 0.0659`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5302`
- Val F1（最优 checkpoint 时，附报）：`0.1463`
- Val loss（最优时）：`0.6972`

**Test（overall）**
- Accuracy：`0.4924`
- Recall：`0.3895`
- Specificity：`0.7054`
- Precision：`0.7324`
- F1：`0.5086`
- Balanced Acc：`0.5475`
- 混淆矩阵：TP=`104` TN=`91` FP=`38` FN=`163`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.7126`

**Test（overall）**
- Accuracy：`0.3241`
- Recall：`0.0037`
- Specificity：`0.9922`
- Precision：`0.5000`
- F1：`0.0074`
- Balanced Acc：`0.4980`
- 混淆矩阵：TP=`1` TN=`128` FP=`1` FN=`268`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.6113`
- Val F1（最优 checkpoint 时，附报）：`0.6455`
- Val loss（最优时）：`0.6881`

**Test（overall）**
- Accuracy：`0.3897`
- Recall：`0.2481`
- Specificity：`0.6797`
- Precision：`0.6132`
- F1：`0.3533`
- Balanced Acc：`0.4639`
- 混淆矩阵：TP=`65` TN=`87` FP=`41` FN=`197`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5391`
- Val F1（最优 checkpoint 时，附报）：`0.3293`
- Val loss（最优时）：`0.6964`

**Test（overall）**
- Accuracy：`0.3967`
- Recall：`0.1365`
- Specificity：`0.9412`
- Precision：`0.8293`
- F1：`0.2345`
- Balanced Acc：`0.5389`
- 混淆矩阵：TP=`34` TN=`112` FP=`7` FN=`215`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5075`
- Val F1（最优 checkpoint 时，附报）：`0.0446`
- Val loss（最优时）：`0.7153`

**Test（overall）**
- Accuracy：`0.3054`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`51` FP=`0` FN=`116`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4153 ± 0.0393`
- Val F1-macro：`0.3410 ± 0.0779`
- Test BalAcc：`0.3693 ± 0.0293`
- Test F1-macro：`0.2757 ± 0.0897`
- Test Acc：`0.3618 ± 0.0378`
- Test Precision-macro：`0.3101 ± 0.1269`
- Test Recall-macro：`0.3693 ± 0.0293`
- Test Recall idle/left/right：`0.5318±0.3868` / `0.3880±0.3106` / `0.1882±0.2399`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3878`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2983`
- Val loss（最优时）：`1.1010`

**Test（overall）**
- Accuracy：`0.3662`
- Balanced Acc：`0.3703`
- F1-macro：`0.2724`
- Precision-macro：`0.2570`
- Recall-macro：`0.3703`
- Recall idle/left/right：`0.2558` / `0.8550` / `0.0000`
- Precision idle/left/right：`0.4177` / `0.3533` / `0.0000`
- F1 idle/left/right：`0.3173` / `0.5000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     33     96      0
  true1     19    112      0
  true2     27    109      0
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4443`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3557`
- Val loss（最优时）：`1.0948`

**Test（overall）**
- Accuracy：`0.3442`
- Balanced Acc：`0.3526`
- F1-macro：`0.2163`
- Precision-macro：`0.3105`
- Recall-macro：`0.3526`
- Recall idle/left/right：`0.9690` / `0.0889` / `0.0000`
- Precision idle/left/right：`0.3316` / `0.6000` / `0.0000`
- F1 idle/left/right：`0.4941` / `0.1548` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    125      3      1
  true1    123     12      0
  true2    129      5      0
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`71`
- 验证最优轮次（best_epoch）：`53`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4439`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3890`
- Val loss（最优时）：`1.0797`

**Test（overall）**
- Accuracy：`0.3718`
- Balanced Acc：`0.3690`
- F1-macro：`0.3139`
- Precision-macro：`0.4522`
- Recall-macro：`0.3690`
- Recall idle/left/right：`0.0391` / `0.4923` / `0.5758`
- Precision idle/left/right：`0.6250` / `0.3478` / `0.3838`
- F1 idle/left/right：`0.0735` / `0.4076` / `0.4606`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      5     66     57
  true1      1     64     65
  true2      2     54     76
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`63`
- 验证最优轮次（best_epoch）：`45`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4495`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4447`
- Val loss（最优时）：`1.0410`

**Test（overall）**
- Accuracy：`0.4212`
- Balanced Acc：`0.4214`
- F1-macro：`0.4201`
- Precision-macro：`0.4287`
- Recall-macro：`0.4214`
- Recall idle/left/right：`0.3950` / `0.5041` / `0.3651`
- Precision idle/left/right：`0.3643` / `0.4218` / `0.5000`
- F1 idle/left/right：`0.3790` / `0.4593` / `0.4220`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     47     50     22
  true1     37     62     24
  true2     45     35     46
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3510`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2174`
- Val loss（最优时）：`1.1022`

**Test（overall）**
- Accuracy：`0.3054`
- Balanced Acc：`0.3333`
- F1-macro：`0.1560`
- Precision-macro：`0.1018`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- Precision idle/left/right：`0.3054` / `0.0000` / `0.0000`
- F1 idle/left/right：`0.4679` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     51      0      0
  true1     53      0      0
  true2     63      0      0
```

### 共用超参
```json
{
  "data_tag": "bci2a_2s",
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
  "protocol": "fixed-2s-cue2to4-bci2a",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true
}
```

- 结束：`2026-08-04T10:38:05`
