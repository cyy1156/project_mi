# 被试独立五折实验记录（20260803_102407 / shallow_1s_balbatch_balacc）

- 开始：`2026-08-03T10:24:07`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`shallow`（原结构）
- 结构：ShallowFBCSPNet（braindecode 默认）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\shallow_1s_balbatch_balacc\bci2a_1s\run_20260803_102407`

---
## 最终结论（Task only）

- Val BalAcc：`0.5936 ± 0.0399`
- Test BalAcc：`0.5593 ± 0.0064`
- Test Spec：`0.5411 ± 0.1909`
- Test Rec：`0.5774 ± 0.1984`
- Test F1：`0.6210 ± 0.1427`
- Test Acc：`0.5631 ± 0.0779`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.6683`
- Val F1（最优 checkpoint 时，附报）：`0.7274`
- Val loss（最优时）：`0.6121`

**Test（overall）**
- Accuracy：`0.5992`
- Recall：`0.6606`
- Specificity：`0.4700`
- Precision：`0.7240`
- F1：`0.6908`
- Balanced Acc：`0.5653`
- 混淆矩阵：TP=`13405` TN=`4532` FP=`5111` FN=`6887`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`47`
- 验证最优轮次（best_epoch）：`29`
- Val 选模分数（Balanced Acc）：`0.6004`
- Val F1（最优 checkpoint 时，附报）：`0.6882`
- Val loss（最优时）：`0.6945`

**Test（overall）**
- Accuracy：`0.5368`
- Recall：`0.4983`
- Specificity：`0.6186`
- Precision：`0.7355`
- F1：`0.5941`
- Balanced Acc：`0.5585`
- 混淆矩阵：TP=`10188` TN=`5941` FP=`3663` FN=`10256`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5715`
- Val F1（最优 checkpoint 时，附报）：`0.7662`
- Val loss（最优时）：`0.6370`

**Test（overall）**
- Accuracy：`0.6297`
- Recall：`0.7689`
- Specificity：`0.3393`
- Precision：`0.7084`
- F1：`0.7374`
- Balanced Acc：`0.5541`
- 混淆矩阵：TP=`15310` TN=`3236` FP=`6302` FN=`4602`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5694`
- Val F1（最优 checkpoint 时，附报）：`0.6352`
- Val loss（最优时）：`0.6920`

**Test（overall）**
- Accuracy：`0.6275`
- Recall：`0.7325`
- Specificity：`0.4030`
- Precision：`0.7242`
- F1：`0.7283`
- Balanced Acc：`0.5677`
- 混淆矩阵：TP=`13861` TN=`3564` FP=`5279` FN=`5063`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5583`
- Val F1（最优 checkpoint 时，附报）：`0.7204`
- Val loss（最优时）：`0.6860`

**Test（overall）**
- Accuracy：`0.4225`
- Recall：`0.2269`
- Specificity：`0.8749`
- Precision：`0.8074`
- F1：`0.3542`
- Balanced Acc：`0.5509`
- 混淆矩阵：TP=`2000` TN=`3335` FP=`477` FN=`6816`

### 共用超参
```json
{
  "data_tag": "bci2a_1s",
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
  "protocol": "1s-offline-native-arch",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 250,
  "no_rap": true
}
```

- 结束：`2026-08-03T10:43:09`
