# 被试独立五折实验记录（20260803_134929 / conformer_1s_balbatch_balacc）

- 开始：`2026-08-03T13:49:29`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`conformer`（原结构）
- 结构：EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\conformer_1s_balbatch_balacc\bci2a_1s\run_20260803_134929`

---
## 最终结论（Task only）

- Val BalAcc：`0.5886 ± 0.0450`
- Test BalAcc：`0.5517 ± 0.0120`
- Test Spec：`0.5114 ± 0.1594`
- Test Rec：`0.5920 ± 0.1474`
- Test F1：`0.6391 ± 0.0887`
- Test Acc：`0.5645 ± 0.0526`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.6765`
- Val F1（最优 checkpoint 时，附报）：`0.7742`
- Val loss（最优时）：`0.5937`

**Test（overall）**
- Accuracy：`0.5534`
- Recall：`0.5591`
- Specificity：`0.5413`
- Precision：`0.7195`
- F1：`0.6293`
- Balanced Acc：`0.5502`
- 混淆矩阵：TP=`11346` TN=`5220` FP=`4423` FN=`8946`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5832`
- Val F1（最优 checkpoint 时，附报）：`0.7447`
- Val loss（最优时）：`0.6407`

**Test（overall）**
- Accuracy：`0.5440`
- Recall：`0.5613`
- Specificity：`0.5072`
- Precision：`0.7080`
- F1：`0.6262`
- Balanced Acc：`0.5343`
- 混淆矩阵：TP=`11476` TN=`4871` FP=`4733` FN=`8968`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5668`
- Val F1（最优 checkpoint 时，附报）：`0.7601`
- Val loss（最优时）：`0.6674`

**Test（overall）**
- Accuracy：`0.6054`
- Recall：`0.7124`
- Specificity：`0.3819`
- Precision：`0.7064`
- F1：`0.7094`
- Balanced Acc：`0.5472`
- 混淆矩阵：TP=`14186` TN=`3643` FP=`5895` FN=`5726`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5635`
- Val F1（最优 checkpoint 时，附报）：`0.6549`
- Val loss（最优时）：`0.6868`

**Test（overall）**
- Accuracy：`0.6358`
- Recall：`0.7762`
- Specificity：`0.3352`
- Precision：`0.7142`
- F1：`0.7439`
- Balanced Acc：`0.5557`
- 混淆矩阵：TP=`14689` TN=`2964` FP=`5879` FN=`4235`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5530`
- Val F1（最优 checkpoint 时，附报）：`0.6799`
- Val loss（最优时）：`0.6645`

**Test（overall）**
- Accuracy：`0.4838`
- Recall：`0.3508`
- Specificity：`0.7912`
- Precision：`0.7953`
- F1：`0.4869`
- Balanced Acc：`0.5710`
- 混淆矩阵：TP=`3093` TN=`3016` FP=`796` FN=`5723`

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

- 结束：`2026-08-03T14:34:20`
