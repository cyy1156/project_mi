# 被试独立五折实验记录（20260803_122729 / eegtcnet_1s_balbatch_balacc）

- 开始：`2026-08-03T12:27:29`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`eegtcnet`（原结构）
- 结构：EEGTCNet（braindecode 默认）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\eegtcnet_1s_balbatch_balacc\bci2a_1s\run_20260803_122729`

---
## 最终结论（Task only）

- Val BalAcc：`0.5837 ± 0.0461`
- Test BalAcc：`0.5433 ± 0.0076`
- Test Spec：`0.5580 ± 0.2367`
- Test Rec：`0.5287 ± 0.2281`
- Test F1：`0.5768 ± 0.1623`
- Test Acc：`0.5350 ± 0.0827`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc）：`0.6743`
- Val F1（最优 checkpoint 时，附报）：`0.7974`
- Val loss（最优时）：`0.5906`

**Test（overall）**
- Accuracy：`0.5771`
- Recall：`0.6413`
- Specificity：`0.4420`
- Precision：`0.7075`
- F1：`0.6728`
- Balanced Acc：`0.5417`
- 混淆矩阵：TP=`13014` TN=`4262` FP=`5381` FN=`7278`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`77`
- 验证最优轮次（best_epoch）：`59`
- Val 选模分数（Balanced Acc）：`0.5729`
- Val F1（最优 checkpoint 时，附报）：`0.5332`
- Val loss（最优时）：`0.7638`

**Test（overall）**
- Accuracy：`0.4580`
- Recall：`0.3189`
- Specificity：`0.7542`
- Precision：`0.7341`
- F1：`0.4446`
- Balanced Acc：`0.5365`
- 混淆矩阵：TP=`6519` TN=`7243` FP=`2361` FN=`13925`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val 选模分数（Balanced Acc）：`0.5468`
- Val F1（最优 checkpoint 时，附报）：`0.7386`
- Val loss（最优时）：`0.6587`

**Test（overall）**
- Accuracy：`0.6336`
- Recall：`0.8025`
- Specificity：`0.2810`
- Precision：`0.6997`
- F1：`0.7476`
- Balanced Acc：`0.5417`
- 混淆矩阵：TP=`15979` TN=`2680` FP=`6858` FN=`3933`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5612`
- Val F1（最优 checkpoint 时，附报）：`0.7019`
- Val loss（最优时）：`0.6819`

**Test（overall）**
- Accuracy：`0.5893`
- Recall：`0.6780`
- Specificity：`0.3993`
- Precision：`0.7072`
- F1：`0.6923`
- Balanced Acc：`0.5387`
- 混淆矩阵：TP=`12831` TN=`3531` FP=`5312` FN=`6093`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`100`
- 验证最优轮次（best_epoch）：`82`
- Val 选模分数（Balanced Acc）：`0.5631`
- Val F1（最优 checkpoint 时，附报）：`0.7016`
- Val loss（最优时）：`0.6615`

**Test（overall）**
- Accuracy：`0.4172`
- Recall：`0.2027`
- Specificity：`0.9134`
- Precision：`0.8441`
- F1：`0.3269`
- Balanced Acc：`0.5581`
- 混淆矩阵：TP=`1787` TN=`3482` FP=`330` FN=`7029`

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

- 结束：`2026-08-03T13:49:17`
