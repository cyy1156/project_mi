# 被试独立五折实验记录（20260804_080802 / dbn_raw_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T08:08:02`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`dbn_raw`（原结构）
- 结构：TemporalEncoder(D=64) + DBN；2s/hop100 原始时域 (B,8,500)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\dbn_raw_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_080802`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5481 ± 0.0307`
- Test BalAcc：`0.5132 ± 0.0131`
- Test Spec：`0.9121 ± 0.1694`
- Test Rec：`0.1144 ± 0.1885`
- Test F1：`0.1477 ± 0.2194`
- Test Acc：`0.3650 ± 0.0758`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`52`
- 验证最优轮次（best_epoch）：`34`
- Val 选模分数（Balanced Acc）：`0.5964`
- Val F1（最优 checkpoint 时，附报）：`0.4280`
- Val loss（最优时）：`0.8790`

**Test（overall）**
- Accuracy：`0.3304`
- Recall：`0.0218`
- Specificity：`0.9878`
- Precision：`0.7922`
- F1：`0.0424`
- Balanced Acc：`0.5048`
- 混淆矩阵：TP=`122` TN=`2600` FP=`32` FN=`5485`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5394`
- Val F1（最优 checkpoint 时，附报）：`0.2311`
- Val loss（最优时）：`0.7424`

**Test（overall）**
- Accuracy：`0.3200`
- Recall：`0.0053`
- Specificity：`1.0000`
- Precision：`1.0000`
- F1：`0.0106`
- Balanced Acc：`0.5027`
- 混淆矩阵：TP=`30` TN=`2614` FP=`0` FN=`5619`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc）：`0.5004`
- Val F1（最优 checkpoint 时，附报）：`0.0015`
- Val loss（最优时）：`1.4167`

**Test（overall）**
- Accuracy：`0.3214`
- Recall：`0.0011`
- Specificity：`1.0000`
- Precision：`1.0000`
- F1：`0.0022`
- Balanced Acc：`0.5005`
- 混淆矩阵：TP=`6` TN=`2597` FP=`0` FN=`5496`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5510`
- Val F1（最优 checkpoint 时，附报）：`0.6172`
- Val loss（最优时）：`0.6926`

**Test（overall）**
- Accuracy：`0.5160`
- Recall：`0.4896`
- Specificity：`0.5734`
- Precision：`0.7139`
- F1：`0.5808`
- Balanced Acc：`0.5315`
- 混淆矩阵：TP=`2560` TN=`1379` FP=`1026` FN=`2669`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`98`
- 验证最优轮次（best_epoch）：`80`
- Val 选模分数（Balanced Acc）：`0.5533`
- Val F1（最优 checkpoint 时，附报）：`0.5144`
- Val loss（最优时）：`0.9429`

**Test（overall）**
- Accuracy：`0.3373`
- Recall：`0.0542`
- Specificity：`0.9990`
- Precision：`0.9925`
- F1：`0.1028`
- Balanced Acc：`0.5266`
- 混淆矩阵：TP=`132` TN=`1041` FP=`1` FN=`2304`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.3898 ± 0.0501`
- Val F1-macro：`0.3021 ± 0.1068`
- Test BalAcc：`0.3553 ± 0.0194`
- Test F1-macro：`0.2354 ± 0.0634`
- Test Acc：`0.3428 ± 0.0204`
- Test Precision-macro：`0.3240 ± 0.1622`
- Test Recall-macro：`0.3553 ± 0.0194`
- Test Recall idle/left/right：`0.7843±0.3638` / `0.1325±0.2058` / `0.1490±0.1774`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4693`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4131`
- Val loss（最优时）：`1.1119`

**Test（overall）**
- Accuracy：`0.3590`
- Balanced Acc：`0.3685`
- F1-macro：`0.2761`
- Precision-macro：`0.4402`
- Recall-macro：`0.3685`
- Recall idle/left/right：`0.8739` / `0.0360` / `0.1957`
- Precision idle/left/right：`0.3355` / `0.5156` / `0.4694`
- F1 idle/left/right：`0.4848` / `0.0673` / `0.2763`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2300     31    301
  true1   2321     99    331
  true2   2235     62    559
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3386`
- Val F1-macro（最优 checkpoint 时，附报）：`0.1861`
- Val loss（最优时）：`1.1700`

**Test（overall）**
- Accuracy：`0.3165`
- Balanced Acc：`0.3334`
- F1-macro：`0.1615`
- Precision-macro：`0.1674`
- Recall-macro：`0.3334`
- Recall idle/left/right：`0.9985` / `0.0018` / `0.0000`
- Precision idle/left/right：`0.3169` / `0.1852` / `0.0000`
- F1 idle/left/right：`0.4811` / `0.0035` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2610      4      0
  true1   2830      5      0
  true2   2796     18      0
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3333`
- Val F1-macro（最优 checkpoint 时，附报）：`0.1600`
- Val loss（最优时）：`1.3100`

**Test（overall）**
- Accuracy：`0.3207`
- Balanced Acc：`0.3333`
- F1-macro：`0.1619`
- Precision-macro：`0.1069`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- Precision idle/left/right：`0.3207` / `0.0000` / `0.0000`
- F1 idle/left/right：`0.4856` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2597      0      0
  true1   2730      0      0
  true2   2772      0      0
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`39`
- 验证最优轮次（best_epoch）：`21`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4073`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3645`
- Val loss（最优时）：`1.1588`

**Test（overall）**
- Accuracy：`0.3664`
- Balanced Acc：`0.3585`
- F1-macro：`0.3195`
- Precision-macro：`0.3716`
- Recall-macro：`0.3585`
- Recall idle/left/right：`0.0628` / `0.5393` / `0.4735`
- Precision idle/left/right：`0.3738` / `0.3382` / `0.4028`
- F1 idle/left/right：`0.1075` / `0.4157` / `0.4353`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    151   1460    794
  true1    126   1393   1064
  true2    127   1266   1253
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`94`
- 验证最优轮次（best_epoch）：`76`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4006`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3867`
- Val loss（最优时）：`1.2749`

**Test（overall）**
- Accuracy：`0.3516`
- Balanced Acc：`0.3825`
- F1-macro：`0.2582`
- Precision-macro：`0.5341`
- Recall-macro：`0.3825`
- Recall idle/left/right：`0.9866` / `0.0854` / `0.0756`
- Precision idle/left/right：`0.3251` / `0.7480` / `0.5291`
- F1 idle/left/right：`0.4891` / `0.1532` / `0.1323`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1028      9      5
  true1    934     95     84
  true2   1200     23    100
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

- 结束：`2026-08-04T08:34:09`
