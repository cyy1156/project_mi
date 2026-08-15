# 被试独立五折实验记录（20260810_011107 / gcbnet_raw）

- 开始：`2026-08-10T01:11:07`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`gcbnet_raw`（单脚本；无 registry）
- 结构：TemporalEncoder(D=64) + GCBNet(k=2, layers=[128])；raw 时域输入，8 导联偶数
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\gcbnet_raw\openbmi_balanced_train_2s\run_20260810_011107`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5812 ± 0.0267`
- Test F1：`0.5137 ± 0.0852`
- Test Acc：`0.5705 ± 0.0314`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`172`
- 验证最优轮次（best_epoch）：`152`
- Val F1（最优）：`0.5061`
- Val loss（最优时）：`0.7516`

**Test（overall）**
- Accuracy：`0.5432`
- Recall：`0.3486`
- Specificity：`0.7377`
- Precision：`0.5707`
- F1：`0.4328`
- Balanced Acc：`0.5432`
- 混淆矩阵：TP=`767` TN=`1623` FP=`577` FN=`1433`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`171`
- 验证最优轮次（best_epoch）：`151`
- Val F1（最优）：`0.5849`
- Val loss（最优时）：`0.7032`

**Test（overall）**
- Accuracy：`0.6132`
- Recall：`0.4964`
- Specificity：`0.7300`
- Precision：`0.6477`
- F1：`0.5620`
- Balanced Acc：`0.6132`
- 混淆矩阵：TP=`1092` TN=`1606` FP=`594` FN=`1108`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.4323`
- Val loss（最优时）：`0.6905`

**Test（overall）**
- Accuracy：`0.5341`
- Recall：`0.3045`
- Specificity：`0.7636`
- Precision：`0.5630`
- F1：`0.3953`
- Balanced Acc：`0.5341`
- 混淆矩阵：TP=`670` TN=`1680` FP=`520` FN=`1530`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`122`
- 验证最优轮次（best_epoch）：`102`
- Val F1（最优）：`0.6300`
- Val loss（最优时）：`0.6695`

**Test（overall）**
- Accuracy：`0.6011`
- Recall：`0.6527`
- Specificity：`0.5495`
- Precision：`0.5917`
- F1：`0.6207`
- Balanced Acc：`0.6011`
- 混淆矩阵：TP=`1436` TN=`1209` FP=`991` FN=`764`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`111`
- 验证最优轮次（best_epoch）：`91`
- Val F1（最优）：`0.5112`
- Val loss（最优时）：`0.8067`

**Test（overall）**
- Accuracy：`0.5610`
- Recall：`0.5535`
- Specificity：`0.5685`
- Precision：`0.5619`
- F1：`0.5577`
- Balanced Acc：`0.5610`
- 混淆矩阵：TP=`1107` TN=`1137` FP=`863` FN=`893`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.4969 ± 0.0164`
- Test F1-macro：`0.2893 ± 0.0614`
- Test Acc：`0.4920 ± 0.0258`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`31`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0004`

**Test（overall）**
- Accuracy：`0.5123`
- F1-macro：`0.3172`
- Recall-macro：`0.3695`
- Recall idle/left/right：`0.9405` / `0.0764` / `0.0918`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2069     68     63
  true1    964     84     52
  true2    973     26    101
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`24`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0056`

**Test（overall）**
- Accuracy：`0.5039`
- F1-macro：`0.2499`
- Recall-macro：`0.3430`
- Recall idle/left/right：`0.9864` / `0.0309` / `0.0118`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2170     10     20
  true1   1053     34     13
  true2   1070     17     13
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`14`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0286`

**Test（overall）**
- Accuracy：`0.4411`
- F1-macro：`0.3969`
- Recall-macro：`0.3974`
- Recall idle/left/right：`0.5723` / `0.3482` / `0.2718`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1259    512    429
  true1    439    383    278
  true2    532    269    299
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`11`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0077`

**Test（overall）**
- Accuracy：`0.5007`
- F1-macro：`0.2536`
- Recall-macro：`0.3423`
- Recall idle/left/right：`0.9759` / `0.0355` / `0.0155`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2147     46      7
  true1   1039     39     22
  true2   1051     32     17
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`5`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0450`

**Test（overall）**
- Accuracy：`0.5022`
- F1-macro：`0.2290`
- Recall-macro：`0.3363`
- Recall idle/left/right：`1.0000` / `0.0020` / `0.0070`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2000      0      0
  true1    990      2      8
  true2    988      5      7
```

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 20,
  "batch_train": 128,
  "batch_eval": 256,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "prefetch_factor": 2,
  "non_blocking": true,
  "torch_num_threads": 6,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true
}
```

- 结束：`2026-08-10T01:27:07`
