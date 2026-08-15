# 被试独立五折实验记录（20260810_000040 / eegtcnet）

- 开始：`2026-08-10T00:00:40`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`eegtcnet`（单脚本；无 registry）
- 结构：EEGTCNet（braindecode 默认结构）
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\eegtcnet\openbmi_balanced_train_2s\run_20260810_000040`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5413 ± 0.0498`
- Test F1：`0.5185 ± 0.2520`
- Test Acc：`0.5338 ± 0.0407`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.6677`
- Val loss（最优时）：`0.7116`

**Test（overall）**
- Accuracy：`0.5002`
- Recall：`0.9995`
- Specificity：`0.0009`
- Precision：`0.5001`
- F1：`0.6667`
- Balanced Acc：`0.5002`
- 混淆矩阵：TP=`2199` TN=`2` FP=`2198` FN=`1`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.0572`
- Val loss（最优时）：`0.7096`

**Test（overall）**
- Accuracy：`0.5009`
- Recall：`0.0091`
- Specificity：`0.9927`
- Precision：`0.5556`
- F1：`0.0179`
- Balanced Acc：`0.5009`
- 混淆矩阵：TP=`20` TN=`2184` FP=`16` FN=`2180`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`67`
- 验证最优轮次（best_epoch）：`47`
- Val F1（最优）：`0.5612`
- Val loss（最优时）：`0.6872`

**Test（overall）**
- Accuracy：`0.5716`
- Recall：`0.6105`
- Specificity：`0.5327`
- Precision：`0.5664`
- F1：`0.5876`
- Balanced Acc：`0.5716`
- 混淆矩阵：TP=`1343` TN=`1172` FP=`1028` FN=`857`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`175`
- 验证最优轮次（best_epoch）：`155`
- Val F1（最优）：`0.6668`
- Val loss（最优时）：`0.6350`

**Test（overall）**
- Accuracy：`0.5941`
- Recall：`0.7636`
- Specificity：`0.4245`
- Precision：`0.5703`
- F1：`0.6529`
- Balanced Acc：`0.5941`
- 混淆矩阵：TP=`1680` TN=`934` FP=`1266` FN=`520`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.6672`
- Val loss（最优时）：`0.7057`

**Test（overall）**
- Accuracy：`0.5022`
- Recall：`0.9985`
- Specificity：`0.0060`
- Precision：`0.5011`
- F1：`0.6673`
- Balanced Acc：`0.5022`
- 混淆矩阵：TP=`1997` TN=`12` FP=`1988` FN=`3`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.3708 ± 0.1104`
- Test F1-macro：`0.2171 ± 0.0435`
- Test Acc：`0.3666 ± 0.1125`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0595`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2222`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2200      0      0
  true1   1100      0      0
  true2   1100      0      0
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`3`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0492`

**Test（overall）**
- Accuracy：`0.4995`
- F1-macro：`0.2222`
- Recall-macro：`0.3330`
- Recall idle/left/right：`0.9991` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2198      2      0
  true1   1100      0      0
  true2   1098      2      0
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`3`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0935`

**Test（overall）**
- Accuracy：`0.3305`
- F1-macro：`0.2750`
- Recall-macro：`0.3232`
- Recall idle/left/right：`0.3523` / `0.5718` / `0.0455`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    775   1314    111
  true1    417    629     54
  true2    438    612     50
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`10`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.1629`

**Test（overall）**
- Accuracy：`0.2486`
- F1-macro：`0.1399`
- Recall-macro：`0.3311`
- Recall idle/left/right：`0.0014` / `0.9818` / `0.0100`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      3   2158     39
  true1      2   1080     18
  true2      0   1089     11
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.1600`

**Test（overall）**
- Accuracy：`0.2545`
- F1-macro：`0.2261`
- Recall-macro：`0.3393`
- Recall idle/left/right：`0.0000` / `0.5340` / `0.4840`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      0   1015    985
  true1      0    534    466
  true2      0    516    484
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

- 结束：`2026-08-10T00:14:08`
