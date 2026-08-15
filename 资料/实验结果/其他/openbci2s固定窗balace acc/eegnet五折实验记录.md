# 被试独立五折实验记录（20260809_232739 / eegnet）

- 开始：`2026-08-09T23:27:39`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`eegnet`（单脚本；无 registry）
- 结构：EEGNet(F1=8, D=2, F2=16)
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\eegnet\openbmi_balanced_train_2s\run_20260809_232739`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5912 ± 0.0273`
- Test F1：`0.5813 ± 0.0589`
- Test Acc：`0.5767 ± 0.0295`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`112`
- 验证最优轮次（best_epoch）：`92`
- Val F1（最优）：`0.5337`
- Val loss（最优时）：`0.6679`

**Test（overall）**
- Accuracy：`0.5843`
- Recall：`0.3741`
- Specificity：`0.7945`
- Precision：`0.6455`
- F1：`0.4737`
- Balanced Acc：`0.5843`
- 混淆矩阵：TP=`823` TN=`1748` FP=`452` FN=`1377`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`111`
- 验证最优轮次（best_epoch）：`91`
- Val F1（最优）：`0.6499`
- Val loss（最优时）：`0.6600`

**Test（overall）**
- Accuracy：`0.5795`
- Recall：`0.6318`
- Specificity：`0.5273`
- Precision：`0.5720`
- F1：`0.6004`
- Balanced Acc：`0.5795`
- 混淆矩阵：TP=`1390` TN=`1160` FP=`1040` FN=`810`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`54`
- 验证最优轮次（best_epoch）：`34`
- Val F1（最优）：`0.5471`
- Val loss（最优时）：`0.6781`

**Test（overall）**
- Accuracy：`0.5527`
- Recall：`0.6573`
- Specificity：`0.4482`
- Precision：`0.5436`
- F1：`0.5951`
- Balanced Acc：`0.5527`
- 混淆矩阵：TP=`1446` TN=`986` FP=`1214` FN=`754`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`114`
- 验证最优轮次（best_epoch）：`94`
- Val F1（最优）：`0.6141`
- Val loss（最优时）：`0.6525`

**Test（overall）**
- Accuracy：`0.6261`
- Recall：`0.7050`
- Specificity：`0.5473`
- Precision：`0.6090`
- F1：`0.6535`
- Balanced Acc：`0.6261`
- 混淆矩阵：TP=`1551` TN=`1204` FP=`996` FN=`649`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`11`
- Val F1（最优）：`0.5762`
- Val loss（最优时）：`0.6928`

**Test（overall）**
- Accuracy：`0.5410`
- Recall：`0.6445`
- Specificity：`0.4375`
- Precision：`0.5340`
- F1：`0.5841`
- Balanced Acc：`0.5410`
- 混淆矩阵：TP=`1289` TN=`875` FP=`1125` FN=`711`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.4488 ± 0.0421`
- Test F1-macro：`0.3615 ± 0.0815`
- Test Acc：`0.4403 ± 0.0406`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0935`

**Test（overall）**
- Accuracy：`0.4198`
- F1-macro：`0.3493`
- Recall-macro：`0.3523`
- Recall idle/left/right：`0.6223` / `0.2009` / `0.2336`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1369    411    420
  true1    641    221    238
  true2    640    203    257
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`161`
- 验证最优轮次（best_epoch）：`141`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0063`

**Test（overall）**
- Accuracy：`0.4727`
- F1-macro：`0.4697`
- Recall-macro：`0.4980`
- Recall idle/left/right：`0.3968` / `0.5291` / `0.5682`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    873    639    688
  true1    204    582    314
  true2    202    273    625
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`36`
- 验证最优轮次（best_epoch）：`16`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0747`

**Test（overall）**
- Accuracy：`0.4232`
- F1-macro：`0.4054`
- Recall-macro：`0.4170`
- Recall idle/left/right：`0.4418` / `0.3109` / `0.4982`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    972    545    683
  true1    407    342    351
  true2    364    188    548
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0877`

**Test（overall）**
- Accuracy：`0.4998`
- F1-macro：`0.2222`
- Recall-macro：`0.3332`
- Recall idle/left/right：`0.9995` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2199      0      1
  true1   1100      0      0
  true2   1100      0      0
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`8`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0900`

**Test（overall）**
- Accuracy：`0.3862`
- F1-macro：`0.3609`
- Recall-macro：`0.3632`
- Recall idle/left/right：`0.4555` / `0.3240` / `0.3100`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    911    492    597
  true1    401    324    275
  true2    411    279    310
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

- 结束：`2026-08-09T23:39:41`
