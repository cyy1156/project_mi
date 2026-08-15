# 被试独立五折实验记录（20260809_233948 / shallow）

- 开始：`2026-08-09T23:39:48`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`shallow`（单脚本；无 registry）
- 结构：ShallowFBCSPNet（braindecode 默认结构）
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\shallow\openbmi_balanced_train_2s\run_20260809_233948`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5969 ± 0.0265`
- Test F1：`0.6095 ± 0.0293`
- Test Acc：`0.6057 ± 0.0151`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`78`
- 验证最优轮次（best_epoch）：`58`
- Val F1（最优）：`0.5930`
- Val loss（最优时）：`0.6871`

**Test（overall）**
- Accuracy：`0.6186`
- Recall：`0.4995`
- Specificity：`0.7377`
- Precision：`0.6557`
- F1：`0.5671`
- Balanced Acc：`0.6186`
- 混淆矩阵：TP=`1099` TN=`1623` FP=`577` FN=`1101`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`94`
- 验证最优轮次（best_epoch）：`74`
- Val F1（最优）：`0.6424`
- Val loss（最优时）：`0.7094`

**Test（overall）**
- Accuracy：`0.6155`
- Recall：`0.6641`
- Specificity：`0.5668`
- Precision：`0.6052`
- F1：`0.6333`
- Balanced Acc：`0.6155`
- 混淆矩阵：TP=`1461` TN=`1247` FP=`953` FN=`739`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`71`
- 验证最优轮次（best_epoch）：`51`
- Val F1（最优）：`0.5420`
- Val loss（最优时）：`0.7649`

**Test（overall）**
- Accuracy：`0.5984`
- Recall：`0.6373`
- Specificity：`0.5595`
- Precision：`0.5913`
- F1：`0.6134`
- Balanced Acc：`0.5984`
- 混淆矩阵：TP=`1402` TN=`1231` FP=`969` FN=`798`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`56`
- 验证最优轮次（best_epoch）：`36`
- Val F1（最优）：`0.6301`
- Val loss（最优时）：`0.6580`

**Test（overall）**
- Accuracy：`0.6168`
- Recall：`0.7023`
- Specificity：`0.5314`
- Precision：`0.5998`
- F1：`0.6470`
- Balanced Acc：`0.6168`
- 混淆矩阵：TP=`1545` TN=`1169` FP=`1031` FN=`655`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`46`
- 验证最优轮次（best_epoch）：`26`
- Val F1（最优）：`0.5923`
- Val loss（最优时）：`0.7471`

**Test（overall）**
- Accuracy：`0.5793`
- Recall：`0.5975`
- Specificity：`0.5610`
- Precision：`0.5765`
- F1：`0.5868`
- Balanced Acc：`0.5793`
- 混淆矩阵：TP=`1195` TN=`1122` FP=`878` FN=`805`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.4823 ± 0.0388`
- Test F1-macro：`0.4676 ± 0.0133`
- Test Acc：`0.4866 ± 0.0188`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`96`
- 验证最优轮次（best_epoch）：`76`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0213`

**Test（overall）**
- Accuracy：`0.5145`
- F1-macro：`0.4752`
- Recall-macro：`0.4735`
- Recall idle/left/right：`0.6377` / `0.3718` / `0.4109`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1403    355    442
  true1    486    409    205
  true2    444    204    452
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`80`
- 验证最优轮次（best_epoch）：`60`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0744`

**Test（overall）**
- Accuracy：`0.4945`
- F1-macro：`0.4808`
- Recall-macro：`0.4988`
- Recall idle/left/right：`0.4818` / `0.4136` / `0.6009`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1060    444    696
  true1    254    455    391
  true2    222    217    661
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`49`
- 验证最优轮次（best_epoch）：`29`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0683`

**Test（overall）**
- Accuracy：`0.4573`
- F1-macro：`0.4424`
- Recall-macro：`0.4521`
- Recall idle/left/right：`0.4727` / `0.4082` / `0.4755`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1040    507    653
  true1    334    449    317
  true2    363    214    523
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`74`
- 验证最优轮次（best_epoch）：`54`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`0.9563`

**Test（overall）**
- Accuracy：`0.4789`
- F1-macro：`0.4712`
- Recall-macro：`0.4980`
- Recall idle/left/right：`0.4214` / `0.6418` / `0.4309`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    927    778    495
  true1    189    706    205
  true2    224    402    474
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`62`
- 验证最优轮次（best_epoch）：`42`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.1876`

**Test（overall）**
- Accuracy：`0.4880`
- F1-macro：`0.4686`
- Recall-macro：`0.4747`
- Recall idle/left/right：`0.5280` / `0.4370` / `0.4590`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1056    464    480
  true1    323    437    240
  true2    360    181    459
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

- 结束：`2026-08-09T23:50:17`
