# 被试独立五折实验记录（20260810_003820 / gcbnet）

- 开始：`2026-08-10T00:38:20`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`gcbnet`（单脚本；无 registry）
- 结构：GCBNet(k=2, layers=[128], dropout=shared drop_prob)；8 导联偶数
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\gcbnet\openbmi_balanced_train_2s\run_20260810_003820`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5371 ± 0.0198`
- Test F1：`0.5363 ± 0.0406`
- Test Acc：`0.5496 ± 0.0160`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`68`
- 验证最优轮次（best_epoch）：`48`
- Val F1（最优）：`0.5373`
- Val loss（最优时）：`0.6869`

**Test（overall）**
- Accuracy：`0.5443`
- Recall：`0.5327`
- Specificity：`0.5559`
- Precision：`0.5454`
- F1：`0.5390`
- Balanced Acc：`0.5443`
- 混淆矩阵：TP=`1172` TN=`1223` FP=`977` FN=`1028`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`61`
- 验证最优轮次（best_epoch）：`41`
- Val F1（最优）：`0.5034`
- Val loss（最优时）：`0.6895`

**Test（overall）**
- Accuracy：`0.5814`
- Recall：`0.5759`
- Specificity：`0.5868`
- Precision：`0.5823`
- F1：`0.5791`
- Balanced Acc：`0.5814`
- 混淆矩阵：TP=`1267` TN=`1291` FP=`909` FN=`933`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`75`
- 验证最优轮次（best_epoch）：`55`
- Val F1（最优）：`0.5153`
- Val loss（最优时）：`0.6983`

**Test（overall）**
- Accuracy：`0.5418`
- Recall：`0.4591`
- Specificity：`0.6245`
- Precision：`0.5501`
- F1：`0.5005`
- Balanced Acc：`0.5418`
- 混淆矩阵：TP=`1010` TN=`1374` FP=`826` FN=`1190`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.4438`
- Val loss（最优时）：`0.6858`

**Test（overall）**
- Accuracy：`0.5389`
- Recall：`0.4277`
- Specificity：`0.6500`
- Precision：`0.5500`
- F1：`0.4812`
- Balanced Acc：`0.5389`
- 混淆矩阵：TP=`941` TN=`1430` FP=`770` FN=`1259`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`10`
- Val F1（最优）：`0.5760`
- Val loss（最优时）：`0.6992`

**Test（overall）**
- Accuracy：`0.5415`
- Recall：`0.6380`
- Specificity：`0.4450`
- Precision：`0.5348`
- F1：`0.5819`
- Balanced Acc：`0.5415`
- 混淆矩阵：TP=`1276` TN=`890` FP=`1110` FN=`724`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.4059 ± 0.0372`
- Test F1-macro：`0.3533 ± 0.0357`
- Test Acc：`0.4099 ± 0.0251`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`12`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0639`

**Test（overall）**
- Accuracy：`0.4568`
- F1-macro：`0.2832`
- Recall-macro：`0.3347`
- Recall idle/left/right：`0.8232` / `0.1691` / `0.0118`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1811    351     38
  true1    901    186     13
  true2    895    192     13
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`75`
- 验证最优轮次（best_epoch）：`55`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0700`

**Test（overall）**
- Accuracy：`0.3925`
- F1-macro：`0.3771`
- Recall-macro：`0.3944`
- Recall idle/left/right：`0.3868` / `0.2773` / `0.5191`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    851    464    885
  true1    332    305    463
  true2    315    214    571
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`25`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0734`

**Test（overall）**
- Accuracy：`0.4114`
- F1-macro：`0.3583`
- Recall-macro：`0.3588`
- Recall idle/left/right：`0.5691` / `0.2300` / `0.2773`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1252    411    537
  true1    552    253    295
  true2    606    189    305
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`12`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0800`

**Test（overall）**
- Accuracy：`0.3857`
- F1-macro：`0.3722`
- Recall-macro：`0.3836`
- Recall idle/left/right：`0.3918` / `0.3136` / `0.4455`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    862    488    850
  true1    343    345    412
  true2    308    302    490
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`8`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0991`

**Test（overall）**
- Accuracy：`0.4030`
- F1-macro：`0.3758`
- Recall-macro：`0.3828`
- Recall idle/left/right：`0.4635` / `0.4340` / `0.2510`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    927    580    493
  true1    369    434    197
  true2    387    362    251
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

- 结束：`2026-08-10T00:45:03`
