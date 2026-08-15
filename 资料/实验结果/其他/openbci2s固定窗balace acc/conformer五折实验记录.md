# 被试独立五折实验记录（20260810_001418 / conformer）

- 开始：`2026-08-10T00:14:18`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`conformer`（单脚本；无 registry）
- 结构：EEGConformer(num_layers=2, num_heads=10, att_drop=0.5)
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\conformer\openbmi_balanced_train_2s\run_20260810_001418`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5950 ± 0.0364`
- Test F1：`0.6303 ± 0.0634`
- Test Acc：`0.5925 ± 0.0269`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`126`
- 验证最优轮次（best_epoch）：`106`
- Val F1（最优）：`0.6510`
- Val loss（最优时）：`0.7324`

**Test（overall）**
- Accuracy：`0.6136`
- Recall：`0.6614`
- Specificity：`0.5659`
- Precision：`0.6037`
- F1：`0.6312`
- Balanced Acc：`0.6136`
- 混淆矩阵：TP=`1455` TN=`1245` FP=`955` FN=`745`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`9`
- Val F1（最优）：`0.6591`
- Val loss（最优时）：`0.7263`

**Test（overall）**
- Accuracy：`0.5961`
- Recall：`0.7641`
- Specificity：`0.4282`
- Precision：`0.5720`
- F1：`0.6542`
- Balanced Acc：`0.5961`
- 混淆矩阵：TP=`1681` TN=`942` FP=`1258` FN=`519`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`70`
- 验证最优轮次（best_epoch）：`50`
- Val F1（最优）：`0.6288`
- Val loss（最优时）：`0.7660`

**Test（overall）**
- Accuracy：`0.5848`
- Recall：`0.7986`
- Specificity：`0.3709`
- Precision：`0.5594`
- F1：`0.6579`
- Balanced Acc：`0.5848`
- 混淆矩阵：TP=`1757` TN=`816` FP=`1384` FN=`443`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`61`
- 验证最优轮次（best_epoch）：`41`
- Val F1（最优）：`0.6845`
- Val loss（最优时）：`0.7419`

**Test（overall）**
- Accuracy：`0.6225`
- Recall：`0.8691`
- Specificity：`0.3759`
- Precision：`0.5820`
- F1：`0.6972`
- Balanced Acc：`0.6225`
- 混淆矩阵：TP=`1912` TN=`827` FP=`1373` FN=`288`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.4575`
- Val loss（最优时）：`0.6924`

**Test（overall）**
- Accuracy：`0.5455`
- Recall：`0.4745`
- Specificity：`0.6165`
- Precision：`0.5530`
- F1：`0.5108`
- Balanced Acc：`0.5455`
- 混淆矩阵：TP=`949` TN=`1233` FP=`767` FN=`1051`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.4668 ± 0.0351`
- Test F1-macro：`0.4282 ± 0.0501`
- Test Acc：`0.4633 ± 0.0219`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0631`

**Test（overall）**
- Accuracy：`0.4361`
- F1-macro：`0.3983`
- Recall-macro：`0.4033`
- Recall idle/left/right：`0.5345` / `0.2400` / `0.4355`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1176    309    715
  true1    439    264    397
  true2    485    136    479
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`14`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.1013`

**Test（overall）**
- Accuracy：`0.4816`
- F1-macro：`0.4778`
- Recall-macro：`0.5074`
- Recall idle/left/right：`0.4041` / `0.5036` / `0.6145`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    889    535    776
  true1    184    554    362
  true2    156    268    676
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`22`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.1095`

**Test（overall）**
- Accuracy：`0.4393`
- F1-macro：`0.4338`
- Recall-macro：`0.4556`
- Recall idle/left/right：`0.3905` / `0.4018` / `0.5745`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    859    482    859
  true1    269    442    389
  true2    307    161    632
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`89`
- 验证最优轮次（best_epoch）：`69`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0495`

**Test（overall）**
- Accuracy：`0.4900`
- F1-macro：`0.4821`
- Recall-macro：`0.5171`
- Recall idle/left/right：`0.4086` / `0.7255` / `0.4173`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    899    845    456
  true1    152    798    150
  true2    206    435    459
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0740`

**Test（overall）**
- Accuracy：`0.4695`
- F1-macro：`0.3491`
- Recall-macro：`0.3923`
- Recall idle/left/right：`0.7010` / `0.4490` / `0.0270`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1402    571     27
  true1    521    449     30
  true2    685    288     27
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

- 结束：`2026-08-10T00:33:11`
