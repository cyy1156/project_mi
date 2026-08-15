# 被试独立五折实验记录（20260809_235023 / deep）

- 开始：`2026-08-09T23:50:23`
- device：`cuda`
- data：`D:\挑战杯专属\project_mi_new\code\preprocess_lab\out\openbmi_balanced_train_2s`（prefix=`openbmi_train`）
- model：`deep`（单脚本；无 registry）
- 结构：Deep4Net（braindecode 默认结构）
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`
- weight_transfer：`False` | classifier：`native`
- AMP：`True` | num_workers：`2` | pin_memory：`True`
- early_stop：`acc_paper` | train_sampler：`balanced_invfreq`
- 权重：`D:\挑战杯专属\project_mi_new\code\train_lab\out\baseline\deep\openbmi_balanced_train_2s\run_20260809_235023`

---
## 最终结论

### Task（静息/任务）
- Val acc_paper：`0.5627 ± 0.0277`
- Test F1：`0.5489 ± 0.0649`
- Test Acc：`0.5578 ± 0.0202`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`59`
- 验证最优轮次（best_epoch）：`39`
- Val F1（最优）：`0.5994`
- Val loss（最优时）：`0.6601`

**Test（overall）**
- Accuracy：`0.5868`
- Recall：`0.5000`
- Specificity：`0.6736`
- Precision：`0.6051`
- F1：`0.5475`
- Balanced Acc：`0.5868`
- 混淆矩阵：TP=`1100` TN=`1482` FP=`718` FN=`1100`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.5054`
- Val loss（最优时）：`0.6913`

**Test（overall）**
- Accuracy：`0.5252`
- Recall：`0.4286`
- Specificity：`0.6218`
- Precision：`0.5313`
- F1：`0.4745`
- Balanced Acc：`0.5252`
- 混淆矩阵：TP=`943` TN=`1368` FP=`832` FN=`1257`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`129`
- 验证最优轮次（best_epoch）：`109`
- Val F1（最优）：`0.4410`
- Val loss（最优时）：`0.7244`

**Test（overall）**
- Accuracy：`0.5661`
- Recall：`0.5009`
- Specificity：`0.6314`
- Precision：`0.5761`
- F1：`0.5359`
- Balanced Acc：`0.5661`
- 混淆矩阵：TP=`1102` TN=`1389` FP=`811` FN=`1098`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`38`
- Val F1（最优）：`0.6592`
- Val loss（最优时）：`0.6772`

**Test（overall）**
- Accuracy：`0.5611`
- Recall：`0.8859`
- Specificity：`0.2364`
- Precision：`0.5371`
- F1：`0.6687`
- Balanced Acc：`0.5611`
- 混淆矩阵：TP=`1949` TN=`520` FP=`1680` FN=`251`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`80`
- 验证最优轮次（best_epoch）：`60`
- Val F1（最优）：`0.5737`
- Val loss（最优时）：`0.7174`

**Test（overall）**
- Accuracy：`0.5497`
- Recall：`0.4835`
- Specificity：`0.6160`
- Precision：`0.5573`
- F1：`0.5178`
- Balanced Acc：`0.5497`
- 混淆矩阵：TP=`967` TN=`1232` FP=`768` FN=`1033`

### Three（空闲/左/右，不使用Task的保存的权重，重新使用新的模型训练）
- Val acc_paper：`0.4442 ± 0.0454`
- Test F1-macro：`0.3076 ± 0.0763`
- Test Acc：`0.4424 ± 0.0480`

### Three 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`1`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0480`

**Test（overall）**
- Accuracy：`0.4961`
- F1-macro：`0.2264`
- Recall-macro：`0.3321`
- Recall idle/left/right：`0.9882` / `0.0018` / `0.0064`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2174      7     19
  true1   1089      2      9
  true2   1092      1      7
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`14`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0608`

**Test（overall）**
- Accuracy：`0.5000`
- F1-macro：`0.2223`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   2200      0      0
  true1   1099      0      1
  true2   1100      0      0
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`5`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0750`

**Test（overall）**
- Accuracy：`0.3957`
- F1-macro：`0.3152`
- Recall-macro：`0.3541`
- Recall idle/left/right：`0.5205` / `0.0436` / `0.4982`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1145     73    982
  true1    534     48    518
  true2    517     35    548
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`4`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0817`

**Test（overall）**
- Accuracy：`0.3870`
- F1-macro：`0.3514`
- Recall-macro：`0.3600`
- Recall idle/left/right：`0.4682` / `0.1882` / `0.4236`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1030    298    872
  true1    455    207    438
  true2    484    150    466
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`61`
- 验证最优轮次（best_epoch）：`41`
- Val F1-macro（最优）：`None`
- Val loss（最优时）：`1.0720`

**Test（overall）**
- Accuracy：`0.4333`
- F1-macro：`0.4228`
- Recall-macro：`0.4353`
- Recall idle/left/right：`0.4270` / `0.4040` / `0.4750`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    854    502    644
  true1    326    404    270
  true2    322    203    475
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

- 结束：`2026-08-10T00:00:34`
