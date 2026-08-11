# 被试独立五折实验记录（20260811_092840 / shallow_s3_mlp_net_enhance_three）

- 开始：`2026-08-11T09:28:40`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\挑战杯专属\project_mi\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=2`
- model：`shallow_s3_mlp` | ShallowFBCSPNet（AdaptiveAvgPool→MLP(n→64→out)+Dropout）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_shallow_net_enhance_three_accpaper\shallow_s3_mlp_net_enhance_three\openbmi_2s_hop100\run_20260811_092840`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 2, 'patience': 2, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6189 ± 0.0000`
- Test Acc_paper：`0.6718 ± 0.0000`
- Test BalAcc_maj：`0.6743 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.6444 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`2` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6189`
- Val BalAcc_maj（附报）：`0.6317`

**Test 试次级**
- Acc_paper：`0.6718`
- BalAcc_maj：`0.6743`
- Acc_majority：`0.6718`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6444` | F1：`0.7030` | Acc：`0.6416`

### Three
- Val Acc_paper：`0.4870 ± 0.0000`
- Test Acc_paper：`0.5097 ± 0.0000`
- Test BalAcc_maj：`0.5282 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.5039 ± 0.0000`

### Three 分折明细

#### Fold 0

- stopped_epoch：`2` | best_epoch：`2`
- Val Acc_paper（早停）：`0.4870`
- Val BalAcc_maj（附报）：`0.5048`

**Test 试次级**
- Acc_paper：`0.5097`
- BalAcc_maj：`0.5282`
- F1-macro（众数）：`0.5268`
- Rec idle/left/right：`0.5927` / `0.5355` / `0.4564`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5039` | F1m：`0.5026`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 2,
  "patience": 2,
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
  "num_workers": 0,
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

- 结束：`2026-08-11T09:33:31`
