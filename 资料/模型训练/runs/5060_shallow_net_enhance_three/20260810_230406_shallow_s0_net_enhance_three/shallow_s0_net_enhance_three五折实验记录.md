# 被试独立五折实验记录（20260810_230406 / shallow_s0_net_enhance_three）

- 开始：`2026-08-10T23:04:06`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\挑战杯专属\project_mi\code\preprocess_lab\out\openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=2`
- model：`shallow_s0` | ShallowFBCSPNet（braindecode 默认；S0 复现锚点）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_shallow_net_enhance_three_accpaper\shallow_s0_net_enhance_three\openbmi_2s_fixed_cue2to4_noz\run_20260810_230406`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 2, 'patience': 2, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6296 ± 0.0000`
- Test Acc_paper：`0.6130 ± 0.0000`
- Test BalAcc_maj：`0.5730 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.5730 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`2` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6296`
- Val BalAcc_maj（附报）：`0.5747`

**Test 试次级**
- Acc_paper：`0.6130`
- BalAcc_maj：`0.5730`
- Acc_majority：`0.6130`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5730` | F1：`0.7049` | Acc：`0.6130`

### Three
- Val Acc_paper：`0.3963 ± 0.0000`
- Test Acc_paper：`0.3961 ± 0.0000`
- Test BalAcc_maj：`0.3961 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.3961 ± 0.0000`

### Three 分折明细

#### Fold 0

- stopped_epoch：`2` | best_epoch：`2`
- Val Acc_paper（早停）：`0.3963`
- Val BalAcc_maj（附报）：`0.3963`

**Test 试次级**
- Acc_paper：`0.3961`
- BalAcc_maj：`0.3961`
- F1-macro（众数）：`0.3832`
- Rec idle/left/right：`0.3173` / `0.2636` / `0.6073`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3961` | F1m：`0.3832`

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

- 结束：`2026-08-10T23:20:32`
