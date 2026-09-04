# 被试独立五折实验记录（20260821_190340 / shallow_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-21T19:03:40`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=1`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）· Tw=3s hop=100ms · 实验20
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260821_190340`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 1, 'patience': 1, 'batch_train': 64, 'batch_eval': 128, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6778 ± 0.0000`
- Test Acc_paper：`0.7209 ± 0.0000`
- Test BalAcc_maj：`0.6916 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.6745 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`1` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6778`
- Val BalAcc_maj（附报）：`0.6708`

**Test 试次级**
- Acc_paper：`0.7209`
- BalAcc_maj：`0.6916`
- Acc_majority：`0.7209`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6745` | F1：`0.7719` | Acc：`0.7019`

### Three
- （本次跳过）

### 共用超参
```json
{
  "data_tag": "openbmi_3s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 1,
  "patience": 1,
  "batch_train": 64,
  "batch_eval": 128,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 750,
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

- 结束：`2026-08-21T19:04:36`
