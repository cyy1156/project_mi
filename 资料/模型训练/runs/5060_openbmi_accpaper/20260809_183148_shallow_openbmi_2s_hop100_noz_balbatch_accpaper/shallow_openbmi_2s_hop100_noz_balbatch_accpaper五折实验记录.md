# 被试独立五折实验记录（20260809_183148 / shallow_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-09T18:31:48`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=2`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\shallow_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260809_183148`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 2, 'patience': 2, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6833 ± 0.0000`
- Test Acc_paper：`0.5000 ± 0.0000`
- Test BalAcc_maj：`0.4975 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.5069 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`2` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6833`
- Val BalAcc_maj（附报）：`0.5425`

**Test 试次级**
- Acc_paper：`0.5000`
- BalAcc_maj：`0.4975`
- Acc_majority：`0.5000`
- n_trials：`300`

**Test 窗级（附报）**
- BalAcc：`0.5069` | F1：`0.5762` | Acc：`0.5059`

### Three
- （本次跳过）

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100_noz",
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
  "protocol": "2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-09T18:31:57`
