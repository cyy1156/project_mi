# 被试独立五折实验记录（20260813_190441 / l_l0c_ciacnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-13T19:04:41`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`l_l0c_ciacnet` | L0 fold0 CIACNet · Acc_paper | CIACNet full (CV1+CV2+IAT+TC)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_ciacnet_mi_accpaper\l_l0c_ciacnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260813_190441`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7337 ± 0.0000`
- Test Acc_paper：`0.7270 ± 0.0000`
- Test BalAcc_maj：`0.6811 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.6583 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.7337`
- Val BalAcc_maj（附报）：`0.6803`

**Test 试次级**
- Acc_paper：`0.7270`
- BalAcc_maj：`0.6811`
- Acc_majority：`0.7270`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6583` | F1：`0.7768` | Acc：`0.7000`

### Three
- Val Acc_paper：`0.5356 ± 0.0000`
- Test Acc_paper：`0.5221 ± 0.0000`
- Test BalAcc_maj：`0.5367 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.5197 ± 0.0000`

### Three 分折明细

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5356`
- Val BalAcc_maj（附报）：`0.5485`

**Test 试次级**
- Acc_paper：`0.5221`
- BalAcc_maj：`0.5367`
- F1-macro（众数）：`0.5367`
- Rec idle/left/right：`0.4955` / `0.5664` / `0.5482`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5197` | F1m：`0.5197`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-13T20:17:38`
