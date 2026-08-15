# 被试独立五折实验记录（20260813_180248 / l_l0e_eegnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-13T18:02:48`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`l_l0e_eegnet` | L0 fold0 EEGNet · Acc_paper | EEGNet F1=8 D=2 F2=16
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_ciacnet_mi_accpaper\l_l0e_eegnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260813_180248`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7011 ± 0.0000`
- Test Acc_paper：`0.7000 ± 0.0000`
- Test BalAcc_maj：`0.6630 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.6372 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`87` | best_epoch：`67`
- Val Acc_paper（早停）：`0.7011`
- Val BalAcc_maj（附报）：`0.6639`

**Test 试次级**
- Acc_paper：`0.7000`
- BalAcc_maj：`0.6630`
- Acc_majority：`0.7000`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6372` | F1：`0.7519` | Acc：`0.6728`

### Three
- Val Acc_paper：`0.5130 ± 0.0000`
- Test Acc_paper：`0.5070 ± 0.0000`
- Test BalAcc_maj：`0.5236 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.5074 ± 0.0000`

### Three 分折明细

#### Fold 0

- stopped_epoch：`96` | best_epoch：`76`
- Val Acc_paper（早停）：`0.5130`
- Val BalAcc_maj（附报）：`0.5230`

**Test 试次级**
- Acc_paper：`0.5070`
- BalAcc_maj：`0.5236`
- F1-macro（众数）：`0.5237`
- Rec idle/left/right：`0.5264` / `0.5082` / `0.5364`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5074` | F1m：`0.5074`

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

- 结束：`2026-08-13T19:04:31`
