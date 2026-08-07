# 被试独立五折实验记录（20260806_132107 / gcbnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T13:21:07`
- device：`cuda`
- 训练设备：**NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet` | GCBNet(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_baseline_openbmi_2s_hop100_accpaper\gcbnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_132107`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 1, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.5078 ± 0.0000`
- Test Acc_paper：`0.6115 ± 0.0000`
- Test BalAcc_maj：`0.5550 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.5501 ± 0.0000`

### Task 分折明细

#### Fold 0

- stopped_epoch：`1` | best_epoch：`1`
- Val Acc_paper（早停）：`0.5078`
- Val BalAcc_maj（附报）：`0.5244`

**Test 试次级**
- Acc_paper：`0.6115`
- BalAcc_maj：`0.5550`
- Acc_majority：`0.6115`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5501` | F1：`0.7098` | Acc：`0.6071`

### Three
- Val Acc_paper：`0.3304 ± 0.0000`
- Test Acc_paper：`0.3339 ± 0.0000`
- Test BalAcc_maj：`0.3385 ± 0.0000`
- Test 窗级 BalAcc（附报）：`0.3419 ± 0.0000`

### Three 分折明细

#### Fold 0

- stopped_epoch：`1` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3304`
- Val BalAcc_maj（附报）：`0.3363`

**Test 试次级**
- Acc_paper：`0.3339`
- BalAcc_maj：`0.3385`
- F1-macro（众数）：`0.2920`
- Rec idle/left/right：`0.4527` / `0.5191` / `0.0436`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3419` | F1m：`0.3039`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 1,
  "patience": 20,
  "batch_train": 128,
  "batch_eval": 256,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "non_blocking": true,
  "use_amp": true,
  "cudnn_benchmark": false,
  "gpu_memory_fraction": 1
}
```

- 结束：`2026-08-06T13:22:21`
