# 被试独立五折实验记录（20260817_133104 / shallow_A0_ref_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-17T13:31:04`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_A0_ref` | A0-ref braindecode Shallow · 500pt · Three-only · Acc_paper（量级参考 · 5060）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_mask_future_dual_expert_accpaper\shallow_A0_ref_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260817_133104`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': False, 'torch_num_threads': 2, 'cudnn_benchmark': False, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- （本次跳过 · three-only）

### Three
- Val Acc_paper：`0.5222 ± 0.0313`
- Test Acc_paper：`0.5403 ± 0.0256`
- Test BalAcc_maj：`0.5583 ± 0.0258`
- Test 窗级 BalAcc（附报）：`0.5300 ± 0.0221`

### Three 分折明细

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5285`
- Val BalAcc_maj（附报）：`0.5463`

**Test 试次级**
- Acc_paper：`0.5264`
- BalAcc_maj：`0.5467`
- F1-macro（众数）：`0.5455`
- Rec idle/left/right：`0.6009` / `0.5582` / `0.4809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5144` | F1m：`0.5138`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5578`

**Test 试次级**
- Acc_paper：`0.5636`
- BalAcc_maj：`0.5788`
- F1-macro（众数）：`0.5793`
- Rec idle/left/right：`0.5918` / `0.6055` / `0.5391`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5503` | F1m：`0.5504`

#### Fold 2

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5026`
- Val BalAcc_maj（附报）：`0.5170`

**Test 试次级**
- Acc_paper：`0.5015`
- BalAcc_maj：`0.5173`
- F1-macro（众数）：`0.5144`
- Rec idle/left/right：`0.5609` / `0.5836` / `0.4073`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4983` | F1m：`0.4959`

#### Fold 3

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5633`
- Val BalAcc_maj（附报）：`0.5752`

**Test 试次级**
- Acc_paper：`0.5724`
- BalAcc_maj：`0.5912`
- F1-macro（众数）：`0.5886`
- Rec idle/left/right：`0.5191` / `0.5309` / `0.7236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5578` | F1m：`0.5555`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4737`
- Val BalAcc_maj（附报）：`0.4937`

**Test 试次级**
- Acc_paper：`0.5373`
- BalAcc_maj：`0.5577`
- F1-macro（众数）：`0.5577`
- Rec idle/left/right：`0.5900` / `0.5510` / `0.5320`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5292` | F1m：`0.5292`

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
  "num_workers": 0,
  "pin_memory": false,
  "persistent_workers": false,
  "prefetch_factor": 2,
  "non_blocking": false,
  "torch_num_threads": 2,
  "cudnn_benchmark": false,
  "deterministic": false,
  "use_amp": true
}
```

- 结束：`2026-08-17T14:14:22`
