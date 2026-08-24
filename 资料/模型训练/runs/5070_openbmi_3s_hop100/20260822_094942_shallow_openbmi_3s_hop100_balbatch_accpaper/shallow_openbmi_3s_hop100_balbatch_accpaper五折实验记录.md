# 被试独立五折实验记录（20260822_094942 / shallow_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-22T09:49:42`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA GeForce RTX 5070 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）· Tw=3s hop=100ms · 实验21·5070
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\MI\code\train_lab\out\5070_baseline_openbmi_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260822_094942`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7319 ± 0.0295`
- Test Acc_paper：`0.7410 ± 0.0288`
- Test BalAcc_maj：`0.7179 ± 0.0220`
- Test 窗级 BalAcc（附报）：`0.7062 ± 0.0193`

### Task 分折明细

#### Fold 0

- stopped_epoch：`67` | best_epoch：`47`
- Val Acc_paper（早停）：`0.7315`
- Val BalAcc_maj（附报）：`0.7139`

**Test 试次级**
- Acc_paper：`0.7394`
- BalAcc_maj：`0.7130`
- Acc_majority：`0.7394`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7031` | F1：`0.7938` | Acc：`0.7293`

#### Fold 1

- stopped_epoch：`54` | best_epoch：`34`
- Val Acc_paper（早停）：`0.7611`
- Val BalAcc_maj（附报）：`0.7147`

**Test 试次级**
- Acc_paper：`0.7694`
- BalAcc_maj：`0.7568`
- Acc_majority：`0.7694`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7390` | F1：`0.8079` | Acc：`0.7527`

#### Fold 2

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.6900`
- Val BalAcc_maj（附报）：`0.6844`

**Test 试次级**
- Acc_paper：`0.7133`
- BalAcc_maj：`0.6948`
- Acc_majority：`0.7133`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6833` | F1：`0.7664` | Acc：`0.7009`

#### Fold 3

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.7670`
- Val BalAcc_maj（附报）：`0.7408`

**Test 试次级**
- Acc_paper：`0.7773`
- BalAcc_maj：`0.7243`
- Acc_majority：`0.7773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7137` | F1：`0.8289` | Acc：`0.7630`

#### Fold 4

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.7096`
- Val BalAcc_maj（附报）：`0.6739`

**Test 试次级**
- Acc_paper：`0.7057`
- BalAcc_maj：`0.7007`
- Acc_majority：`0.7057`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6919` | F1：`0.7569` | Acc：`0.6971`

### Three
- Val Acc_paper：`0.5702 ± 0.0278`
- Test Acc_paper：`0.5873 ± 0.0252`
- Test BalAcc_maj：`0.5910 ± 0.0248`
- Test 窗级 BalAcc（附报）：`0.5816 ± 0.0241`

### Three 分折明细

#### Fold 0

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.5819`
- Val BalAcc_maj（附报）：`0.5859`

**Test 试次级**
- Acc_paper：`0.5576`
- BalAcc_maj：`0.5618`
- F1-macro（众数）：`0.5612`
- Rec idle/left/right：`0.6127` / `0.5218` / `0.5509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5515` | F1m：`0.5509`

#### Fold 1

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.5781`
- Val BalAcc_maj（附报）：`0.5819`

**Test 试次级**
- Acc_paper：`0.6155`
- BalAcc_maj：`0.6191`
- F1-macro（众数）：`0.6192`
- Rec idle/left/right：`0.6573` / `0.6500` / `0.5500`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6099` | F1m：`0.6099`

#### Fold 2

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.5519`
- Val BalAcc_maj（附报）：`0.5552`

**Test 试次级**
- Acc_paper：`0.5685`
- BalAcc_maj：`0.5721`
- F1-macro（众数）：`0.5703`
- Rec idle/left/right：`0.6300` / `0.6127` / `0.4736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5638` | F1m：`0.5620`

#### Fold 3

- stopped_epoch：`98` | best_epoch：`78`
- Val Acc_paper（早停）：`0.6104`
- Val BalAcc_maj（附报）：`0.6152`

**Test 试次级**
- Acc_paper：`0.6191`
- BalAcc_maj：`0.6221`
- F1-macro（众数）：`0.6229`
- Rec idle/left/right：`0.6018` / `0.6109` / `0.6536`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6099` | F1m：`0.6104`

#### Fold 4

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.5289`
- Val BalAcc_maj（附报）：`0.5330`

**Test 试次级**
- Acc_paper：`0.5757`
- BalAcc_maj：`0.5800`
- F1-macro（众数）：`0.5778`
- Rec idle/left/right：`0.6810` / `0.5610` / `0.4980`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5730` | F1m：`0.5713`

### 共用超参
```json
{
  "data_tag": "openbmi_3s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 20,
  "batch_train": 256,
  "batch_eval": 512,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 750,
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

- 结束：`2026-08-22T12:13:43`
