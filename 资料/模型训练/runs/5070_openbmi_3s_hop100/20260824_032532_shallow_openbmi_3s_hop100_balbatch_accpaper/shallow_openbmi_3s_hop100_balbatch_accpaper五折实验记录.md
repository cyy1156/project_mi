# 被试独立五折实验记录（20260824_032532 / shallow_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-24T03:25:32`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA GeForce RTX 5070 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070 scheme25_aug` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet · Tw=3s hop=100ms · 方案25 G1 域增广 (post-zscore · train_device=5070)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\MI\code\train_lab\out\5070_aug_3s_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260824_032532`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070 scheme25_aug', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7284 ± 0.0334`
- Test Acc_paper：`0.7409 ± 0.0284`
- Test BalAcc_maj：`0.7162 ± 0.0196`
- Test 窗级 BalAcc（附报）：`0.7059 ± 0.0177`

### Task 分折明细

#### Fold 0

- stopped_epoch：`74` | best_epoch：`54`
- Val Acc_paper（早停）：`0.7293`
- Val BalAcc_maj（附报）：`0.7125`

**Test 试次级**
- Acc_paper：`0.7409`
- BalAcc_maj：`0.7159`
- Acc_majority：`0.7409`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7069` | F1：`0.7939` | Acc：`0.7307`

#### Fold 1

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.7607`
- Val BalAcc_maj（附报）：`0.7103`

**Test 试次级**
- Acc_paper：`0.7727`
- BalAcc_maj：`0.7484`
- Acc_majority：`0.7727`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7336` | F1：`0.8164` | Acc：`0.7580`

#### Fold 2

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.6763`
- Val BalAcc_maj（附报）：`0.6811`

**Test 试次级**
- Acc_paper：`0.7133`
- BalAcc_maj：`0.6980`
- Acc_majority：`0.7133`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6867` | F1：`0.7654` | Acc：`0.7014`

#### Fold 3

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.7663`
- Val BalAcc_maj（附报）：`0.7450`

**Test 试次级**
- Acc_paper：`0.7724`
- BalAcc_maj：`0.7243`
- Acc_majority：`0.7724`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7149` | F1：`0.8251` | Acc：`0.7598`

#### Fold 4

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.7093`
- Val BalAcc_maj（附报）：`0.6708`

**Test 试次级**
- Acc_paper：`0.7053`
- BalAcc_maj：`0.6943`
- Acc_majority：`0.7053`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6872` | F1：`0.7601` | Acc：`0.6976`

### Three
- Val Acc_paper：`0.5689 ± 0.0281`
- Test Acc_paper：`0.5889 ± 0.0266`
- Test BalAcc_maj：`0.5962 ± 0.0264`
- Test 窗级 BalAcc（附报）：`0.5816 ± 0.0241`

### Three 分折明细

#### Fold 0

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.5807`
- Val BalAcc_maj（附报）：`0.5870`

**Test 试次级**
- Acc_paper：`0.5558`
- BalAcc_maj：`0.5627`
- F1-macro（众数）：`0.5619`
- Rec idle/left/right：`0.6264` / `0.5145` / `0.5473`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5508` | F1m：`0.5501`

#### Fold 1

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5756`
- Val BalAcc_maj（附报）：`0.5819`

**Test 试次级**
- Acc_paper：`0.6203`
- BalAcc_maj：`0.6245`
- F1-macro（众数）：`0.6258`
- Rec idle/left/right：`0.6345` / `0.6482` / `0.5909`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6108` | F1m：`0.6120`

#### Fold 2

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.5544`
- Val BalAcc_maj（附报）：`0.5600`

**Test 试次级**
- Acc_paper：`0.5697`
- BalAcc_maj：`0.5761`
- F1-macro（众数）：`0.5742`
- Rec idle/left/right：`0.6345` / `0.6164` / `0.4773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5649` | F1m：`0.5630`

#### Fold 3

- stopped_epoch：`68` | best_epoch：`48`
- Val Acc_paper（早停）：`0.6089`
- Val BalAcc_maj（附报）：`0.6119`

**Test 试次级**
- Acc_paper：`0.6203`
- BalAcc_maj：`0.6294`
- F1-macro（众数）：`0.6273`
- Rec idle/left/right：`0.6064` / `0.5255` / `0.7564`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6089` | F1m：`0.6071`

#### Fold 4

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.5248`
- Val BalAcc_maj（附报）：`0.5274`

**Test 试次级**
- Acc_paper：`0.5787`
- BalAcc_maj：`0.5883`
- F1-macro（众数）：`0.5867`
- Rec idle/left/right：`0.6760` / `0.5710` / `0.5180`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5725` | F1m：`0.5713`

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
  "protocol": "3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5070 scheme25_aug",
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

- 结束：`2026-08-24T06:28:11`
