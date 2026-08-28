# 被试独立五折实验记录（20260823_135213 / conformer_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-23T13:52:13`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer layers=2, heads=10 · 3s · 5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
# 被试独立五折实验记录（20260823_135213 / conformer_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-23T14:37:33`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer layers=2, heads=10 · 3s · 5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
# 被试独立五折实验记录（20260823_135213 / conformer_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-23T14:38:34`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer layers=2, heads=10 · 3s · 5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
# 被试独立五折实验记录（20260823_135213 / conformer_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-23T14:41:05`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer layers=2, heads=10 · 3s · 5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
# 被试独立五折实验记录（20260823_135213 / conformer_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-28T19:19:54`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer layers=2, heads=10 · 3s · 5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7518 ± 0.0127`
- Test Acc_paper：`0.7597 ± 0.0310`
- Test BalAcc_maj：`0.7067 ± 0.0375`
- Test 窗级 BalAcc（附报）：`0.6959 ± 0.0308`

### Task 分折明细

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.7459`
- Val BalAcc_maj（附报）：`0.7003`

**Test 试次级**
- Acc_paper：`0.7364`
- BalAcc_maj：`0.6714`
- Acc_majority：`0.7364`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6681` | F1：`0.8098` | Acc：`0.7313`

#### Fold 1

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.7715`
- Val BalAcc_maj（附报）：`0.6969`

**Test 试次级**
- Acc_paper：`0.8091`
- BalAcc_maj：`0.7639`
- Acc_majority：`0.8091`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7415` | F1：`0.8460` | Acc：`0.7869`

#### Fold 2

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.7363`
- Val BalAcc_maj（附报）：`0.6850`

**Test 试次级**
- Acc_paper：`0.7291`
- BalAcc_maj：`0.6730`
- Acc_majority：`0.7291`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6681` | F1：`0.7995` | Acc：`0.7223`

#### Fold 3

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.7611`
- Val BalAcc_maj（附报）：`0.7172`

**Test 试次级**
- Acc_paper：`0.7827`
- BalAcc_maj：`0.7382`
- Acc_majority：`0.7827`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7240` | F1：`0.8306` | Acc：`0.7676`

#### Fold 4

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.7441`
- Val BalAcc_maj（附报）：`0.6678`

**Test 试次级**
- Acc_paper：`0.7410`
- BalAcc_maj：`0.6873`
- Acc_majority：`0.7410`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6778` | F1：`0.8052` | Acc：`0.7305`

### Three
- （本次跳过）

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
  "protocol": "3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 750,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": false,
  "persistent_workers": false,
  "prefetch_factor": 2,
  "non_blocking": true,
  "torch_num_threads": 8,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true,
  "t0_weight_alpha": 0.0,
  "t0_filter_max": 0.0
}
```

- 结束：`2026-08-28T22:01:05`
