# 被试独立五折实验记录（20260823_095327 / shallow_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-23T09:53:27`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet · Tw=3s hop=100ms · 方案24·5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_095327`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
## 最终结论（主报 Acc_paper）

### Task
- （本次跳过 · three-only）

### Three
- Val Acc_paper：`0.5719 ± 0.0298`
- Test Acc_paper：`0.5839 ± 0.0268`
- Test BalAcc_maj：`0.5869 ± 0.0269`
- Test 窗级 BalAcc（附报）：`0.5769 ± 0.0230`

### Three 分折明细

#### Fold 0

- stopped_epoch：`50` | best_epoch：`30`
- Val Acc_paper（早停）：`0.5859`
- Val BalAcc_maj（附报）：`0.5878`

**Test 试次级**
- Acc_paper：`0.5579`
- BalAcc_maj：`0.5612`
- F1-macro（众数）：`0.5592`
- Rec idle/left/right：`0.6455` / `0.5591` / `0.4791`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5516` | F1m：`0.5502`

#### Fold 1

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.5796`
- Val BalAcc_maj（附报）：`0.5826`

**Test 试次级**
- Acc_paper：`0.6197`
- BalAcc_maj：`0.6224`
- F1-macro（众数）：`0.6233`
- Rec idle/left/right：`0.6518` / `0.6309` / `0.5845`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6065` | F1m：`0.6072`

#### Fold 2

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5422`
- Val BalAcc_maj（附报）：`0.5452`

**Test 试次级**
- Acc_paper：`0.5545`
- BalAcc_maj：`0.5567`
- F1-macro（众数）：`0.5510`
- Rec idle/left/right：`0.6800` / `0.5827` / `0.4073`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5527` | F1m：`0.5476`

#### Fold 3

- stopped_epoch：`77` | best_epoch：`57`
- Val Acc_paper（早停）：`0.6163`
- Val BalAcc_maj（附报）：`0.6178`

**Test 试次级**
- Acc_paper：`0.6106`
- BalAcc_maj：`0.6142`
- F1-macro（众数）：`0.6127`
- Rec idle/left/right：`0.5545` / `0.5382` / `0.7500`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5998` | F1m：`0.5983`

#### Fold 4

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.5356`
- Val BalAcc_maj（附报）：`0.5378`

**Test 试次级**
- Acc_paper：`0.5767`
- BalAcc_maj：`0.5800`
- F1-macro（众数）：`0.5779`
- Rec idle/left/right：`0.6860` / `0.5310` / `0.5230`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5737` | F1m：`0.5719`

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

- 结束：`2026-08-23T11:56:26`
# 被试独立五折实验记录（20260823_095327 / shallow_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-28T10:33:49`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet · Tw=3s hop=100ms · 方案24·5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_095327`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
# 被试独立五折实验记录（20260823_095327 / shallow_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-28T10:56:00`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5090**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet · Tw=3s hop=100ms · 方案24·5090
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_095327`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN device=5090 scheme24', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True, 't0_weight_alpha': 0.0, 't0_filter_max': 0.0}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7300 ± 0.0300`
- Test Acc_paper：`0.7424 ± 0.0318`
- Test BalAcc_maj：`0.7181 ± 0.0227`
- Test 窗级 BalAcc（附报）：`0.7065 ± 0.0206`

### Task 分折明细

#### Fold 0

- stopped_epoch：`29` | best_epoch：`29`
- Val Acc_paper（早停）：`0.7222`
- Val BalAcc_maj（附报）：`0.7097`

**Test 试次级**
- Acc_paper：`0.7391`
- BalAcc_maj：`0.7189`
- Acc_majority：`0.7391`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7073` | F1：`0.7896` | Acc：`0.7274`

#### Fold 1

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.7641`
- Val BalAcc_maj（附报）：`0.7183`

**Test 试次级**
- Acc_paper：`0.7794`
- BalAcc_maj：`0.7557`
- Acc_majority：`0.7794`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7395` | F1：`0.8208` | Acc：`0.7637`

#### Fold 2

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.6922`
- Val BalAcc_maj（附报）：`0.6800`

**Test 试次级**
- Acc_paper：`0.7103`
- BalAcc_maj：`0.6927`
- Acc_majority：`0.7103`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6800` | F1：`0.7634` | Acc：`0.6974`

#### Fold 3

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.7656`
- Val BalAcc_maj（附报）：`0.7408`

**Test 试次级**
- Acc_paper：`0.7779`
- BalAcc_maj：`0.7264`
- Acc_majority：`0.7779`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7149` | F1：`0.8290` | Acc：`0.7634`

#### Fold 4

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.7059`
- Val BalAcc_maj（附报）：`0.6706`

**Test 试次级**
- Acc_paper：`0.7053`
- BalAcc_maj：`0.6970`
- Acc_majority：`0.7053`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6905` | F1：`0.7605` | Acc：`0.6992`

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

- 结束：`2026-08-28T12:44:11`
