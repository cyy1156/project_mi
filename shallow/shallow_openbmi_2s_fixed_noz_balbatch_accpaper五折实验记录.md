# 被试独立五折实验记录（20260809_192352 / shallow_openbmi_2s_fixed_noz_balbatch_accpaper）

- 开始：`2026-08-09T19:23:52`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_fixed_cue2to4_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_fixed_noz_accpaper\shallow_openbmi_2s_fixed_noz_balbatch_accpaper\openbmi_2s_fixed_cue2to4_noz\run_20260809_192352`
- shared hp：`{'data_tag': 'openbmi_2s_fixed_cue2to4_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6179 ± 0.0385`
- Test Acc_paper：`0.6271 ± 0.0141`
- Test BalAcc_maj：`0.5898 ± 0.0231`
- Test 窗级 BalAcc（附报）：`0.5898 ± 0.0231`

### Task 分折明细

#### Fold 0

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.6404`
- Val BalAcc_maj（附报）：`0.5936`

**Test 试次级**
- Acc_paper：`0.6130`
- BalAcc_maj：`0.5986`
- Acc_majority：`0.6130`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5986` | F1：`0.6886` | Acc：`0.6130`

#### Fold 1

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.6659`
- Val BalAcc_maj（附报）：`0.6083`

**Test 试次级**
- Acc_paper：`0.6506`
- BalAcc_maj：`0.6205`
- Acc_majority：`0.6506`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6205` | F1：`0.7307` | Acc：`0.6506`

#### Fold 2

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.5619`
- Val BalAcc_maj（附报）：`0.5503`

**Test 试次级**
- Acc_paper：`0.6288`
- BalAcc_maj：`0.5716`
- Acc_majority：`0.6288`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5716` | F1：`0.7275` | Acc：`0.6288`

#### Fold 3

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.6367`
- Val BalAcc_maj（附报）：`0.6056`

**Test 试次级**
- Acc_paper：`0.6312`
- BalAcc_maj：`0.6025`
- Acc_majority：`0.6312`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6025` | F1：`0.7134` | Acc：`0.6312`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5848`
- Val BalAcc_maj（附报）：`0.5108`

**Test 试次级**
- Acc_paper：`0.6120`
- BalAcc_maj：`0.5557`
- Acc_majority：`0.6120`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5557` | F1：`0.7134` | Acc：`0.6120`

### Three
- Val Acc_paper：`0.4679 ± 0.0431`
- Test Acc_paper：`0.4812 ± 0.0192`
- Test BalAcc_maj：`0.4812 ± 0.0192`
- Test 窗级 BalAcc（附报）：`0.4812 ± 0.0192`

### Three 分折明细

#### Fold 0

- stopped_epoch：`85` | best_epoch：`65`
- Val Acc_paper（早停）：`0.4830`
- Val BalAcc_maj（附报）：`0.4830`

**Test 试次级**
- Acc_paper：`0.4727`
- BalAcc_maj：`0.4727`
- F1-macro（众数）：`0.4725`
- Rec idle/left/right：`0.5027` / `0.4791` / `0.4364`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4727` | F1m：`0.4725`

#### Fold 1

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.5015`
- Val BalAcc_maj（附报）：`0.5015`

**Test 试次级**
- Acc_paper：`0.5003`
- BalAcc_maj：`0.5003`
- F1-macro（众数）：`0.4992`
- Rec idle/left/right：`0.4364` / `0.5864` / `0.4782`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5003` | F1m：`0.4992`

#### Fold 2

- stopped_epoch：`78` | best_epoch：`58`
- Val Acc_paper（早停）：`0.4374`
- Val BalAcc_maj（附报）：`0.4374`

**Test 试次级**
- Acc_paper：`0.4506`
- BalAcc_maj：`0.4506`
- F1-macro（众数）：`0.4455`
- Rec idle/left/right：`0.3436` / `0.5864` / `0.4218`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4506` | F1m：`0.4455`

#### Fold 3

- stopped_epoch：`62` | best_epoch：`42`
- Val Acc_paper（早停）：`0.5174`
- Val BalAcc_maj（附报）：`0.5174`

**Test 试次级**
- Acc_paper：`0.5027`
- BalAcc_maj：`0.5027`
- F1-macro（众数）：`0.5021`
- Rec idle/left/right：`0.4700` / `0.4827` / `0.5555`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5027` | F1m：`0.5021`

#### Fold 4

- stopped_epoch：`79` | best_epoch：`59`
- Val Acc_paper（早停）：`0.4004`
- Val BalAcc_maj（附报）：`0.4004`

**Test 试次级**
- Acc_paper：`0.4797`
- BalAcc_maj：`0.4797`
- F1-macro（众数）：`0.4788`
- Rec idle/left/right：`0.4440` / `0.5570` / `0.4380`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4797` | F1m：`0.4788`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_fixed_cue2to4_noz",
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
  "protocol": "2s-fixed-cue2to4-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 4,
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

- 结束：`2026-08-09T21:41:48`
