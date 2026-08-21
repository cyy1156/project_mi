# 被试独立五折实验记录（20260821_190504 / shallow_openbmi_3s_hop100_balbatch_accpaper）

- 开始：`2026-08-21T19:05:04`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100`（**仅 OpenBMI / 3s hop100**；blocks=EEG_MI_train）
- protocol：`3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=3s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）· Tw=3s hop=100ms · 实验20
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260821_190504`
- shared hp：`{'data_tag': 'openbmi_3s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 750, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7358 ± 0.0261`
- Test Acc_paper：`0.7415 ± 0.0306`
- Test BalAcc_maj：`0.7219 ± 0.0201`
- Test 窗级 BalAcc（附报）：`0.7078 ± 0.0180`

### Task 分折明细

#### Fold 0

- stopped_epoch：`65` | best_epoch：`45`
- Val Acc_paper（早停）：`0.7322`
- Val BalAcc_maj（附报）：`0.7169`

**Test 试次级**
- Acc_paper：`0.7324`
- BalAcc_maj：`0.7109`
- Acc_majority：`0.7324`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6965` | F1：`0.7834` | Acc：`0.7187`

#### Fold 1

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.7663`
- Val BalAcc_maj（附报）：`0.7178`

**Test 试次级**
- Acc_paper：`0.7833`
- BalAcc_maj：`0.7586`
- Acc_majority：`0.7833`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7402` | F1：`0.8225` | Acc：`0.7653`

#### Fold 2

- stopped_epoch：`74` | best_epoch：`54`
- Val Acc_paper（早停）：`0.7052`
- Val BalAcc_maj（附报）：`0.7103`

**Test 试次级**
- Acc_paper：`0.7100`
- BalAcc_maj：`0.7080`
- Acc_majority：`0.7100`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6943` | F1：`0.7538` | Acc：`0.6958`

#### Fold 3

- stopped_epoch：`60` | best_epoch：`40`
- Val Acc_paper（早停）：`0.7652`
- Val BalAcc_maj（附报）：`0.7414`

**Test 试次级**
- Acc_paper：`0.7712`
- BalAcc_maj：`0.7280`
- Acc_majority：`0.7712`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7150` | F1：`0.8218` | Acc：`0.7569`

#### Fold 4

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.7100`
- Val BalAcc_maj（附报）：`0.6822`

**Test 试次级**
- Acc_paper：`0.7103`
- BalAcc_maj：`0.7040`
- Acc_majority：`0.7103`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6933` | F1：`0.7598` | Acc：`0.6997`

### Three
- Val Acc_paper：`0.5687 ± 0.0299`
- Test Acc_paper：`0.5876 ± 0.0296`
- Test BalAcc_maj：`0.5919 ± 0.0301`
- Test 窗级 BalAcc（附报）：`0.5808 ± 0.0257`

### Three 分折明细

#### Fold 0

- stopped_epoch：`86` | best_epoch：`66`
- Val Acc_paper（早停）：`0.5856`
- Val BalAcc_maj（附报）：`0.5893`

**Test 试次级**
- Acc_paper：`0.5533`
- BalAcc_maj：`0.5579`
- F1-macro（众数）：`0.5567`
- Rec idle/left/right：`0.6282` / `0.5418` / `0.5036`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5516` | F1m：`0.5506`

#### Fold 1

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5830`
- Val BalAcc_maj（附报）：`0.5878`

**Test 试次级**
- Acc_paper：`0.6227`
- BalAcc_maj：`0.6264`
- F1-macro（众数）：`0.6271`
- Rec idle/left/right：`0.6327` / `0.6745` / `0.5718`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6071` | F1m：`0.6077`

#### Fold 2

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.5448`
- Val BalAcc_maj（附报）：`0.5478`

**Test 试次级**
- Acc_paper：`0.5652`
- BalAcc_maj：`0.5664`
- F1-macro（众数）：`0.5643`
- Rec idle/left/right：`0.6400` / `0.5900` / `0.4691`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5608` | F1m：`0.5588`

#### Fold 3

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.6063`
- Val BalAcc_maj（附报）：`0.6085`

**Test 试次级**
- Acc_paper：`0.6233`
- BalAcc_maj：`0.6291`
- F1-macro（众数）：`0.6297`
- Rec idle/left/right：`0.6082` / `0.6191` / `0.6600`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6158` | F1m：`0.6161`

#### Fold 4

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.5241`
- Val BalAcc_maj（附报）：`0.5267`

**Test 试次级**
- Acc_paper：`0.5737`
- BalAcc_maj：`0.5797`
- F1-macro（众数）：`0.5789`
- Rec idle/left/right：`0.6330` / `0.5830` / `0.5230`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5686` | F1m：`0.5680`

### 共用超参
```json
{
  "data_tag": "openbmi_3s_hop100",
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
  "protocol": "3s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-21T20:26:04`
