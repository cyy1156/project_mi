# 被试独立五折实验记录（20260807_135828 / shallow_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-07T13:58:28`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow` | ShallowFBCSPNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_openbmi_2s_hop100_accpaper\shallow_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260807_135828`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6827 ± 0.0362`
- Test Acc_paper：`0.6941 ± 0.0349`
- Test BalAcc_maj：`0.6762 ± 0.0184`
- Test 窗级 BalAcc（附报）：`0.6510 ± 0.0150`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6763`
- Val BalAcc_maj（附报）：`0.6694`

**Test 试次级**
- Acc_paper：`0.6900`
- BalAcc_maj：`0.6807`
- Acc_majority：`0.6900`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6505` | F1：`0.7258` | Acc：`0.6592`

#### Fold 1

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.7352`
- Val BalAcc_maj（附报）：`0.6728`

**Test 试次级**
- Acc_paper：`0.7282`
- BalAcc_maj：`0.6923`
- Acc_majority：`0.7282`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6659` | F1：`0.7702` | Acc：`0.6975`

#### Fold 2

- stopped_epoch：`54` | best_epoch：`34`
- Val Acc_paper（早停）：`0.6415`
- Val BalAcc_maj（附报）：`0.6364`

**Test 试次级**
- Acc_paper：`0.6464`
- BalAcc_maj：`0.6509`
- Acc_majority：`0.6464`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6272` | F1：`0.6889` | Acc：`0.6254`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7122`
- Val BalAcc_maj（附报）：`0.7028`

**Test 试次级**
- Acc_paper：`0.7382`
- BalAcc_maj：`0.6982`
- Acc_majority：`0.7382`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6678` | F1：`0.7790` | Acc：`0.7051`

#### Fold 4

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.6485`
- Val BalAcc_maj（附报）：`0.6181`

**Test 试次级**
- Acc_paper：`0.6677`
- BalAcc_maj：`0.6590`
- Acc_majority：`0.6677`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6437` | F1：`0.7210` | Acc：`0.6531`

### Three
- Val Acc_paper：`0.5226 ± 0.0316`
- Test Acc_paper：`0.5404 ± 0.0256`
- Test BalAcc_maj：`0.5583 ± 0.0256`
- Test 窗级 BalAcc（附报）：`0.5300 ± 0.0221`

### Three 分折明细

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5293`
- Val BalAcc_maj（附报）：`0.5474`

**Test 试次级**
- Acc_paper：`0.5267`
- BalAcc_maj：`0.5464`
- F1-macro（众数）：`0.5452`
- Rec idle/left/right：`0.6009` / `0.5573` / `0.4809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5144` | F1m：`0.5137`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5433`
- Val BalAcc_maj（附报）：`0.5578`

**Test 试次级**
- Acc_paper：`0.5636`
- BalAcc_maj：`0.5788`
- F1-macro（众数）：`0.5793`
- Rec idle/left/right：`0.5918` / `0.6055` / `0.5391`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5502` | F1m：`0.5503`

#### Fold 2

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5026`
- Val BalAcc_maj（附报）：`0.5170`

**Test 试次级**
- Acc_paper：`0.5015`
- BalAcc_maj：`0.5176`
- F1-macro（众数）：`0.5147`
- Rec idle/left/right：`0.5609` / `0.5845` / `0.4073`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4983` | F1m：`0.4959`

#### Fold 3

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5641`
- Val BalAcc_maj（附报）：`0.5752`

**Test 试次级**
- Acc_paper：`0.5724`
- BalAcc_maj：`0.5909`
- F1-macro（众数）：`0.5882`
- Rec idle/left/right：`0.5191` / `0.5300` / `0.7236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5577` | F1m：`0.5554`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4737`
- Val BalAcc_maj（附报）：`0.4937`

**Test 试次级**
- Acc_paper：`0.5377`
- BalAcc_maj：`0.5580`
- F1-macro（众数）：`0.5581`
- Rec idle/left/right：`0.5900` / `0.5520` / `0.5320`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5292` | F1m：`0.5293`

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

- 结束：`2026-08-07T21:31:43`
