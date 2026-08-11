# 被试独立五折实验记录（20260811_104849 / shallow_s0_net_enhance_three）

- 开始：`2026-08-11T10:48:49`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\挑战杯专属\project_mi\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_s0` | ShallowFBCSPNet（braindecode 默认；S0 复现锚点）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\挑战杯专属\project_mi\code\train_lab\out\5060_shallow_net_enhance_three_accpaper\shallow_s0_net_enhance_three\openbmi_2s_hop100\run_20260811_104849`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6828 ± 0.0365`
- Test Acc_paper：`0.6969 ± 0.0350`
- Test BalAcc_maj：`0.6765 ± 0.0187`
- Test 窗级 BalAcc（附报）：`0.6517 ± 0.0151`

### Task 分折明细

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.6774`
- Val BalAcc_maj（附报）：`0.6639`

**Test 试次级**
- Acc_paper：`0.7048`
- BalAcc_maj：`0.6839`
- Acc_majority：`0.7048`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6543` | F1：`0.7434` | Acc：`0.6730`

#### Fold 1

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.7356`
- Val BalAcc_maj（附报）：`0.6733`

**Test 试次级**
- Acc_paper：`0.7282`
- BalAcc_maj：`0.6925`
- Acc_majority：`0.7282`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6659` | F1：`0.7702` | Acc：`0.6974`

#### Fold 2

- stopped_epoch：`54` | best_epoch：`34`
- Val Acc_paper（早停）：`0.6407`
- Val BalAcc_maj（附报）：`0.6342`

**Test 试次级**
- Acc_paper：`0.6461`
- BalAcc_maj：`0.6502`
- Acc_majority：`0.6461`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6270` | F1：`0.6896` | Acc：`0.6258`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7122`
- Val BalAcc_maj（附报）：`0.7025`

**Test 试次级**
- Acc_paper：`0.7376`
- BalAcc_maj：`0.6973`
- Acc_majority：`0.7376`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6678` | F1：`0.7791` | Acc：`0.7052`

#### Fold 4

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.6481`
- Val BalAcc_maj（附报）：`0.6175`

**Test 试次级**
- Acc_paper：`0.6677`
- BalAcc_maj：`0.6587`
- Acc_majority：`0.6677`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6435` | F1：`0.7210` | Acc：`0.6531`

### Three
- Val Acc_paper：`0.5226 ± 0.0317`
- Test Acc_paper：`0.5401 ± 0.0257`
- Test BalAcc_maj：`0.5583 ± 0.0257`
- Test 窗级 BalAcc（附报）：`0.5300 ± 0.0221`

### Three 分折明细

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5293`
- Val BalAcc_maj（附报）：`0.5474`

**Test 试次级**
- Acc_paper：`0.5261`
- BalAcc_maj：`0.5458`
- F1-macro（众数）：`0.5446`
- Rec idle/left/right：`0.6000` / `0.5573` / `0.4800`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5142` | F1m：`0.5136`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5437`
- Val BalAcc_maj（附报）：`0.5581`

**Test 试次级**
- Acc_paper：`0.5636`
- BalAcc_maj：`0.5785`
- F1-macro（众数）：`0.5790`
- Rec idle/left/right：`0.5918` / `0.6055` / `0.5382`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5503` | F1m：`0.5504`

#### Fold 2

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5022`
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
- Val BalAcc_maj（附报）：`0.5756`

**Test 试次级**
- Acc_paper：`0.5724`
- BalAcc_maj：`0.5912`
- F1-macro（众数）：`0.5886`
- Rec idle/left/right：`0.5191` / `0.5309` / `0.7236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5577` | F1m：`0.5554`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4737`
- Val BalAcc_maj（附报）：`0.4941`

**Test 试次级**
- Acc_paper：`0.5370`
- BalAcc_maj：`0.5583`
- F1-macro（众数）：`0.5584`
- Rec idle/left/right：`0.5900` / `0.5530` / `0.5320`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5293` | F1m：`0.5293`

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

- 结束：`2026-08-11T12:52:53`
