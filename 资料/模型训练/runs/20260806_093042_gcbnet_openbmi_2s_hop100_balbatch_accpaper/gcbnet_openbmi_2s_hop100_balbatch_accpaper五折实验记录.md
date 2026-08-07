# 被试独立五折实验记录（20260806_093042 / gcbnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T09:30:42`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet` | GCBNet(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_openbmi_2s_hop100_accpaper\gcbnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_093042`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6157 ± 0.0268`
- Test Acc_paper：`0.6169 ± 0.0426`
- Test BalAcc_maj：`0.5624 ± 0.0149`
- Test 窗级 BalAcc（附报）：`0.5539 ± 0.0106`

### Task 分折明细

#### Fold 0

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5641`
- Val BalAcc_maj（附报）：`0.5503`

**Test 试次级**
- Acc_paper：`0.6430`
- BalAcc_maj：`0.5716`
- Acc_majority：`0.6430`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5574` | F1：`0.7288` | Acc：`0.6241`

#### Fold 1

- stopped_epoch：`72` | best_epoch：`52`
- Val Acc_paper（早停）：`0.6363`
- Val BalAcc_maj（附报）：`0.5619`

**Test 试次级**
- Acc_paper：`0.6585`
- BalAcc_maj：`0.5580`
- Acc_majority：`0.6585`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5547` | F1：`0.7512` | Acc：`0.6408`

#### Fold 2

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.6156`
- Val BalAcc_maj（附报）：`0.5556`

**Test 试次级**
- Acc_paper：`0.5367`
- BalAcc_maj：`0.5364`
- Acc_majority：`0.5367`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5338` | F1：`0.6129` | Acc：`0.5385`

#### Fold 3

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.6281`
- Val BalAcc_maj（附报）：`0.5731`

**Test 试次级**
- Acc_paper：`0.6321`
- BalAcc_maj：`0.5800`
- Acc_majority：`0.6321`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5645` | F1：`0.7071` | Acc：`0.6111`

#### Fold 4

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.6344`
- Val BalAcc_maj（附报）：`0.5472`

**Test 试次级**
- Acc_paper：`0.6143`
- BalAcc_maj：`0.5660`
- Acc_majority：`0.6143`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5591` | F1：`0.6985` | Acc：`0.6028`

### Three
- Val Acc_paper：`0.3773 ± 0.0131`
- Test Acc_paper：`0.3702 ± 0.0159`
- Test BalAcc_maj：`0.3899 ± 0.0147`
- Test 窗级 BalAcc（附报）：`0.3885 ± 0.0137`

### Three 分折明细

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.3648`
- Val BalAcc_maj（附报）：`0.3863`

**Test 试次级**
- Acc_paper：`0.3667`
- BalAcc_maj：`0.3842`
- F1-macro（众数）：`0.3692`
- Rec idle/left/right：`0.3018` / `0.6082` / `0.2427`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3848` | F1m：`0.3728`

#### Fold 1

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.3956`
- Val BalAcc_maj（附报）：`0.4222`

**Test 试次级**
- Acc_paper：`0.3770`
- BalAcc_maj：`0.4015`
- F1-macro（众数）：`0.3908`
- Rec idle/left/right：`0.2536` / `0.5873` / `0.3636`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3988` | F1m：`0.3900`

#### Fold 2

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.3719`
- Val BalAcc_maj（附报）：`0.3904`

**Test 试次级**
- Acc_paper：`0.3421`
- BalAcc_maj：`0.3661`
- F1-macro（众数）：`0.3621`
- Rec idle/left/right：`0.4664` / `0.3518` / `0.2800`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3655` | F1m：`0.3629`

#### Fold 3

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.3904`
- Val BalAcc_maj（附报）：`0.4067`

**Test 试次级**
- Acc_paper：`0.3897`
- BalAcc_maj：`0.4085`
- F1-macro（众数）：`0.4004`
- Rec idle/left/right：`0.3436` / `0.5791` / `0.3027`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4055` | F1m：`0.4002`

#### Fold 4

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3641`
- Val BalAcc_maj（附报）：`0.3859`

**Test 试次级**
- Acc_paper：`0.3757`
- BalAcc_maj：`0.3893`
- F1-macro（众数）：`0.3856`
- Rec idle/left/right：`0.4930` / `0.3020` / `0.3730`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.3880` | F1m：`0.3857`

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

- 结束：`2026-08-06T11:17:22`
