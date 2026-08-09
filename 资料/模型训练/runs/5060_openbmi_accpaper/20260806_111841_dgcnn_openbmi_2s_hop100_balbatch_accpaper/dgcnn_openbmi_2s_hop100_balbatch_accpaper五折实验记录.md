# 被试独立五折实验记录（20260806_111841 / dgcnn_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T11:18:41`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dgcnn` | DGCNN(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\dgcnn_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_111841`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.5885 ± 0.0409`
- Test Acc_paper：`0.5943 ± 0.0480`
- Test BalAcc_maj：`0.5659 ± 0.0181`
- Test 窗级 BalAcc（附报）：`0.5588 ± 0.0147`

### Task 分折明细

#### Fold 0

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5230`
- Val BalAcc_maj（附报）：`0.5397`

**Test 试次级**
- Acc_paper：`0.6152`
- BalAcc_maj：`0.5618`
- Acc_majority：`0.6152`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5541` | F1：`0.7026` | Acc：`0.6035`

#### Fold 1

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.5848`
- Val BalAcc_maj（附报）：`0.5919`

**Test 试次级**
- Acc_paper：`0.6564`
- BalAcc_maj：`0.6016`
- Acc_majority：`0.6564`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5878` | F1：`0.7333` | Acc：`0.6396`

#### Fold 2

- stopped_epoch：`72` | best_epoch：`52`
- Val Acc_paper（早停）：`0.6089`
- Val BalAcc_maj（附报）：`0.5700`

**Test 试次级**
- Acc_paper：`0.5255`
- BalAcc_maj：`0.5525`
- Acc_majority：`0.5255`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5472` | F1：`0.5760` | Acc：`0.5258`

#### Fold 3

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.6478`
- Val BalAcc_maj（附报）：`0.5792`

**Test 试次级**
- Acc_paper：`0.6218`
- BalAcc_maj：`0.5568`
- Acc_majority：`0.6218`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5524` | F1：`0.7155` | Acc：`0.6121`

#### Fold 4

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5781`
- Val BalAcc_maj（附报）：`0.5578`

**Test 试次级**
- Acc_paper：`0.5527`
- BalAcc_maj：`0.5570`
- Acc_majority：`0.5527`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5525` | F1：`0.6171` | Acc：`0.5498`

### Three
- Val Acc_paper：`0.3859 ± 0.0203`
- Test Acc_paper：`0.3916 ± 0.0152`
- Test BalAcc_maj：`0.4099 ± 0.0151`
- Test 窗级 BalAcc（附报）：`0.3999 ± 0.0114`

### Three 分折明细

#### Fold 0

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.3637`
- Val BalAcc_maj（附报）：`0.3789`

**Test 试次级**
- Acc_paper：`0.3815`
- BalAcc_maj：`0.3970`
- F1-macro（众数）：`0.3877`
- Rec idle/left/right：`0.2636` / `0.5627` / `0.3645`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3915` | F1m：`0.3833`

#### Fold 1

- stopped_epoch：`73` | best_epoch：`53`
- Val Acc_paper（早停）：`0.4122`
- Val BalAcc_maj（附报）：`0.4378`

**Test 试次级**
- Acc_paper：`0.4109`
- BalAcc_maj：`0.4291`
- F1-macro（众数）：`0.4134`
- Rec idle/left/right：`0.3391` / `0.6800` / `0.2682`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4158` | F1m：`0.4030`

#### Fold 2

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.3774`
- Val BalAcc_maj（附报）：`0.3959`

**Test 试次级**
- Acc_paper：`0.3682`
- BalAcc_maj：`0.3882`
- F1-macro（众数）：`0.3804`
- Rec idle/left/right：`0.5200` / `0.3900` / `0.2545`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3831` | F1m：`0.3779`

#### Fold 3

- stopped_epoch：`91` | best_epoch：`71`
- Val Acc_paper（早停）：`0.4081`
- Val BalAcc_maj（附报）：`0.4244`

**Test 试次级**
- Acc_paper：`0.3952`
- BalAcc_maj：`0.4155`
- F1-macro（众数）：`0.4123`
- Rec idle/left/right：`0.3527` / `0.5236` / `0.3700`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4043` | F1m：`0.4018`

#### Fold 4

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.3681`
- Val BalAcc_maj（附报）：`0.3878`

**Test 试次级**
- Acc_paper：`0.4023`
- BalAcc_maj：`0.4197`
- F1-macro（众数）：`0.4194`
- Rec idle/left/right：`0.4430` / `0.4430` / `0.3730`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4047` | F1m：`0.4043`

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

- 结束：`2026-08-06T13:19:49`
