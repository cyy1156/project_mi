# 被试独立五折实验记录（20260810_231638 / dbn_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-10T23:16:38`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dbn` | DBN + 2s μ/β log bandpower (N,8,2)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\dbn_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260810_231638`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6299 ± 0.0316`
- Test Acc_paper：`0.6513 ± 0.0255`
- Test BalAcc_maj：`0.5884 ± 0.0289`
- Test 窗级 BalAcc（附报）：`0.5859 ± 0.0256`

### Task 分折明细

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6141`
- Val BalAcc_maj（附报）：`0.5783`

**Test 试次级**
- Acc_paper：`0.6561`
- BalAcc_maj：`0.5820`
- Acc_majority：`0.6561`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5781` | F1：`0.7536` | Acc：`0.6517`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.6722`
- Val BalAcc_maj（附报）：`0.6006`

**Test 试次级**
- Acc_paper：`0.6948`
- BalAcc_maj：`0.6345`
- Acc_majority：`0.6948`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6297` | F1：`0.7745` | Acc：`0.6879`

#### Fold 2

- stopped_epoch：`62` | best_epoch：`42`
- Val Acc_paper（早停）：`0.5811`
- Val BalAcc_maj（附报）：`0.5339`

**Test 试次级**
- Acc_paper：`0.6327`
- BalAcc_maj：`0.5593`
- Acc_majority：`0.6327`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5641` | F1：`0.7359` | Acc：`0.6322`

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6537`
- Val BalAcc_maj（附报）：`0.5883`

**Test 试次级**
- Acc_paper：`0.6527`
- BalAcc_maj：`0.6068`
- Acc_majority：`0.6527`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5979` | F1：`0.7333` | Acc：`0.6436`

#### Fold 4

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6285`
- Val BalAcc_maj（附报）：`0.5342`

**Test 试次级**
- Acc_paper：`0.6200`
- BalAcc_maj：`0.5595`
- Acc_majority：`0.6200`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5598` | F1：`0.7206` | Acc：`0.6189`

### Three
- Val Acc_paper：`0.4530 ± 0.0377`
- Test Acc_paper：`0.4644 ± 0.0336`
- Test BalAcc_maj：`0.4743 ± 0.0329`
- Test 窗级 BalAcc（附报）：`0.4642 ± 0.0287`

### Three 分折明细

#### Fold 0

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.4515`
- Val BalAcc_maj（附报）：`0.4578`

**Test 试次级**
- Acc_paper：`0.4167`
- BalAcc_maj：`0.4255`
- F1-macro（众数）：`0.4135`
- Rec idle/left/right：`0.2682` / `0.3900` / `0.6182`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4232` | F1m：`0.4122`

#### Fold 1

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.4811`
- Val BalAcc_maj（附报）：`0.4904`

**Test 试次级**
- Acc_paper：`0.4967`
- BalAcc_maj：`0.5061`
- F1-macro（众数）：`0.4996`
- Rec idle/left/right：`0.4673` / `0.6682` / `0.3827`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4926` | F1m：`0.4870`

#### Fold 2

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.4026`
- Val BalAcc_maj（附报）：`0.4122`

**Test 试次级**
- Acc_paper：`0.4533`
- BalAcc_maj：`0.4630`
- F1-macro（众数）：`0.4548`
- Rec idle/left/right：`0.3564` / `0.6509` / `0.3818`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4518` | F1m：`0.4440`

#### Fold 3

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5067`
- Val BalAcc_maj（附报）：`0.5130`

**Test 试次级**
- Acc_paper：`0.5082`
- BalAcc_maj：`0.5158`
- F1-macro（众数）：`0.5124`
- Rec idle/left/right：`0.4009` / `0.6000` / `0.5464`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5008` | F1m：`0.4979`

#### Fold 4

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.4233`
- Val BalAcc_maj（附报）：`0.4263`

**Test 试次级**
- Acc_paper：`0.4473`
- BalAcc_maj：`0.4613`
- F1-macro（众数）：`0.4471`
- Rec idle/left/right：`0.2700` / `0.6700` / `0.4440`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4524` | F1m：`0.4397`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100_noz",
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
  "protocol": "2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-10T23:57:13`
