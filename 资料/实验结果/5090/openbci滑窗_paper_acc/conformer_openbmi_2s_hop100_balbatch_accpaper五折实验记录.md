# 被试独立五折实验记录（20260806_003518 / conformer_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T00:35:18`
- device：`cuda`
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\baseline_openbmi_2s_hop100_accpaper\conformer_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_003518`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7104 ± 0.0302`
- Test Acc_paper：`0.7102 ± 0.0237`
- Test BalAcc_maj：`0.6496 ± 0.0232`
- Test 窗级 BalAcc（附报）：`0.6335 ± 0.0189`

### Task 分折明细

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.7222`
- Val BalAcc_maj（附报）：`0.6467`

**Test 试次级**
- Acc_paper：`0.7052`
- BalAcc_maj：`0.6205`
- Acc_majority：`0.7052`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6204` | F1：`0.7903` | Acc：`0.6981`

#### Fold 1

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7337`
- Val BalAcc_maj（附报）：`0.6589`

**Test 试次级**
- Acc_paper：`0.7373`
- BalAcc_maj：`0.6895`
- Acc_majority：`0.7373`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6674` | F1：`0.7880` | Acc：`0.7122`

#### Fold 2

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.6556`
- Val BalAcc_maj（附报）：`0.6231`

**Test 试次级**
- Acc_paper：`0.6764`
- BalAcc_maj：`0.6380`
- Acc_majority：`0.6764`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6142` | F1：`0.7359` | Acc：`0.6519`

#### Fold 3

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.7385`
- Val BalAcc_maj（附报）：`0.6775`

**Test 试次级**
- Acc_paper：`0.7367`
- BalAcc_maj：`0.6577`
- Acc_majority：`0.7367`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6399` | F1：`0.7992` | Acc：`0.7126`

#### Fold 4

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.7022`
- Val BalAcc_maj（附报）：`0.6208`

**Test 试次级**
- Acc_paper：`0.6957`
- BalAcc_maj：`0.6422`
- Acc_majority：`0.6957`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6257` | F1：`0.7627` | Acc：`0.6768`

### Three
- Val Acc_paper：`0.5154 ± 0.0283`
- Test Acc_paper：`0.5378 ± 0.0272`
- Test BalAcc_maj：`0.5519 ± 0.0264`
- Test 窗级 BalAcc（附报）：`0.5282 ± 0.0213`

### Three 分折明细

#### Fold 0

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5148`
- Val BalAcc_maj（附报）：`0.5267`

**Test 试次级**
- Acc_paper：`0.5188`
- BalAcc_maj：`0.5312`
- F1-macro（众数）：`0.5287`
- Rec idle/left/right：`0.4218` / `0.5864` / `0.5855`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5133` | F1m：`0.5109`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5319`
- Val BalAcc_maj（附报）：`0.5452`

**Test 试次级**
- Acc_paper：`0.5715`
- BalAcc_maj：`0.5830`
- F1-macro（众数）：`0.5824`
- Rec idle/left/right：`0.5536` / `0.6773` / `0.5182`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5526` | F1m：`0.5516`

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5019`
- Val BalAcc_maj（附报）：`0.5093`

**Test 试次级**
- Acc_paper：`0.4958`
- BalAcc_maj：`0.5118`
- F1-macro（众数）：`0.5105`
- Rec idle/left/right：`0.4500` / `0.5891` / `0.4964`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4960` | F1m：`0.4945`

#### Fold 3

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5563`
- Val BalAcc_maj（附报）：`0.5715`

**Test 试次级**
- Acc_paper：`0.5564`
- BalAcc_maj：`0.5715`
- F1-macro（众数）：`0.5668`
- Rec idle/left/right：`0.4173` / `0.6309` / `0.6664`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5486` | F1m：`0.5439`

#### Fold 4

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.4722`
- Val BalAcc_maj（附报）：`0.4911`

**Test 试次级**
- Acc_paper：`0.5467`
- BalAcc_maj：`0.5620`
- F1-macro（众数）：`0.5616`
- Rec idle/left/right：`0.5210` / `0.5890` / `0.5760`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5304` | F1m：`0.5299`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "non_blocking": true,
  "use_amp": true,
  "cudnn_benchmark": false,
  "gpu_memory_fraction": 1
}
```

- 结束：`2026-08-06T04:57:39`
