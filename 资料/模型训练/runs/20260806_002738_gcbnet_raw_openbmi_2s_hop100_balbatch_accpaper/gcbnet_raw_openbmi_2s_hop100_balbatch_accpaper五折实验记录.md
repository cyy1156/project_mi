# 被试独立五折实验记录（20260806_002738 / gcbnet_raw_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T00:27:38`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet_raw` | TemporalEncoder(D=64) + GCBNet(k=2)；OpenBMI 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_openbmi_2s_hop100_accpaper\gcbnet_raw_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_002738`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6865 ± 0.0353`
- Test Acc_paper：`0.6722 ± 0.0240`
- Test BalAcc_maj：`0.6447 ± 0.0076`
- Test 窗级 BalAcc（附报）：`0.6262 ± 0.0052`

### Task 分折明细

#### Fold 0

- stopped_epoch：`55` | best_epoch：`35`
- Val Acc_paper（早停）：`0.6967`
- Val BalAcc_maj（附报）：`0.6614`

**Test 试次级**
- Acc_paper：`0.6830`
- BalAcc_maj：`0.6452`
- Acc_majority：`0.6830`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6221` | F1：`0.7388` | Acc：`0.6572`

#### Fold 1

- stopped_epoch：`69` | best_epoch：`49`
- Val Acc_paper（早停）：`0.6874`
- Val BalAcc_maj（附报）：`0.6561`

**Test 试次级**
- Acc_paper：`0.6427`
- BalAcc_maj：`0.6430`
- Acc_majority：`0.6427`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6254` | F1：`0.6870` | Acc：`0.6235`

#### Fold 2

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6989`
- Val BalAcc_maj（附报）：`0.6633`

**Test 试次级**
- Acc_paper：`0.6785`
- BalAcc_maj：`0.6380`
- Acc_majority：`0.6785`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6224` | F1：`0.7430` | Acc：`0.6604`

#### Fold 3

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.7281`
- Val BalAcc_maj（附报）：`0.6903`

**Test 试次级**
- Acc_paper：`0.7082`
- BalAcc_maj：`0.6589`
- Acc_majority：`0.7082`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6364` | F1：`0.7634` | Acc：`0.6813`

#### Fold 4

- stopped_epoch：`83` | best_epoch：`63`
- Val Acc_paper（早停）：`0.6215`
- Val BalAcc_maj（附报）：`0.5947`

**Test 试次级**
- Acc_paper：`0.6483`
- BalAcc_maj：`0.6385`
- Acc_majority：`0.6483`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6249` | F1：`0.7026` | Acc：`0.6332`

### Three
- Val Acc_paper：`0.4761 ± 0.0352`
- Test Acc_paper：`0.4802 ± 0.0229`
- Test BalAcc_maj：`0.4992 ± 0.0216`
- Test 窗级 BalAcc（附报）：`0.4795 ± 0.0190`

### Three 分折明细

#### Fold 0

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.4726`
- Val BalAcc_maj（附报）：`0.4878`

**Test 试次级**
- Acc_paper：`0.4712`
- BalAcc_maj：`0.4903`
- F1-macro（众数）：`0.4856`
- Rec idle/left/right：`0.3691` / `0.6191` / `0.4827`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4688` | F1m：`0.4649`

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.4837`
- Val BalAcc_maj（附报）：`0.4933`

**Test 试次级**
- Acc_paper：`0.5136`
- BalAcc_maj：`0.5303`
- F1-macro（众数）：`0.5293`
- Rec idle/left/right：`0.5818` / `0.5455` / `0.4636`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5099` | F1m：`0.5091`

#### Fold 2

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.4611`
- Val BalAcc_maj（附报）：`0.4815`

**Test 试次级**
- Acc_paper：`0.4458`
- BalAcc_maj：`0.4658`
- F1-macro（众数）：`0.4657`
- Rec idle/left/right：`0.4000` / `0.4927` / `0.5045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4540` | F1m：`0.4533`

#### Fold 3

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5356`
- Val BalAcc_maj（附报）：`0.5522`

**Test 试次级**
- Acc_paper：`0.4755`
- BalAcc_maj：`0.4979`
- F1-macro（众数）：`0.4969`
- Rec idle/left/right：`0.4218` / `0.4827` / `0.5891`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4756` | F1m：`0.4748`

#### Fold 4

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.4274`
- Val BalAcc_maj（附报）：`0.4463`

**Test 试次级**
- Acc_paper：`0.4950`
- BalAcc_maj：`0.5117`
- F1-macro（众数）：`0.5116`
- Rec idle/left/right：`0.5020` / `0.4960` / `0.5370`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4890` | F1m：`0.4890`

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

- 结束：`2026-08-06T03:14:45`
