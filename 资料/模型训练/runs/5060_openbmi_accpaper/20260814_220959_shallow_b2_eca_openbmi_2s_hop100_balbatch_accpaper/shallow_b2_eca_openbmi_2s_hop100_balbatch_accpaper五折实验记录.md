# 被试独立五折实验记录（20260814_220959 / shallow_b2_eca_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-14T22:09:59`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-se-eca subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_b2_eca` | B2 ECA→Shallow · 五折 · Acc_paper | ECA k=3
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_se_eca_accpaper\shallow_b2_eca_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260814_220959`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-se-eca subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6842 ± 0.0321`
- Test Acc_paper：`0.6972 ± 0.0313`
- Test BalAcc_maj：`0.6746 ± 0.0167`
- Test 窗级 BalAcc（附报）：`0.6490 ± 0.0139`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6859`
- Val BalAcc_maj（附报）：`0.6769`

**Test 试次级**
- Acc_paper：`0.6942`
- BalAcc_maj：`0.6752`
- Acc_majority：`0.6942`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6480` | F1：`0.7329` | Acc：`0.6631`

#### Fold 1

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.7281`
- Val BalAcc_maj（附报）：`0.6789`

**Test 试次级**
- Acc_paper：`0.7282`
- BalAcc_maj：`0.7002`
- Acc_majority：`0.7282`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6684` | F1：`0.7636` | Acc：`0.6933`

#### Fold 2

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6493`
- Val BalAcc_maj（附报）：`0.6544`

**Test 试次级**
- Acc_paper：`0.6642`
- BalAcc_maj：`0.6548`
- Acc_majority：`0.6642`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6306` | F1：`0.7100` | Acc：`0.6405`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.7100`
- Val BalAcc_maj（附报）：`0.6858`

**Test 试次级**
- Acc_paper：`0.7373`
- BalAcc_maj：`0.6839`
- Acc_majority：`0.7373`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6599` | F1：`0.7871` | Acc：`0.7090`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6478`
- Val BalAcc_maj（附报）：`0.6150`

**Test 试次级**
- Acc_paper：`0.6620`
- BalAcc_maj：`0.6587`
- Acc_majority：`0.6620`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6379` | F1：`0.7102` | Acc：`0.6436`

### Three
- Val Acc_paper：`0.5224 ± 0.0320`
- Test Acc_paper：`0.5427 ± 0.0303`
- Test BalAcc_maj：`0.5599 ± 0.0316`
- Test 窗级 BalAcc（附报）：`0.5300 ± 0.0237`

### Three 分折明细

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5296`
- Val BalAcc_maj（附报）：`0.5459`

**Test 试次级**
- Acc_paper：`0.5239`
- BalAcc_maj：`0.5427`
- F1-macro（众数）：`0.5411`
- Rec idle/left/right：`0.6218` / `0.5327` / `0.4736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5152` | F1m：`0.5139`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5400`
- Val BalAcc_maj（附报）：`0.5585`

**Test 试次级**
- Acc_paper：`0.5773`
- BalAcc_maj：`0.5933`
- F1-macro（众数）：`0.5935`
- Rec idle/left/right：`0.6018` / `0.6309` / `0.5473`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5535` | F1m：`0.5532`

#### Fold 2

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5041`
- Val BalAcc_maj（附报）：`0.5178`

**Test 试次级**
- Acc_paper：`0.4979`
- BalAcc_maj：`0.5133`
- F1-macro（众数）：`0.5077`
- Rec idle/left/right：`0.6255` / `0.5491` / `0.3655`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4950` | F1m：`0.4905`

#### Fold 3

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5659`
- Val BalAcc_maj（附报）：`0.5793`

**Test 试次级**
- Acc_paper：`0.5748`
- BalAcc_maj：`0.5970`
- F1-macro（众数）：`0.5956`
- Rec idle/left/right：`0.5391` / `0.5436` / `0.7082`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5585` | F1m：`0.5569`

#### Fold 4

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.4722`
- Val BalAcc_maj（附报）：`0.4952`

**Test 试次级**
- Acc_paper：`0.5393`
- BalAcc_maj：`0.5530`
- F1-macro（众数）：`0.5525`
- Rec idle/left/right：`0.5450` / `0.6020` / `0.5120`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5277` | F1m：`0.5273`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-se-eca subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-15T02:08:05`
