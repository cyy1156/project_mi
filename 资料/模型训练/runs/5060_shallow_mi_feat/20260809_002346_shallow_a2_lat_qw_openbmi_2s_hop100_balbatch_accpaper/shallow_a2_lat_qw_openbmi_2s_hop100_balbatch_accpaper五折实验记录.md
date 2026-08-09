# 被试独立五折实验记录（20260809_002346 / shallow_a2_lat_qw_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-09T00:23:46`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-mi-feat subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_a2_lat_qw` | ShallowFBCSPNet A2 A1-ch + trial quality weighted CE
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_mi_feat_openbmi_accpaper\shallow_a2_lat_qw_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260809_002346`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-mi-feat subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6805 ± 0.0339`
- Test Acc_paper：`0.6962 ± 0.0345`
- Test BalAcc_maj：`0.6738 ± 0.0192`
- Test 窗级 BalAcc（附报）：`0.6489 ± 0.0169`

### Task 分折明细

#### Fold 0

- stopped_epoch：`63` | best_epoch：`43`
- Val Acc_paper（早停）：`0.6767`
- Val BalAcc_maj（附报）：`0.6617`

**Test 试次级**
- Acc_paper：`0.6964`
- BalAcc_maj：`0.6825`
- Acc_majority：`0.6964`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6555` | F1：`0.7364` | Acc：`0.6685`

#### Fold 1

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.7270`
- Val BalAcc_maj（附报）：`0.6761`

**Test 试次级**
- Acc_paper：`0.7261`
- BalAcc_maj：`0.6977`
- Acc_majority：`0.7261`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6689` | F1：`0.7647` | Acc：`0.6943`

#### Fold 2

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.6415`
- Val BalAcc_maj（附报）：`0.6392`

**Test 试次级**
- Acc_paper：`0.6548`
- BalAcc_maj：`0.6502`
- Acc_majority：`0.6548`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6244` | F1：`0.6971` | Acc：`0.6295`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.7107`
- Val BalAcc_maj（附报）：`0.6808`

**Test 试次级**
- Acc_paper：`0.7424`
- BalAcc_maj：`0.6866`
- Acc_majority：`0.7424`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6618` | F1：`0.7908` | Acc：`0.7127`

#### Fold 4

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6467`
- Val BalAcc_maj（附报）：`0.6008`

**Test 试次级**
- Acc_paper：`0.6613`
- BalAcc_maj：`0.6520`
- Acc_majority：`0.6613`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6339` | F1：`0.7118` | Acc：`0.6430`

### Three
- Val Acc_paper：`0.5076 ± 0.0376`
- Test Acc_paper：`0.5298 ± 0.0269`
- Test BalAcc_maj：`0.5473 ± 0.0270`
- Test 窗级 BalAcc（附报）：`0.5232 ± 0.0227`

### Three 分折明细

#### Fold 0

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5170`
- Val BalAcc_maj（附报）：`0.5315`

**Test 试次级**
- Acc_paper：`0.5288`
- BalAcc_maj：`0.5464`
- F1-macro（众数）：`0.5451`
- Rec idle/left/right：`0.6100` / `0.5482` / `0.4809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5167` | F1m：`0.5157`

#### Fold 1

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5244`
- Val BalAcc_maj（附报）：`0.5404`

**Test 试次级**
- Acc_paper：`0.5612`
- BalAcc_maj：`0.5767`
- F1-macro（众数）：`0.5740`
- Rec idle/left/right：`0.6245` / `0.6545` / `0.4509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5461` | F1m：`0.5435`

#### Fold 2

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.4930`
- Val BalAcc_maj（附报）：`0.5126`

**Test 试次级**
- Acc_paper：`0.4815`
- BalAcc_maj：`0.4982`
- F1-macro（众数）：`0.4935`
- Rec idle/left/right：`0.5136` / `0.6127` / `0.3682`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4828` | F1m：`0.4787`

#### Fold 3

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.5585`
- Val BalAcc_maj（附报）：`0.5704`

**Test 试次级**
- Acc_paper：`0.5470`
- BalAcc_maj：`0.5664`
- F1-macro（众数）：`0.5667`
- Rec idle/left/right：`0.5218` / `0.5909` / `0.5864`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5419` | F1m：`0.5417`

#### Fold 4

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.4452`
- Val BalAcc_maj（附报）：`0.4574`

**Test 试次级**
- Acc_paper：`0.5303`
- BalAcc_maj：`0.5490`
- F1-macro（众数）：`0.5488`
- Rec idle/left/right：`0.5620` / `0.5740` / `0.5110`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5287` | F1m：`0.5286`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-mi-feat subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-09T02:58:36`
