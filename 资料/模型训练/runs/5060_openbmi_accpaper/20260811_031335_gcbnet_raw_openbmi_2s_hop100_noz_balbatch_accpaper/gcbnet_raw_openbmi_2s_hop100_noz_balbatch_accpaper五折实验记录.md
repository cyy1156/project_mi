# 被试独立五折实验记录（20260811_031335 / gcbnet_raw_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-11T03:13:35`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet_raw` | TemporalEncoder(D=64) + GCBNet(k=2)；OpenBMI 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\gcbnet_raw_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260811_031335`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6882 ± 0.0227`
- Test Acc_paper：`0.6888 ± 0.0177`
- Test BalAcc_maj：`0.6063 ± 0.0045`
- Test 窗级 BalAcc（附报）：`0.5999 ± 0.0052`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6937`
- Val BalAcc_maj（附报）：`0.6336`

**Test 试次级**
- Acc_paper：`0.6858`
- BalAcc_maj：`0.6141`
- Acc_majority：`0.6858`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6071` | F1：`0.7683` | Acc：`0.6745`

#### Fold 1

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6993`
- Val BalAcc_maj（附报）：`0.6131`

**Test 试次级**
- Acc_paper：`0.6758`
- BalAcc_maj：`0.6014`
- Acc_majority：`0.6758`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6003` | F1：`0.7692` | Acc：`0.6728`

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6619`
- Val BalAcc_maj（附报）：`0.6269`

**Test 试次级**
- Acc_paper：`0.6658`
- BalAcc_maj：`0.6084`
- Acc_majority：`0.6658`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5913` | F1：`0.7433` | Acc：`0.6486`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7222`
- Val BalAcc_maj（附报）：`0.6050`

**Test 试次级**
- Acc_paper：`0.7148`
- BalAcc_maj：`0.6041`
- Acc_majority：`0.7148`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5985` | F1：`0.8080` | Acc：`0.7071`

#### Fold 4

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6641`
- Val BalAcc_maj（附报）：`0.5569`

**Test 试次级**
- Acc_paper：`0.7020`
- BalAcc_maj：`0.6035`
- Acc_majority：`0.7020`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6024` | F1：`0.7959` | Acc：`0.6970`

### Three
- Val Acc_paper：`0.4790 ± 0.0369`
- Test Acc_paper：`0.4886 ± 0.0157`
- Test BalAcc_maj：`0.4964 ± 0.0150`
- Test 窗级 BalAcc（附报）：`0.4845 ± 0.0134`

### Three 分折明细

#### Fold 0

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.4956`
- Val BalAcc_maj（附报）：`0.5052`

**Test 试次级**
- Acc_paper：`0.4727`
- BalAcc_maj：`0.4830`
- F1-macro（众数）：`0.4798`
- Rec idle/left/right：`0.3855` / `0.5827` / `0.4809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4703` | F1m：`0.4678`

#### Fold 1

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.5104`
- Val BalAcc_maj（附报）：`0.5156`

**Test 试次级**
- Acc_paper：`0.5042`
- BalAcc_maj：`0.5076`
- F1-macro（众数）：`0.4836`
- Rec idle/left/right：`0.2264` / `0.6336` / `0.6627`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4957` | F1m：`0.4732`

#### Fold 2

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.4452`
- Val BalAcc_maj（附报）：`0.4515`

**Test 试次级**
- Acc_paper：`0.4682`
- BalAcc_maj：`0.4752`
- F1-macro（众数）：`0.4479`
- Rec idle/left/right：`0.2045` / `0.7291` / `0.4918`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4664` | F1m：`0.4415`

#### Fold 3

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5181`
- Val BalAcc_maj（附报）：`0.5215`

**Test 试次级**
- Acc_paper：`0.5061`
- BalAcc_maj：`0.5148`
- F1-macro（众数）：`0.5022`
- Rec idle/left/right：`0.3036` / `0.5682` / `0.6727`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4981` | F1m：`0.4862`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4256`
- Val BalAcc_maj（附报）：`0.4348`

**Test 试次级**
- Acc_paper：`0.4917`
- BalAcc_maj：`0.5013`
- F1-macro（众数）：`0.4851`
- Rec idle/left/right：`0.2790` / `0.7090` / `0.5160`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4922` | F1m：`0.4779`

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

- 结束：`2026-08-11T04:40:46`
