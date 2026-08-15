# 被试独立五折实验记录（20260810_213229 / gcbnet_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-10T21:32:29`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet` | GCBNet(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\gcbnet_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260810_213229`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6138 ± 0.0358`
- Test Acc_paper：`0.6038 ± 0.0197`
- Test BalAcc_maj：`0.5830 ± 0.0109`
- Test 窗级 BalAcc（附报）：`0.5747 ± 0.0070`

### Task 分折明细

#### Fold 0

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6022`
- Val BalAcc_maj（附报）：`0.6044`

**Test 试次级**
- Acc_paper：`0.6155`
- BalAcc_maj：`0.5800`
- Acc_majority：`0.6155`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5761` | F1：`0.6980` | Acc：`0.6096`

#### Fold 1

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6719`
- Val BalAcc_maj（附报）：`0.6242`

**Test 试次级**
- Acc_paper：`0.6273`
- BalAcc_maj：`0.6039`
- Acc_majority：`0.6273`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5851` | F1：`0.6885` | Acc：`0.6071`

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5611`
- Val BalAcc_maj（附报）：`0.5606`

**Test 试次级**
- Acc_paper：`0.6100`
- BalAcc_maj：`0.5730`
- Acc_majority：`0.6100`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5650` | F1：`0.6917` | Acc：`0.6006`

#### Fold 3

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6241`
- Val BalAcc_maj（附报）：`0.6108`

**Test 试次级**
- Acc_paper：`0.5697`
- BalAcc_maj：`0.5820`
- Acc_majority：`0.5697`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5780` | F1：`0.6274` | Acc：`0.5674`

#### Fold 4

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6096`
- Val BalAcc_maj（附报）：`0.5489`

**Test 试次级**
- Acc_paper：`0.5967`
- BalAcc_maj：`0.5760`
- Acc_majority：`0.5967`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5695` | F1：`0.6739` | Acc：`0.5909`

### Three
- Val Acc_paper：`0.4469 ± 0.0274`
- Test Acc_paper：`0.4598 ± 0.0238`
- Test BalAcc_maj：`0.4707 ± 0.0241`
- Test 窗级 BalAcc（附报）：`0.4581 ± 0.0239`

### Three 分折明细

#### Fold 0

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.4511`
- Val BalAcc_maj（附报）：`0.4611`

**Test 试次级**
- Acc_paper：`0.4300`
- BalAcc_maj：`0.4376`
- F1-macro（众数）：`0.4298`
- Rec idle/left/right：`0.2845` / `0.4800` / `0.5482`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4260` | F1m：`0.4183`

#### Fold 1

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.4689`
- Val BalAcc_maj（附报）：`0.4774`

**Test 试次级**
- Acc_paper：`0.4924`
- BalAcc_maj：`0.5021`
- F1-macro（众数）：`0.4912`
- Rec idle/left/right：`0.3064` / `0.6391` / `0.5609`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4916` | F1m：`0.4830`

#### Fold 2

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.4159`
- Val BalAcc_maj（附报）：`0.4274`

**Test 试次级**
- Acc_paper：`0.4370`
- BalAcc_maj：`0.4488`
- F1-macro（众数）：`0.4443`
- Rec idle/left/right：`0.3709` / `0.5818` / `0.3936`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4375` | F1m：`0.4335`

#### Fold 3

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.4830`
- Val BalAcc_maj（附报）：`0.4915`

**Test 试次级**
- Acc_paper：`0.4788`
- BalAcc_maj：`0.4882`
- F1-macro（众数）：`0.4862`
- Rec idle/left/right：`0.5527` / `0.4045` / `0.5073`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4743` | F1m：`0.4729`

#### Fold 4

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.4156`
- Val BalAcc_maj（附报）：`0.4244`

**Test 试次级**
- Acc_paper：`0.4607`
- BalAcc_maj：`0.4767`
- F1-macro（众数）：`0.4685`
- Rec idle/left/right：`0.3120` / `0.5990` / `0.5190`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4612` | F1m：`0.4537`

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

- 结束：`2026-08-10T22:21:52`
