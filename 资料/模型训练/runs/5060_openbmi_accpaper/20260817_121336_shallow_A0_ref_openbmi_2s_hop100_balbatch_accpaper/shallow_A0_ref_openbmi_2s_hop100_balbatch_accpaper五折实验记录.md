# 被试独立五折实验记录（20260817_121336 / shallow_A0_ref_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-17T12:13:36`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_A0_ref` | A0-ref braindecode Shallow · 500pt · Acc_paper（量级参考）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\shallow_A0_ref_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260817_121336`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': False, 'persistent_workers': False, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6837 ± 0.0348`
- Test Acc_paper：`0.6909 ± 0.0380`
- Test BalAcc_maj：`0.6765 ± 0.0209`
- Test 窗级 BalAcc（附报）：`0.6506 ± 0.0159`

### Task 分折明细

#### Fold 0

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.6796`
- Val BalAcc_maj（附报）：`0.6639`

**Test 试次级**
- Acc_paper：`0.6988`
- BalAcc_maj：`0.6827`
- Acc_majority：`0.6988`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6548` | F1：`0.7374` | Acc：`0.6690`

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.7304`
- Val BalAcc_maj（附报）：`0.6844`

**Test 试次级**
- Acc_paper：`0.7161`
- BalAcc_maj：`0.7018`
- Acc_majority：`0.7161`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6684` | F1：`0.7481` | Acc：`0.6818`

#### Fold 2

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.6437`
- Val BalAcc_maj（附报）：`0.6547`

**Test 试次级**
- Acc_paper：`0.6458`
- BalAcc_maj：`0.6541`
- Acc_majority：`0.6458`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6292` | F1：`0.6843` | Acc：`0.6235`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7159`
- Val BalAcc_maj（附报）：`0.6903`

**Test 试次级**
- Acc_paper：`0.7439`
- BalAcc_maj：`0.6936`
- Acc_majority：`0.7439`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6656` | F1：`0.7889` | Acc：`0.7124`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6489`
- Val BalAcc_maj（附报）：`0.6136`

**Test 试次级**
- Acc_paper：`0.6500`
- BalAcc_maj：`0.6500`
- Acc_majority：`0.6500`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6350` | F1：`0.6992` | Acc：`0.6353`

### Three
- Val Acc_paper：`0.5202 ± 0.0295`
- Test Acc_paper：`0.5425 ± 0.0306`
- Test BalAcc_maj：`0.5602 ± 0.0284`
- Test 窗级 BalAcc（附报）：`0.5302 ± 0.0230`

### Three 分折明细

#### Fold 0

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.5278`
- Val BalAcc_maj（附报）：`0.5430`

**Test 试次级**
- Acc_paper：`0.5185`
- BalAcc_maj：`0.5412`
- F1-macro（众数）：`0.5392`
- Rec idle/left/right：`0.6373` / `0.5091` / `0.4773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5147` | F1m：`0.5133`

#### Fold 1

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5367`
- Val BalAcc_maj（附报）：`0.5552`

**Test 试次级**
- Acc_paper：`0.5809`
- BalAcc_maj：`0.5952`
- F1-macro（众数）：`0.5961`
- Rec idle/left/right：`0.6000` / `0.6118` / `0.5736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5591` | F1m：`0.5594`

#### Fold 2

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5048`
- Val BalAcc_maj（附报）：`0.5200`

**Test 试次级**
- Acc_paper：`0.5076`
- BalAcc_maj：`0.5248`
- F1-macro（众数）：`0.5214`
- Rec idle/left/right：`0.6173` / `0.5518` / `0.4055`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4999` | F1m：`0.4975`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5593`
- Val BalAcc_maj（附报）：`0.5693`

**Test 试次级**
- Acc_paper：`0.5773`
- BalAcc_maj：`0.5924`
- F1-macro（众数）：`0.5915`
- Rec idle/left/right：`0.5364` / `0.5609` / `0.6800`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5544` | F1m：`0.5535`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4726`
- Val BalAcc_maj（附报）：`0.4889`

**Test 试次级**
- Acc_paper：`0.5283`
- BalAcc_maj：`0.5473`
- F1-macro（众数）：`0.5467`
- Rec idle/left/right：`0.6180` / `0.5060` / `0.5180`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5227` | F1m：`0.5221`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 20,
  "batch_train": 256,
  "batch_eval": 512,
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
  "pin_memory": false,
  "persistent_workers": false,
  "prefetch_factor": 2,
  "non_blocking": true,
  "torch_num_threads": 8,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true
}
```

- 结束：`2026-08-17T13:59:22`
