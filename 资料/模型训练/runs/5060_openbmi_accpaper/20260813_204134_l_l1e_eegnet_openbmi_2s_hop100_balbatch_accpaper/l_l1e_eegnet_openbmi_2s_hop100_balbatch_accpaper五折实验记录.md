# 被试独立五折实验记录（20260813_204134 / l_l1e_eegnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-13T20:41:34`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`l_l1e_eegnet` | L1 五折 EEGNet · Acc_paper | EEGNet F1=8 D=2 F2=16
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_ciacnet_mi_accpaper\l_l1e_eegnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260813_204134`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6878 ± 0.0163`
- Test Acc_paper：`0.6866 ± 0.0328`
- Test BalAcc_maj：`0.6585 ± 0.0274`
- Test 窗级 BalAcc（附报）：`0.6368 ± 0.0213`

### Task 分折明细

#### Fold 0

- stopped_epoch：`81` | best_epoch：`61`
- Val Acc_paper（早停）：`0.7011`
- Val BalAcc_maj（附报）：`0.6631`

**Test 试次级**
- Acc_paper：`0.6976`
- BalAcc_maj：`0.6589`
- Acc_majority：`0.6976`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6337` | F1：`0.7507` | Acc：`0.6705`

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.7067`
- Val BalAcc_maj（附报）：`0.6817`

**Test 试次级**
- Acc_paper：`0.6736`
- BalAcc_maj：`0.6591`
- Acc_majority：`0.6736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6393` | F1：`0.7214` | Acc：`0.6517`

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6852`
- Val BalAcc_maj（附报）：`0.6586`

**Test 试次级**
- Acc_paper：`0.6412`
- BalAcc_maj：`0.6211`
- Acc_majority：`0.6412`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6039` | F1：`0.7014` | Acc：`0.6236`

#### Fold 3

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.6863`
- Val BalAcc_maj（附报）：`0.6867`

**Test 试次级**
- Acc_paper：`0.7412`
- BalAcc_maj：`0.7059`
- Acc_majority：`0.7412`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6711` | F1：`0.7731` | Acc：`0.7016`

#### Fold 4

- stopped_epoch：`114` | best_epoch：`94`
- Val Acc_paper（早停）：`0.6596`
- Val BalAcc_maj（附报）：`0.5825`

**Test 试次级**
- Acc_paper：`0.6793`
- BalAcc_maj：`0.6478`
- Acc_majority：`0.6793`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6361` | F1：`0.7448` | Acc：`0.6670`

### Three
- Val Acc_paper：`0.5107 ± 0.0356`
- Test Acc_paper：`0.5305 ± 0.0331`
- Test BalAcc_maj：`0.5434 ± 0.0343`
- Test 窗级 BalAcc（附报）：`0.5231 ± 0.0254`

### Three 分折明细

#### Fold 0

- stopped_epoch：`107` | best_epoch：`87`
- Val Acc_paper（早停）：`0.5119`
- Val BalAcc_maj（附报）：`0.5226`

**Test 试次级**
- Acc_paper：`0.5109`
- BalAcc_maj：`0.5245`
- F1-macro（众数）：`0.5237`
- Rec idle/left/right：`0.4609` / `0.5800` / `0.5327`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5077` | F1m：`0.5069`

#### Fold 1

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5319`
- Val BalAcc_maj（附报）：`0.5452`

**Test 试次级**
- Acc_paper：`0.5703`
- BalAcc_maj：`0.5809`
- F1-macro（众数）：`0.5801`
- Rec idle/left/right：`0.4809` / `0.6436` / `0.6182`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5524` | F1m：`0.5508`

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.4952`
- Val BalAcc_maj（附报）：`0.5096`

**Test 试次级**
- Acc_paper：`0.4794`
- BalAcc_maj：`0.4900`
- F1-macro（众数）：`0.4822`
- Rec idle/left/right：`0.4545` / `0.6655` / `0.3500`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4833` | F1m：`0.4765`

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5604`
- Val BalAcc_maj（附报）：`0.5763`

**Test 试次级**
- Acc_paper：`0.5606`
- BalAcc_maj：`0.5791`
- F1-macro（众数）：`0.5779`
- Rec idle/left/right：`0.4945` / `0.5555` / `0.6873`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5465` | F1m：`0.5442`

#### Fold 4

- stopped_epoch：`90` | best_epoch：`70`
- Val Acc_paper（早停）：`0.4544`
- Val BalAcc_maj（附报）：`0.4689`

**Test 试次级**
- Acc_paper：`0.5313`
- BalAcc_maj：`0.5423`
- F1-macro（众数）：`0.5418`
- Rec idle/left/right：`0.5800` / `0.5660` / `0.4810`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5256` | F1m：`0.5252`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-13T23:47:27`
