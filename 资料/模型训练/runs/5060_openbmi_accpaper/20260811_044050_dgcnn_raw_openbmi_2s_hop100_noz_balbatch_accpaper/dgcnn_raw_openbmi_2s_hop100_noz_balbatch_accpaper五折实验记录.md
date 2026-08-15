# 被试独立五折实验记录（20260811_044050 / dgcnn_raw_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-11T04:40:50`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dgcnn_raw` | TemporalEncoder(D=64) + DGCNN(k=2)；OpenBMI 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\dgcnn_raw_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260811_044050`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7018 ± 0.0241`
- Test Acc_paper：`0.6902 ± 0.0150`
- Test BalAcc_maj：`0.5849 ± 0.0171`
- Test 窗级 BalAcc（附报）：`0.5847 ± 0.0134`

### Task 分折明细

#### Fold 0

- stopped_epoch：`76` | best_epoch：`56`
- Val Acc_paper（早停）：`0.7167`
- Val BalAcc_maj（附报）：`0.6006`

**Test 试次级**
- Acc_paper：`0.6912`
- BalAcc_maj：`0.5823`
- Acc_majority：`0.6912`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5836` | F1：`0.7934` | Acc：`0.6883`

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.7059`
- Val BalAcc_maj（附报）：`0.5647`

**Test 试次级**
- Acc_paper：`0.6655`
- BalAcc_maj：`0.5539`
- Acc_majority：`0.6655`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5607` | F1：`0.7809` | Acc：`0.6688`

#### Fold 2

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.6819`
- Val BalAcc_maj（附报）：`0.5928`

**Test 试次级**
- Acc_paper：`0.6933`
- BalAcc_maj：`0.5891`
- Acc_majority：`0.6933`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5859` | F1：`0.7904` | Acc：`0.6863`

#### Fold 3

- stopped_epoch：`63` | best_epoch：`43`
- Val Acc_paper（早停）：`0.7359`
- Val BalAcc_maj（附报）：`0.6489`

**Test 试次级**
- Acc_paper：`0.7124`
- BalAcc_maj：`0.6043`
- Acc_majority：`0.7124`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6007` | F1：`0.8046` | Acc：`0.7046`

#### Fold 4

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.6685`
- Val BalAcc_maj（附报）：`0.5639`

**Test 试次级**
- Acc_paper：`0.6887`
- BalAcc_maj：`0.5948`
- Acc_majority：`0.6887`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5924` | F1：`0.7843` | Acc：`0.6830`

### Three
- Val Acc_paper：`0.4921 ± 0.0479`
- Test Acc_paper：`0.4909 ± 0.0245`
- Test BalAcc_maj：`0.4986 ± 0.0251`
- Test 窗级 BalAcc（附报）：`0.4893 ± 0.0210`

### Three 分折明细

#### Fold 0

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.5181`
- Val BalAcc_maj（附报）：`0.5285`

**Test 试次级**
- Acc_paper：`0.4642`
- BalAcc_maj：`0.4724`
- F1-macro（众数）：`0.4649`
- Rec idle/left/right：`0.3136` / `0.5818` / `0.5218`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4695` | F1m：`0.4628`

#### Fold 1

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.5167`
- Val BalAcc_maj（附报）：`0.5219`

**Test 试次级**
- Acc_paper：`0.5000`
- BalAcc_maj：`0.5052`
- F1-macro（众数）：`0.4924`
- Rec idle/left/right：`0.2927` / `0.6209` / `0.6018`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4947` | F1m：`0.4827`

#### Fold 2

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.4415`
- Val BalAcc_maj（附报）：`0.4470`

**Test 试次级**
- Acc_paper：`0.4594`
- BalAcc_maj：`0.4661`
- F1-macro（众数）：`0.4400`
- Rec idle/left/right：`0.2173` / `0.7473` / `0.4336`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4602` | F1m：`0.4366`

#### Fold 3

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.5541`
- Val BalAcc_maj（附报）：`0.5633`

**Test 试次级**
- Acc_paper：`0.5173`
- BalAcc_maj：`0.5273`
- F1-macro（众数）：`0.5193`
- Rec idle/left/right：`0.3491` / `0.5836` / `0.6491`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5115` | F1m：`0.5039`

#### Fold 4

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.4304`
- Val BalAcc_maj（附报）：`0.4374`

**Test 试次级**
- Acc_paper：`0.5137`
- BalAcc_maj：`0.5220`
- F1-macro（众数）：`0.5146`
- Rec idle/left/right：`0.3730` / `0.6840` / `0.5090`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5105` | F1m：`0.5036`

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

- 结束：`2026-08-11T06:52:16`
