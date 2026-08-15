# 被试独立五折实验记录（20260810_185008 / eegtcnet_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-10T18:50:08`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegtcnet` | EEGTCNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\eegtcnet_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260810_185008`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6844 ± 0.0328`
- Test Acc_paper：`0.6869 ± 0.0164`
- Test BalAcc_maj：`0.5805 ± 0.0568`
- Test 窗级 BalAcc（附报）：`0.5772 ± 0.0481`

### Task 分折明细

#### Fold 0

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6793`
- Val BalAcc_maj（附报）：`0.6047`

**Test 试次级**
- Acc_paper：`0.6979`
- BalAcc_maj：`0.6077`
- Acc_majority：`0.6979`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6009` | F1：`0.7885` | Acc：`0.6898`

#### Fold 1

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.7322`
- Val BalAcc_maj（附报）：`0.6608`

**Test 试次级**
- Acc_paper：`0.7130`
- BalAcc_maj：`0.6748`
- Acc_majority：`0.7130`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6565` | F1：`0.7705` | Acc：`0.6942`

#### Fold 2

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6400`
- Val BalAcc_maj（附报）：`0.5339`

**Test 试次级**
- Acc_paper：`0.6818`
- BalAcc_maj：`0.5691`
- Acc_majority：`0.6818`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5658` | F1：`0.7861` | Acc：`0.6754`

#### Fold 3

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.7089`
- Val BalAcc_maj（附报）：`0.5883`

**Test 试次级**
- Acc_paper：`0.6721`
- BalAcc_maj：`0.5377`
- Acc_majority：`0.6721`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5455` | F1：`0.7936` | Acc：`0.6755`

#### Fold 4

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6619`
- Val BalAcc_maj（附报）：`0.5108`

**Test 试次级**
- Acc_paper：`0.6697`
- BalAcc_maj：`0.5130`
- Acc_majority：`0.6697`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5173` | F1：`0.7987` | Acc：`0.6711`

### Three
- Val Acc_paper：`0.5027 ± 0.0449`
- Test Acc_paper：`0.5218 ± 0.0256`
- Test BalAcc_maj：`0.5309 ± 0.0247`
- Test 窗级 BalAcc（附报）：`0.5149 ± 0.0253`

### Three 分折明细

#### Fold 0

- stopped_epoch：`86` | best_epoch：`66`
- Val Acc_paper（早停）：`0.5252`
- Val BalAcc_maj（附报）：`0.5356`

**Test 试次级**
- Acc_paper：`0.4970`
- BalAcc_maj：`0.5064`
- F1-macro（众数）：`0.4990`
- Rec idle/left/right：`0.3382` / `0.5718` / `0.6091`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4918` | F1m：`0.4849`

#### Fold 1

- stopped_epoch：`100` | best_epoch：`80`
- Val Acc_paper（早停）：`0.5459`
- Val BalAcc_maj（附报）：`0.5526`

**Test 试次级**
- Acc_paper：`0.5518`
- BalAcc_maj：`0.5594`
- F1-macro（众数）：`0.5569`
- Rec idle/left/right：`0.4355` / `0.6736` / `0.5691`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5484` | F1m：`0.5459`

#### Fold 2

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.4652`
- Val BalAcc_maj（附报）：`0.4737`

**Test 试次级**
- Acc_paper：`0.4873`
- BalAcc_maj：`0.4982`
- F1-macro（众数）：`0.4796`
- Rec idle/left/right：`0.2536` / `0.6818` / `0.5591`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4808` | F1m：`0.4649`

#### Fold 3

- stopped_epoch：`75` | best_epoch：`55`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5504`

**Test 试次级**
- Acc_paper：`0.5448`
- BalAcc_maj：`0.5536`
- F1-macro（众数）：`0.5439`
- Rec idle/left/right：`0.4182` / `0.4527` / `0.7900`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5332` | F1m：`0.5246`

#### Fold 4

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.4344`
- Val BalAcc_maj（附报）：`0.4400`

**Test 试次级**
- Acc_paper：`0.5280`
- BalAcc_maj：`0.5370`
- F1-macro（众数）：`0.5303`
- Rec idle/left/right：`0.3820` / `0.6710` / `0.5580`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5205` | F1m：`0.5139`

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

- 结束：`2026-08-10T21:31:07`
