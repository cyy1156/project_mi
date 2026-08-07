# 被试独立五折实验记录（20260806_132818 / gcbnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T13:28:18`
- device：`cuda`
- 训练设备：**NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet` | GCBNet(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_baseline_openbmi_2s_hop100_accpaper\gcbnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_132818`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6193 ± 0.0290`
- Test Acc_paper：`0.6151 ± 0.0486`
- Test BalAcc_maj：`0.5596 ± 0.0163`
- Test 窗级 BalAcc（附报）：`0.5521 ± 0.0134`

### Task 分折明细

#### Fold 0

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5670`
- Val BalAcc_maj（附报）：`0.5514`

**Test 试次级**
- Acc_paper：`0.6427`
- BalAcc_maj：`0.5730`
- Acc_majority：`0.6427`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5606` | F1：`0.7291` | Acc：`0.6256`

#### Fold 1

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.6363`
- Val BalAcc_maj（附报）：`0.5764`

**Test 试次级**
- Acc_paper：`0.6676`
- BalAcc_maj：`0.5732`
- Acc_majority：`0.6676`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5683` | F1：`0.7532` | Acc：`0.6476`

#### Fold 2

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.6096`
- Val BalAcc_maj（附报）：`0.5622`

**Test 试次级**
- Acc_paper：`0.5288`
- BalAcc_maj：`0.5307`
- Acc_majority：`0.5288`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5332` | F1：`0.6076` | Acc：`0.5353`

#### Fold 3

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.6363`
- Val BalAcc_maj（附报）：`0.5683`

**Test 试次级**
- Acc_paper：`0.6388`
- BalAcc_maj：`0.5684`
- Acc_majority：`0.6388`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5588` | F1：`0.7222` | Acc：`0.6197`

#### Fold 4

- stopped_epoch：`76` | best_epoch：`56`
- Val Acc_paper（早停）：`0.6474`
- Val BalAcc_maj（附报）：`0.5700`

**Test 试次级**
- Acc_paper：`0.5977`
- BalAcc_maj：`0.5527`
- Acc_majority：`0.5977`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5394` | F1：`0.6745` | Acc：`0.5782`

### Three
- Val Acc_paper：`0.3748 ± 0.0095`
- Test Acc_paper：`0.3746 ± 0.0128`
- Test BalAcc_maj：`0.3942 ± 0.0137`
- Test 窗级 BalAcc（附报）：`0.3905 ± 0.0130`

### Three 分折明细

#### Fold 0

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.3656`
- Val BalAcc_maj（附报）：`0.3793`

**Test 试次级**
- Acc_paper：`0.3703`
- BalAcc_maj：`0.3900`
- F1-macro（众数）：`0.3710`
- Rec idle/left/right：`0.3173` / `0.6364` / `0.2164`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3872` | F1m：`0.3730`

#### Fold 1

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.3867`
- Val BalAcc_maj（附报）：`0.4137`

**Test 试次级**
- Acc_paper：`0.3827`
- BalAcc_maj：`0.4061`
- F1-macro（众数）：`0.3956`
- Rec idle/left/right：`0.2600` / `0.5945` / `0.3636`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3999` | F1m：`0.3913`

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.3700`
- Val BalAcc_maj（附报）：`0.3919`

**Test 试次级**
- Acc_paper：`0.3548`
- BalAcc_maj：`0.3736`
- F1-macro（众数）：`0.3476`
- Rec idle/left/right：`0.6527` / `0.2909` / `0.1773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3695` | F1m：`0.3520`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.3859`
- Val BalAcc_maj（附报）：`0.4078`

**Test 试次级**
- Acc_paper：`0.3930`
- BalAcc_maj：`0.4124`
- F1-macro（众数）：`0.4108`
- Rec idle/left/right：`0.3909` / `0.4891` / `0.3573`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4078` | F1m：`0.4068`

#### Fold 4

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3659`
- Val BalAcc_maj（附报）：`0.3844`

**Test 试次级**
- Acc_paper：`0.3720`
- BalAcc_maj：`0.3890`
- F1-macro（众数）：`0.3863`
- Rec idle/left/right：`0.4680` / `0.3040` / `0.3950`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.3884` | F1m：`0.3866`

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

- 结束：`2026-08-06T16:07:38`
