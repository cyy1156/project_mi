# 被试独立五折实验记录（20260804_214540 / dgcnn_raw_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T21:45:40`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-offline-native` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`dgcnn_raw` | TemporalEncoder(D=64) + DGCNN(k=2)；2s/hop100 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\dgcnn_raw_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_214540`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.5924 ± 0.1116`
- Test Acc_paper：`0.4889 ± 0.1360`
- Test BalAcc_maj：`0.5094 ± 0.0437`
- Test 窗级 BalAcc（附报）：`0.5064 ± 0.0338`

### Task 分折明细

#### Fold 0

- stopped_epoch：`21` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6909`
- Val BalAcc_maj（附报）：`0.5200`

**Test 试次级**
- Acc_paper：`0.5303`
- BalAcc_maj：`0.4674`
- Acc_majority：`0.5303`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.4749` | F1：`0.6276` | Acc：`0.5186`

#### Fold 1

- stopped_epoch：`53` | best_epoch：`35`
- Val Acc_paper（早停）：`0.5526`
- Val BalAcc_maj（附报）：`0.5971`

**Test 试次级**
- Acc_paper：`0.3844`
- BalAcc_maj：`0.5325`
- Acc_majority：`0.3844`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.5256` | F1：`0.2326` | Acc：`0.3828`

#### Fold 2

- stopped_epoch：`21` | best_epoch：`3`
- Val Acc_paper（早停）：`0.6421`
- Val BalAcc_maj（附报）：`0.4772`

**Test 试次级**
- Acc_paper：`0.5231`
- BalAcc_maj：`0.4652`
- Acc_majority：`0.5231`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.4704` | F1：`0.6428` | Acc：`0.5266`

#### Fold 3

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6842`
- Val BalAcc_maj（附报）：`0.5082`

**Test 试次级**
- Acc_paper：`0.7011`
- BalAcc_maj：`0.5817`
- Acc_majority：`0.7011`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.5612` | F1：`0.7742` | Acc：`0.6644`

#### Fold 4

- stopped_epoch：`23` | best_epoch：`5`
- Val Acc_paper（早停）：`0.3923`
- Val BalAcc_maj（附报）：`0.4618`

**Test 试次级**
- Acc_paper：`0.3054`
- BalAcc_maj：`0.5000`
- Acc_majority：`0.3054`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.5000` | F1：`0.0000` | Acc：`0.2996`

### Three
- Val Acc_paper：`0.4301 ± 0.0798`
- Test Acc_paper：`0.3413 ± 0.0280`
- Test BalAcc_maj：`0.3651 ± 0.0278`
- Test 窗级 BalAcc（附报）：`0.3702 ± 0.0249`

### Three 分折明细

#### Fold 0

- stopped_epoch：`27` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5697`
- Val BalAcc_maj（附报）：`0.5997`

**Test 试次级**
- Acc_paper：`0.3258`
- BalAcc_maj：`0.3462`
- F1-macro（众数）：`0.2916`
- Rec idle/left/right：`0.7442` / `0.2061` / `0.0882`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.3547` | F1m：`0.3181`

#### Fold 1

- stopped_epoch：`52` | best_epoch：`34`
- Val Acc_paper（早停）：`0.4526`
- Val BalAcc_maj（附报）：`0.4766`

**Test 试次级**
- Acc_paper：`0.3894`
- BalAcc_maj：`0.4148`
- F1-macro（众数）：`0.3989`
- Rec idle/left/right：`0.6279` / `0.3926` / `0.2239`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4050` | F1m：`0.3928`

#### Fold 2

- stopped_epoch：`77` | best_epoch：`59`
- Val Acc_paper（早停）：`0.3842`
- Val BalAcc_maj（附报）：`0.4020`

**Test 试次级**
- Acc_paper：`0.3487`
- BalAcc_maj：`0.3627`
- F1-macro（众数）：`0.2863`
- Rec idle/left/right：`0.7500` / `0.3231` / `0.0152`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.3708` | F1m：`0.3132`

#### Fold 3

- stopped_epoch：`26` | best_epoch：`8`
- Val Acc_paper（早停）：`0.4105`
- Val BalAcc_maj（附报）：`0.4201`

**Test 试次级**
- Acc_paper：`0.3370`
- BalAcc_maj：`0.3686`
- F1-macro（众数）：`0.3689`
- Rec idle/left/right：`0.3109` / `0.3902` / `0.4048`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.3871` | F1m：`0.3871`

#### Fold 4

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.3333`
- Val BalAcc_maj（附报）：`0.3385`

**Test 试次级**
- Acc_paper：`0.3054`
- BalAcc_maj：`0.3333`
- F1-macro（众数）：`0.1560`
- Rec idle/left/right：`1.0000` / `0.0000` / `0.0000`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.3333` | F1m：`0.1537`

### 共用超参
```json
{
  "data_tag": "bci2a_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 18,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-offline-native",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true
}
```

- 结束：`2026-08-04T22:10:05`
