# 被试独立五折实验记录（20260812_013626 / shallow_s1a_t50_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-12T01:36:26`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_s1a_t50` | ShallowFBCSPNet S1a_t50 | tlen=50 nF=40/40 pool_stride=15
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_net_enhance_three_accpaper\shallow_s1a_t50_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260812_013626`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6806 ± 0.0327`
- Test Acc_paper：`0.6906 ± 0.0349`
- Test BalAcc_maj：`0.6774 ± 0.0214`
- Test 窗级 BalAcc（附报）：`0.6495 ± 0.0160`

### Task 分折明细

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6785`
- Val BalAcc_maj（附报）：`0.6711`

**Test 试次级**
- Acc_paper：`0.6933`
- BalAcc_maj：`0.6784`
- Acc_majority：`0.6933`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6485` | F1：`0.7299` | Acc：`0.6612`

#### Fold 1

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.7270`
- Val BalAcc_maj（附报）：`0.6786`

**Test 试次级**
- Acc_paper：`0.7267`
- BalAcc_maj：`0.7075`
- Acc_majority：`0.7267`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6705` | F1：`0.7579` | Acc：`0.6898`

#### Fold 2

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.6507`
- Val BalAcc_maj（附报）：`0.6511`

**Test 试次级**
- Acc_paper：`0.6370`
- BalAcc_maj：`0.6493`
- Acc_majority：`0.6370`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6246` | F1：`0.6721` | Acc：`0.6141`

#### Fold 3

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.7063`
- Val BalAcc_maj（附报）：`0.7019`

**Test 试次级**
- Acc_paper：`0.7279`
- BalAcc_maj：`0.6930`
- Acc_majority：`0.7279`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6622` | F1：`0.7690` | Acc：`0.6951`

#### Fold 4

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.6404`
- Val BalAcc_maj（附报）：`0.6042`

**Test 试次级**
- Acc_paper：`0.6680`
- BalAcc_maj：`0.6587`
- Acc_majority：`0.6680`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6419` | F1：`0.7206` | Acc：`0.6522`

### Three
- Val Acc_paper：`0.5187 ± 0.0307`
- Test Acc_paper：`0.5389 ± 0.0247`
- Test BalAcc_maj：`0.5540 ± 0.0256`
- Test 窗级 BalAcc（附报）：`0.5271 ± 0.0205`

### Three 分折明细

#### Fold 0

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5348`
- Val BalAcc_maj（附报）：`0.5533`

**Test 试次级**
- Acc_paper：`0.5230`
- BalAcc_maj：`0.5409`
- F1-macro（众数）：`0.5396`
- Rec idle/left/right：`0.6164` / `0.5218` / `0.4845`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5184` | F1m：`0.5176`

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5278`
- Val BalAcc_maj（附报）：`0.5489`

**Test 试次级**
- Acc_paper：`0.5712`
- BalAcc_maj：`0.5870`
- F1-macro（众数）：`0.5859`
- Rec idle/left/right：`0.5864` / `0.6682` / `0.5064`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5519` | F1m：`0.5506`

#### Fold 2

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5081`
- Val BalAcc_maj（附报）：`0.5215`

**Test 试次级**
- Acc_paper：`0.5076`
- BalAcc_maj：`0.5197`
- F1-macro（众数）：`0.5160`
- Rec idle/left/right：`0.6436` / `0.5000` / `0.4155`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4942` | F1m：`0.4915`

#### Fold 3

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5570`
- Val BalAcc_maj（附报）：`0.5652`

**Test 试次级**
- Acc_paper：`0.5645`
- BalAcc_maj：`0.5803`
- F1-macro（众数）：`0.5791`
- Rec idle/left/right：`0.4991` / `0.5627` / `0.6791`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5453` | F1m：`0.5439`

#### Fold 4

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.4659`
- Val BalAcc_maj（附报）：`0.4822`

**Test 试次级**
- Acc_paper：`0.5283`
- BalAcc_maj：`0.5420`
- F1-macro（众数）：`0.5416`
- Rec idle/left/right：`0.5780` / `0.5430` / `0.5050`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5256` | F1m：`0.5255`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-12T03:27:39`
