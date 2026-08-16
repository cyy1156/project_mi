# 被试独立五折实验记录（20260816_121646 / shallow_hier_s0_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`20260816_121646` · chain step **S0_three**
- device：`cuda` · **train_mode=`fast`**（5090 全量五折）
- 训练设备：**NVIDIA RTX 5090**（32GB · conda cyy · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（OpenBMI hop100 · blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5090` | early_stop=**Acc_paper** | **balbatch** | no_rap
- model：`shallow_hier_s0_openbmi_2s_hop100_balbatch_accpaper` | S0 plain CE · shallow
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_three_hier_loss_accpaper\shallow_hier_s0_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260816_121646`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5090', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'keep_fold_packs': False, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---

## 最终结论（主报 Acc_paper）

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

