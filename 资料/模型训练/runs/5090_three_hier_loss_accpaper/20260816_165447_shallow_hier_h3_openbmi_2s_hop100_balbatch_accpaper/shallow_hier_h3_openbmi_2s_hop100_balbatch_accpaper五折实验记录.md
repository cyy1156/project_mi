# 被试独立五折实验记录（20260816_165447 / shallow_hier_h3_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`20260816_165447` · chain step **H3_three**
- device：`cuda` · **train_mode=`fast`**（5090 全量五折）
- 训练设备：**NVIDIA RTX 5090**（32GB · conda cyy · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（OpenBMI hop100 · blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5090` | early_stop=**Acc_paper** | **balbatch** | no_rap
- model：`shallow_hier_h3_openbmi_2s_hop100_balbatch_accpaper` | H3 H2 (+trial_cons later) · shallow
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_three_hier_loss_accpaper\shallow_hier_h3_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260816_165447`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5090', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'keep_fold_packs': False, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---

## 最终结论（主报 Acc_paper）

### Three
- Val Acc_paper：`0.5210 ± 0.0315`
- Test Acc_paper：`0.5437 ± 0.0299`
- Test BalAcc_maj：`0.5599 ± 0.0296`
- Test 窗级 BalAcc（附报）：`0.5304 ± 0.0229`

### Three 分折明细

#### Fold 0

- stopped_epoch：`54` | best_epoch：`34`
- Val Acc_paper（早停）：`0.5244`
- Val BalAcc_maj（附报）：`0.5430`

**Test 试次级**
- Acc_paper：`0.5270`
- BalAcc_maj：`0.5452`
- F1-macro（众数）：`0.5426`
- Rec idle/left/right：`0.6091` / `0.5845` / `0.4418`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5157` | F1m：`0.5138`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5367`
- Val BalAcc_maj（附报）：`0.5493`

**Test 试次级**
- Acc_paper：`0.5730`
- BalAcc_maj：`0.5885`
- F1-macro（众数）：`0.5893`
- Rec idle/left/right：`0.5627` / `0.6391` / `0.5636`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5565` | F1m：`0.5565`

#### Fold 2

- stopped_epoch：`50` | best_epoch：`30`
- Val Acc_paper（早停）：`0.5067`
- Val BalAcc_maj（附报）：`0.5189`

**Test 试次级**
- Acc_paper：`0.5045`
- BalAcc_maj：`0.5197`
- F1-macro（众数）：`0.5168`
- Rec idle/left/right：`0.5755` / `0.5791` / `0.4045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5002` | F1m：`0.4978`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5659`
- Val BalAcc_maj（附报）：`0.5774`

**Test 试次级**
- Acc_paper：`0.5839`
- BalAcc_maj：`0.5994`
- F1-macro（众数）：`0.5988`
- Rec idle/left/right：`0.5409` / `0.5791` / `0.6782`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5576` | F1m：`0.5568`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4711`
- Val BalAcc_maj（附报）：`0.4915`

**Test 试次级**
- Acc_paper：`0.5300`
- BalAcc_maj：`0.5467`
- F1-macro（众数）：`0.5456`
- Rec idle/left/right：`0.6270` / `0.4990` / `0.5140`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5219` | F1m：`0.5213`

