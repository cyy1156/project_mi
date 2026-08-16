# 被试独立五折实验记录（20260816_133223 / shallow_hier_h1_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`20260816_133223` · chain step **H1_three**
- device：`cuda` · **train_mode=`fast`**（5090 全量五折）
- 训练设备：**NVIDIA RTX 5090**（32GB · conda cyy · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（OpenBMI hop100 · blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5090` | early_stop=**Acc_paper** | **balbatch** | no_rap
- model：`shallow_hier_h1_openbmi_2s_hop100_balbatch_accpaper` | H1 CE+MI+LR · shallow
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_three_hier_loss_accpaper\shallow_hier_h1_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260816_133223`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 256, 'batch_eval': 512, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-hier-loss subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train device=5090', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'keep_fold_packs': False, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 8, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---

## 最终结论（主报 Acc_paper）

### Three
- Val Acc_paper：`0.5207 ± 0.0299`
- Test Acc_paper：`0.5420 ± 0.0288`
- Test BalAcc_maj：`0.5577 ± 0.0292`
- Test 窗级 BalAcc（附报）：`0.5305 ± 0.0241`

### Three 分折明细

#### Fold 0

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.5248`
- Val BalAcc_maj（附报）：`0.5422`

**Test 试次级**
- Acc_paper：`0.5242`
- BalAcc_maj：`0.5406`
- F1-macro（众数）：`0.5402`
- Rec idle/left/right：`0.5645` / `0.5573` / `0.5000`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5190` | F1m：`0.5188`

#### Fold 1

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5363`
- Val BalAcc_maj（附报）：`0.5526`

**Test 试次级**
- Acc_paper：`0.5764`
- BalAcc_maj：`0.5936`
- F1-macro（众数）：`0.5948`
- Rec idle/left/right：`0.5836` / `0.6145` / `0.5827`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5597` | F1m：`0.5600`

#### Fold 2

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.5030`
- Val BalAcc_maj（附报）：`0.5152`

**Test 试次级**
- Acc_paper：`0.5027`
- BalAcc_maj：`0.5170`
- F1-macro（众数）：`0.5123`
- Rec idle/left/right：`0.6373` / `0.5291` / `0.3845`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4951` | F1m：`0.4914`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5637`
- Val BalAcc_maj（附报）：`0.5741`

**Test 试次级**
- Acc_paper：`0.5739`
- BalAcc_maj：`0.5885`
- F1-macro（众数）：`0.5874`
- Rec idle/left/right：`0.5200` / `0.5627` / `0.6827`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5553` | F1m：`0.5542`

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4756`
- Val BalAcc_maj（附报）：`0.4922`

**Test 试次级**
- Acc_paper：`0.5327`
- BalAcc_maj：`0.5487`
- F1-macro（众数）：`0.5481`
- Rec idle/left/right：`0.6140` / `0.5070` / `0.5250`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5233` | F1m：`0.5229`

