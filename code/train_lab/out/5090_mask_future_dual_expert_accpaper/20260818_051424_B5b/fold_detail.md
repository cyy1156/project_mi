# B5b · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_051424_B5b`
- Test Acc_paper: `0.5738 ± 0.0154`
- Test BalAcc_maj: `0.5792 ± 0.0157`
- Test win F1: `0.5688 ± 0.0148`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5733`
- Val BalAcc_maj（附报）：`0.5785`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5539`
- BalAcc_maj：`0.5597`
- Acc_majority：`0.5597`
- F1-macro（众数）：`0.5603`
- Recall-macro：`0.5597`
- Recall idle/left/right：`0.5382` / `0.5545` / `0.5864`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5518` | F1m：`0.5522` | Acc：`0.5518`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    592    268    240
  true1    150    610    340
  true2    203    252    645
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9776   4750   4174
  true1   2653  10206   5841
  true2   3386   4339  10975
```

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5767`
- Val BalAcc_maj（附报）：`0.5822`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5915`
- BalAcc_maj：`0.5952`
- Acc_majority：`0.5952`
- F1-macro（众数）：`0.5941`
- Recall-macro：`0.5952`
- Recall idle/left/right：`0.4927` / `0.7264` / `0.5664`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5844` | F1m：`0.5837` | Acc：`0.5844`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    542    340    218
  true1     66    799    235
  true2    107    370    623
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9065   5825   3810
  true1   1310  13184   4206
  true2   1803   6361  10536
```

#### Fold 2

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5459`
- Val BalAcc_maj（附报）：`0.5515`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5606`
- BalAcc_maj：`0.5633`
- Acc_majority：`0.5633`
- F1-macro（众数）：`0.5626`
- Recall-macro：`0.5633`
- Recall idle/left/right：`0.5518` / `0.6309` / `0.5073`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5539` | F1m：`0.5532` | Acc：`0.5539`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    607    265    228
  true1    235    694    171
  true2    190    352    558
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10187   4556   3957
  true1   4165  11526   3009
  true2   3345   5995   9360
```

#### Fold 3

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6193`
- Val BalAcc_maj（附报）：`0.6267`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5912`
- BalAcc_maj：`0.5976`
- Acc_majority：`0.5976`
- F1-macro（众数）：`0.5965`
- Recall-macro：`0.5976`
- Recall idle/left/right：`0.5118` / `0.5918` / `0.6891`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5885` | F1m：`0.5875` | Acc：`0.5885`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    563    282    255
  true1    142    651    307
  true2    104    238    758
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9461   4855   4384
  true1   2459  10851   5390
  true2   1918   4078  12704
```

#### Fold 4

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.5126`
- Val BalAcc_maj（附报）：`0.5204`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5717`
- BalAcc_maj：`0.5800`
- Acc_majority：`0.5800`
- F1-macro（众数）：`0.5780`
- Recall-macro：`0.5800`
- Recall idle/left/right：`0.6850` / `0.5250` / `0.5300`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5693` | F1m：`0.5674` | Acc：`0.5693`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    685    137    178
  true1    264    525    211
  true2    280    190    530
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11395   2471   3134
  true1   4594   8764   3642
  true2   4880   3244   8876
```
