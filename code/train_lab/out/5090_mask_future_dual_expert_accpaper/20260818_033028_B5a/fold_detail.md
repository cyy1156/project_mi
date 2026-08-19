# B5a · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_033028_B5a`
- Test Acc_paper: `0.5648 ± 0.0221`
- Test BalAcc_maj: `0.5709 ± 0.0220`
- Test win F1: `0.5609 ± 0.0206`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5796`
- Val BalAcc_maj（附报）：`0.5867`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5433`
- BalAcc_maj：`0.5479`
- Acc_majority：`0.5479`
- F1-macro（众数）：`0.5481`
- Recall-macro：`0.5479`
- Recall idle/left/right：`0.5318` / `0.4982` / `0.6136`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5453` | F1m：`0.5453` | Acc：`0.5453`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    585    229    286
  true1    149    548    403
  true2    209    216    675
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9732   4148   4820
  true1   2689   9355   6656
  true2   3302   3891  11507
```

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5785`
- Val BalAcc_maj（附报）：`0.5819`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5964`
- BalAcc_maj：`0.6018`
- Acc_majority：`0.6018`
- F1-macro（众数）：`0.6026`
- Recall-macro：`0.6018`
- Recall idle/left/right：`0.5282` / `0.6900` / `0.5873`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5902` | F1m：`0.5905` | Acc：`0.5902`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    581    300    219
  true1     76    759    265
  true2    115    339    646
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9589   5099   4012
  true1   1499  12721   4480
  true2   2069   5828  10803
```

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5419`
- Val BalAcc_maj（附报）：`0.5474`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5370`
- BalAcc_maj：`0.5442`
- Acc_majority：`0.5442`
- F1-macro（众数）：`0.5426`
- Recall-macro：`0.5442`
- Recall idle/left/right：`0.4900` / `0.6482` / `0.4945`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5343` | F1m：`0.5327` | Acc：`0.5343`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    539    316    245
  true1    217    713    170
  true2    163    393    544
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9001   5530   4169
  true1   3876  11883   2941
  true2   2876   6733   9091
```

#### Fold 3

- stopped_epoch：`90` | best_epoch：`70`
- Val Acc_paper（早停）：`0.6111`
- Val BalAcc_maj（附报）：`0.6167`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5785`
- BalAcc_maj：`0.5852`
- Acc_majority：`0.5852`
- F1-macro（众数）：`0.5842`
- Recall-macro：`0.5852`
- Recall idle/left/right：`0.5100` / `0.5918` / `0.6536`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5759` | F1m：`0.5752` | Acc：`0.5759`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    561    303    236
  true1    167    651    282
  true2    144    237    719
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9530   5123   4047
  true1   3033  10801   4866
  true2   2471   4253  11976
```

#### Fold 4

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5096`
- Val BalAcc_maj（附报）：`0.5170`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5690`
- BalAcc_maj：`0.5753`
- Acc_majority：`0.5753`
- F1-macro（众数）：`0.5716`
- Recall-macro：`0.5753`
- Recall idle/left/right：`0.7190` / `0.5530` / `0.4540`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5645` | F1m：`0.5608` | Acc：`0.5645`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    719    159    122
  true1    327    553    120
  true2    341    205    454
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11926   2834   2240
  true1   5490   9213   2297
  true2   5801   3551   7648
```
