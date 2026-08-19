# B4 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_015651_B4`
- Test Acc_paper: `0.5699 ± 0.0219`
- Test BalAcc_maj: `0.5752 ± 0.0219`
- Test win F1: `0.5667 ± 0.0193`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.5711`
- Val BalAcc_maj（附报）：`0.5752`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5570`
- BalAcc_maj：`0.5636`
- Acc_majority：`0.5636`
- F1-macro（众数）：`0.5639`
- Recall-macro：`0.5636`
- Recall idle/left/right：`0.5755` / `0.5909` / `0.5245`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5560` | F1m：`0.5563` | Acc：`0.5560`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    633    278    189
  true1    166    650    284
  true2    227    296    577
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10410   4871   3419
  true1   2956  10889   4855
  true2   3797   5009   9894
```

#### Fold 1

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5822`
- Val BalAcc_maj（附报）：`0.5863`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5979`
- BalAcc_maj：`0.6027`
- Acc_majority：`0.6027`
- F1-macro（众数）：`0.6037`
- Recall-macro：`0.6027`
- Recall idle/left/right：`0.5564` / `0.6627` / `0.5891`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5945` | F1m：`0.5951` | Acc：`0.5945`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    612    278    210
  true1    102    729    269
  true2    141    311    648
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10196   4829   3675
  true1   2024  12248   4428
  true2   2455   5338  10907
```

#### Fold 2

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5404`
- Val BalAcc_maj（附报）：`0.5456`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5358`
- BalAcc_maj：`0.5400`
- Acc_majority：`0.5400`
- F1-macro（众数）：`0.5394`
- Recall-macro：`0.5400`
- Recall idle/left/right：`0.5673` / `0.5773` / `0.4755`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5392` | F1m：`0.5385` | Acc：`0.5392`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    624    257    219
  true1    304    635    161
  true2    209    368    523
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10451   4501   3748
  true1   5012  10927   2761
  true2   3607   6222   8871
```

#### Fold 3

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.6070`
- Val BalAcc_maj（附报）：`0.6133`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5864`
- BalAcc_maj：`0.5912`
- Acc_majority：`0.5912`
- F1-macro（众数）：`0.5883`
- Recall-macro：`0.5912`
- Recall idle/left/right：`0.4682` / `0.6027` / `0.7027`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5812` | F1m：`0.5789` | Acc：`0.5812`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    515    339    246
  true1    131    663    306
  true2     98    229    773
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8751   5504   4445
  true1   2312  11089   5299
  true2   1755   4180  12765
```

#### Fold 4

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.5085`
- Val BalAcc_maj（附报）：`0.5167`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5727`
- BalAcc_maj：`0.5787`
- Acc_majority：`0.5787`
- F1-macro（众数）：`0.5756`
- Recall-macro：`0.5787`
- Recall idle/left/right：`0.6910` / `0.5740` / `0.4710`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5671` | F1m：`0.5646` | Acc：`0.5671`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    691    173    136
  true1    276    574    150
  true2    308    221    471
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11395   3031   2574
  true1   4821   9508   2671
  true2   5159   3822   8019
```
