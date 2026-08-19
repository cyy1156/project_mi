# B6 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_064547_B6`
- Test Acc_paper: `0.5695 ± 0.0196`
- Test BalAcc_maj: `0.5752 ± 0.0195`
- Test win F1: `0.5633 ± 0.0181`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5707`
- Val BalAcc_maj（附报）：`0.5763`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5594`
- BalAcc_maj：`0.5667`
- Acc_majority：`0.5667`
- F1-macro（众数）：`0.5673`
- Recall-macro：`0.5667`
- Recall idle/left/right：`0.5500` / `0.5545` / `0.5955`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5538` | F1m：`0.5543` | Acc：`0.5538`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    605    249    246
  true1    150    610    340
  true2    205    240    655
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10089   4332   4279
  true1   2761  10022   5917
  true2   3520   4222  10958
```

#### Fold 1

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.5793`
- Val BalAcc_maj（附报）：`0.5844`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5921`
- BalAcc_maj：`0.5976`
- Acc_majority：`0.5976`
- F1-macro（众数）：`0.5981`
- Recall-macro：`0.5976`
- Recall idle/left/right：`0.5082` / `0.6782` / `0.6064`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5852` | F1m：`0.5854` | Acc：`0.5852`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    559    305    236
  true1     63    746    291
  true2    118    315    667
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9324   5075   4301
  true1   1429  12364   4907
  true2   2033   5527  11140
```

#### Fold 2

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5467`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5412`
- BalAcc_maj：`0.5467`
- Acc_majority：`0.5467`
- F1-macro（众数）：`0.5451`
- Recall-macro：`0.5467`
- Recall idle/left/right：`0.5791` / `0.6036` / `0.4573`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5426` | F1m：`0.5415` | Acc：`0.5426`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    637    267    196
  true1    268    664    168
  true2    230    367    503
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10575   4643   3482
  true1   4543  11168   2989
  true2   3696   6306   8698
```

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6022`
- Val BalAcc_maj（附报）：`0.6067`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5912`
- BalAcc_maj：`0.5970`
- Acc_majority：`0.5970`
- F1-macro（众数）：`0.5946`
- Recall-macro：`0.5970`
- Recall idle/left/right：`0.4782` / `0.6173` / `0.6955`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5866` | F1m：`0.5843` | Acc：`0.5866`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    526    327    247
  true1    125    679    296
  true2     98    237    765
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8789   5567   4344
  true1   2271  11357   5072
  true2   1700   4238  12762
```

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5174`
- Val BalAcc_maj（附报）：`0.5248`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5637`
- BalAcc_maj：`0.5683`
- Acc_majority：`0.5683`
- F1-macro（众数）：`0.5646`
- Recall-macro：`0.5683`
- Recall idle/left/right：`0.7210` / `0.5030` / `0.4810`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5551` | F1m：`0.5510` | Acc：`0.5551`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    721    147    132
  true1    331    503    166
  true2    343    176    481
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12064   2495   2441
  true1   5802   8202   2996
  true2   5944   3013   8043
```
