# C2a · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_200842_C2a`
- Test Acc_paper: `0.5758 ± 0.0204`
- Test BalAcc_maj: `0.5818 ± 0.0203`
- Test win F1: `0.5700 ± 0.0175`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5778`
- Val BalAcc_maj（附报）：`0.5841`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5506`
- BalAcc_maj：`0.5558`
- Acc_majority：`0.5558`
- F1-macro（众数）：`0.5564`
- Recall-macro：`0.5558`
- Recall idle/left/right：`0.5500` / `0.5536` / `0.5636`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5525` | F1m：`0.5532` | Acc：`0.5525`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    605    260    235
  true1    161    609    330
  true2    215    265    620
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10118   4572   4010
  true1   2897  10343   5460
  true2   3512   4651  10537
```

#### Fold 1

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5848`
- Val BalAcc_maj（附报）：`0.5922`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6052`
- BalAcc_maj：`0.6100`
- Acc_majority：`0.6100`
- F1-macro（众数）：`0.6102`
- Recall-macro：`0.6100`
- Recall idle/left/right：`0.5300` / `0.7427` / `0.5573`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5973` | F1m：`0.5972` | Acc：`0.5973`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    583    346    171
  true1     68    817    215
  true2    100    387    613
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9694   5810   3196
  true1   1360  13504   3836
  true2   1898   6489  10313
```

#### Fold 2

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5415`
- Val BalAcc_maj（附报）：`0.5489`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5552`
- BalAcc_maj：`0.5618`
- Acc_majority：`0.5618`
- F1-macro（众数）：`0.5609`
- Recall-macro：`0.5618`
- Recall idle/left/right：`0.5645` / `0.6264` / `0.4945`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5499` | F1m：`0.5491` | Acc：`0.5499`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    621    262    217
  true1    250    689    161
  true2    202    354    544
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10320   4499   3881
  true1   4348  11369   2983
  true2   3478   6064   9158
```

#### Fold 3

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.6052`
- Val BalAcc_maj（附报）：`0.6126`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5867`
- BalAcc_maj：`0.5939`
- Acc_majority：`0.5939`
- F1-macro（众数）：`0.5894`
- Recall-macro：`0.5939`
- Recall idle/left/right：`0.4500` / `0.5809` / `0.7509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5823` | F1m：`0.5783` | Acc：`0.5823`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    495    285    320
  true1     97    639    364
  true2     74    200    826
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8398   4817   5485
  true1   1874  10670   6156
  true2   1461   3640  13599
```

#### Fold 4

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.5096`
- Val BalAcc_maj（附报）：`0.5167`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5817`
- BalAcc_maj：`0.5877`
- Acc_majority：`0.5877`
- F1-macro（众数）：`0.5859`
- Recall-macro：`0.5877`
- Recall idle/left/right：`0.6830` / `0.5310` / `0.5490`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5741` | F1m：`0.5724` | Acc：`0.5741`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    683    166    151
  true1    255    531    214
  true2    265    186    549
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11355   2816   2829
  true1   4403   8803   3794
  true2   4581   3296   9123
```
