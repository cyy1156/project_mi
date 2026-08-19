# C2b · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_212906_C2b`
- Test Acc_paper: `0.5695 ± 0.0224`
- Test BalAcc_maj: `0.5748 ± 0.0213`
- Test win F1: `0.5620 ± 0.0211`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5696`
- Val BalAcc_maj（附报）：`0.5744`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5485`
- BalAcc_maj：`0.5539`
- Acc_majority：`0.5539`
- F1-macro（众数）：`0.5517`
- Recall-macro：`0.5539`
- Recall idle/left/right：`0.5100` / `0.4545` / `0.6973`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5438` | F1m：`0.5413` | Acc：`0.5438`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    561    204    335
  true1    130    500    470
  true2    160    173    767
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9407   3576   5717
  true1   2471   8329   7900
  true2   2849   3082  12769
```

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5726`
- Val BalAcc_maj（附报）：`0.5807`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5945`
- BalAcc_maj：`0.5988`
- Acc_majority：`0.5988`
- F1-macro（众数）：`0.5983`
- Recall-macro：`0.5988`
- Recall idle/left/right：`0.5418` / `0.7173` / `0.5373`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5900` | F1m：`0.5894` | Acc：`0.5900`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    596    335    169
  true1    110    789    201
  true2    136    373    591
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9961   5637   3102
  true1   2034  13114   3552
  true2   2432   6246  10022
```

#### Fold 2

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5467`
- Val BalAcc_maj（附报）：`0.5548`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5439`
- BalAcc_maj：`0.5518`
- Acc_majority：`0.5518`
- F1-macro（众数）：`0.5508`
- Recall-macro：`0.5518`
- Recall idle/left/right：`0.5300` / `0.6355` / `0.4900`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5427` | F1m：`0.5417` | Acc：`0.5427`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    583    302    215
  true1    226    699    175
  true2    187    374    539
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9829   5084   3787
  true1   4026  11585   3089
  true2   3313   6358   9029
```

#### Fold 3

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6122`
- Val BalAcc_maj（附报）：`0.6170`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5970`
- BalAcc_maj：`0.6009`
- Acc_majority：`0.6009`
- F1-macro（众数）：`0.5976`
- Recall-macro：`0.6009`
- Recall idle/left/right：`0.4773` / `0.5891` / `0.7364`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5885` | F1m：`0.5852` | Acc：`0.5885`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    525    288    287
  true1    118    648    334
  true2     87    203    810
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8723   5043   4934
  true1   2162  10888   5650
  true2   1642   3653  13405
```

#### Fold 4

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.5178`
- Val BalAcc_maj（附报）：`0.5252`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5633`
- BalAcc_maj：`0.5683`
- Acc_majority：`0.5683`
- F1-macro（众数）：`0.5642`
- Recall-macro：`0.5683`
- Recall idle/left/right：`0.7080` / `0.5410` / `0.4560`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5559` | F1m：`0.5525` | Acc：`0.5559`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    708    156    136
  true1    302    541    157
  true2    322    222    456
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11621   2787   2592
  true1   5215   8871   2914
  true2   5416   3727   7857
```
