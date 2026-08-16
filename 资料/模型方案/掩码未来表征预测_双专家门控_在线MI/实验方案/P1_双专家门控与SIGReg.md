# 方案 P1：双专家 + Gate + SIGReg（无 Decoder 阶梯）

## 1. 目的

在打开 Decoder 之前，先验证：预测未来表征 + 双专家门控 + SIGReg。  
**不是**训练定稿主结果（主结果见 P2 / `flowchart_A_train.png`）。

## 2. 优先级

**必做**（阶梯 + B 系列锚点）。仅 Three。

## 3. 开关

| 模块 | 训练 | 推理 |
|------|------|------|
| 自写 Encoder + §3.2.1 | 开 | 开 |
| X_full no_grad target | 开 | 关 |
| Predictor | 开 | 开 |
| Expert_cur / future | **`D→64→C`** | 开 |
| Gate | **`Linear(2D→64→1)`，仅两 z** | 开 |
| SIGReg | 开（LeJEPA，slices=1024） | 关 |
| Decoder | **关** | 关 |

## 4. 冻结

| 项 | 值 |
|----|-----|
| L_cls | CE(cur)+CE(final) |
| λ_pred / λ_sig / λ_dec | 1.0 / 0.05 / **0** |
| D | 40（无 128） |
| 优化 | Adam；256/512；patience 20 |
| p_future 标签 | 同 trial 弱监督 |

## 5. 损失

`λ_cls*(CE_cur+CE_final) + λ_pred*L_pred + λ_sig*L_SIGReg`

## 6. 结果

| 项 | 值 |
|----|-----|
| Acc_paper | |
| vs P0 / A1 | |
