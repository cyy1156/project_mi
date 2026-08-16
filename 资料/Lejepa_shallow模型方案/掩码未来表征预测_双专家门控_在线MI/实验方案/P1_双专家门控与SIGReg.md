# 方案 P1：双专家 + Gate + SIGReg（主结果）

## 1. 目的

完整方法主线（**不含 Decoder**）：可见表征预测未来 + 双专家概率门控 + SIGReg。

## 2. 优先级

**必做**；论文主表报本方案（及经 L1 回填的「当前采用」）。P2 为另一主结果行。

## 3. 模块开关

| 模块 | 训练 | 推理 |
|------|------|------|
| Encoder 共享（**自写** `shallowfbcsp` + `forward_features`） | 开 | 开 |
| X_full + **no_grad** target | 开 | 关 |
| Predictor | 开 | 开 |
| Expert_cur / Expert_future | 开 | 开 |
| Gate（**仅两 z**） | 开 | 开 |
| SIGReg | 开 | 关（不算损失） |
| Decoder | **关** | 关 |

## 4. 冻结开跑

| 项 | 值 |
|----|-----|
| 读出 | **segment_mean**（§3.2.1；开跑 D=40；L1 必做 `40→128` 消融） |
| Predictor | D→2D→2D→D |
| Expert | D→64→C，两头不共享 |
| Gate 输入 | **仅两个 z** |
| 分类损失 | **CE(cur)+CE(final)**（非三项） |
| SIGReg | LeJEPA 官方，`num_slices=1024`（§3.7） |
| λ_pred / λ_sig / λ_dec | 1.0 / 0.05 / **0** |
| mask | zero |
| 优化 | Adam, lr=1e-4；batch 256/512；patience 20 |
| 头任务 | Task + Three |

必做变体见 B 系列与 L1（Gate+z+p、+CE(future)、λ 网格等）。

## 5. 损失（冻结）

`λ_cls*(CE_cur+CE_final) + λ_pred*L_pred + λ_sig*L_SIGReg`

## 6. 推理

`p_final = α p_cur + (1-α) p_future`，`ŷ=argmax(p_final)`。

## 7. 验收 / 读法

- 相对 A1、A2/P0 是否稳定提升（**Acc_paper**）
- α 是否塌成常数
- 与 B 系列交叉验证故事

## 8. 当前采用（跑完 L1/消融后填写）

| 项 | 值 |
|----|-----|
| 读出 / Gate / 损失组合 | |
| λ_pred / λ_sig | |
| Acc_paper mean±std | |
