# 消融 B 系列（相对 P1）

> 基线：P1「当前采用」。原则：**一次只改一个开关**。  
> 优先级：**B1–B10 全部必做**；仅 Three。

## 统一说明

- 协议：定稿第 15 节 / [`协议_滑窗投票与Acc_paper.md`](./协议_滑窗投票与Acc_paper.md)
- 指标：**Acc_paper** 五折 mean±std，附报 vs P1 Δ

---

## B1 w/o L_pred（必做）

- 开关：`λ_pred=0`（可保留空跑 Predictor 结构，注明）
- 目的：预测损失是否必要

## B2 w/o no_grad（必做；可称 w/o stop-grad）

- 开关：对 `X_full` **去掉** `torch.no_grad()`，让 target 路可回传（仍可对 `z_target` 做 detach，与默认对照）
- 目的：是否崩训练 / Acc 变化

## B3 w/o SIGReg（必做）

- 开关：`λ_sig=0`
- 目的：各向同性正则

## B4 w/o Expert_future（必做）

- 开关：训练与推理只用 `p_cur`（可关未来头）
- 目的：未来专家贡献

## B5 w/o 可学习 Gate（必做）

- 开关：固定 `α=1.0` **与** `α=0.5`（两个子实验都跑）
- 目的：门控是否优于固定融合

## B6 + CE(p_future)（必做）

- 开关：在 P1 默认 `CE(cur)+CE(final)` 上**追加** `CE(p_future)`（三项）
- 目的：未来弱监督是否有益（相对默认两项，不是「去掉」）

## B7 w/o CE(p_final)（必做）

- 开关：不直接监督融合概率
- 目的：Gate 是否需要 final CE

## B8 learnable mask token（必做）

- 开关：`mask_mode=learnable_token`
- 目的：对比零填

## B9 泄漏上限（必做；上限参考）

- 开关：评估支路偷看 `X_full`
- 目的：上界参考；主表不注为公平结果

## B10 EMA target（必做）

- 开关：target 用 EMA Encoder
- 目的：对照 share+sg

---

## 结果总表

| ID | Acc_paper mean±std | vs P1 Δ | 备注 |
|----|--------------------|---------|------|
| P1 | | 0 | 基线 |
| B1 | | | |
| B2 | | | |
| B3 | | | |
| B4 | | | |
| B5 α=1 | | | |
| B5 α=0.5 | | | |
| B6 +CE(fut) | | | |
| B7 | | | |
| B8 | | | |
| B9 | | | 上界 |
| B10 | | | |
