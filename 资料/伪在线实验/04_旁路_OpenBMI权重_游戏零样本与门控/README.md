# 04 · OpenBMI 正式 shallow · 游戏零样本与门控（旁路）

用 5060 正式 OpenBMI Acc_paper **shallow**（`run_20260807_135828`）在游戏 sub02/sub03 上做：

1. **Q0**：无门控零样本（≡ H0）→ Three 相对 01 BCI2a **有增益**  
2. **Q1**：在线 teachable 门控 H0–H3 → **H1 转阳**（对照 03-P2 BCI2a 阴性）

推理前强制通道重排：游戏序 → OpenBMI 序。  
**状态：建议已接受**——见 [上级「当前部署口径」](../README.md)。主记录：[`总结/结果登记表.md`](总结/结果登记表.md)。

## 文档

| 文件 | 说明 |
|------|------|
| [方案.md](方案.md) | 冻结范围与成功线 |
| [总结/结果登记表.md](总结/结果登记表.md) | 主报登记 |

## 对照

- [01 零样本](../01_不微调_零样本/)（BCI2a）
- [03 质量门控](../03_旁路_teachable质量门控/)（BCI2a-P2 已阴性）

## 跑数

```bash
PY=D:/cyy/MI/.venv/Scripts/python.exe
cd code/train_lab/src/step/game_pseudo_online_hop100
$PY eval_openbmi_game.py --model shallow --gates H0,H1,H2,H3
```

产物：`results/`（不写 01/03）。
