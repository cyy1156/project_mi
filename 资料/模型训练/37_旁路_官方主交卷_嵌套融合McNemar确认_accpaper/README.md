# 实验 37 · 资料入口

> 方案：[方案.md](./方案.md)（**v0.2**）  
> 登记：[总结/结果登记表.md](./总结/结果登记表.md)  
> 上游：[../36_旁路_官方主交卷_扩池与跨轨融合_accpaper/](../36_旁路_官方主交卷_扩池与跨轨融合_accpaper/)

## 一句话

对 Exp36 的 M7 做 **嵌套重放**：主读 = 嵌套 Wilcoxon（N7 vs N0）；辅证 = McNemar + 被试 cluster bootstrap；主确认臂固定为双流 N7。

## v0.2 相对 v0.1

1. 嵌套 Wilcoxon **升主检验**（嵌套可能解开 fold0 塌缩 → n=6 地板 0.031）  
2. McNemar 降辅证，并加 **被试级 cluster bootstrap**  
3. 废止「均值最高选臂」→ **先验层级 N7 主确认**

## 跑法（代码落地后）

```powershell
conda activate cyy
cd D:\MI\code\train_lab\src\step\5070_challenge_exp37_nested_mcnemar_accpaper
python run_exp37.py --update-registry
```
