# 合并库合训 · 方案 A 实验资料夹

> 日期：2026-07-27  
> 状态：**方案 A 代码已落地**；合并产物已生成；过夜待跑  
> 目的：冻结方案 A，并归档今早单库过夜对照。

## 本夹文件

| 文件 | 内容 |
|------|------|
| [技术方案A_简单拼接合并合训.md](./技术方案A_简单拼接合并合训.md) | 方案 A 设计 + §9 落地路径 |
| [20260727_上午单库过夜实验结果.md](./20260727_上午单库过夜实验结果.md) | 今早 BCI2a / Stieger 过夜汇总 |

## 已落地命令

```powershell
cd D:\cyy\MI\code\preprocess_lab
$env:PYTHONPATH='.'
D:\cyy\MI\.venv\Scripts\python.exe -m src.datasets.merge_bci2a_stieger

cd D:\cyy\MI\code\train_lab\src\step
D:\cyy\MI\.venv\Scripts\python.exe run_overnight_kfold.py
```

当前合并库：`out/merged_2s/`（约 `(31089,1,8,500)`，22 人）。
