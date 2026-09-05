# Stieger 目录 — 实现状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `batch.py` | **已实现** | 对照 `资料/数据集说明/Stieger2021_预处理流程与示例代码.md` §5.8；输出默认 `out/stieger_2s`（2s/500） |

运行（在 `code/preprocess_lab` 下）：

```text
python -m src.datasets.stieger.batch
# 或指定
python -m src.datasets.stieger.batch --glob "D:/.../DATA/stieger/S*_Session_*.mat" --out out/stieger_2s
```
