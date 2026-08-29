# experiment_game · 按机位分目录

> **换机 / 新笔记本**：先读 [`../docs/换机最小运行与可移植性重构建议_20260828.md`](../docs/换机最小运行与可移植性重构建议_20260828.md)（零改业务代码清单 + 重构建议）。

| 目录 | 机位 | Python | 说明 |
|------|------|--------|------|
| [`5070_laptop/`](./5070_laptop/) | RTX 5070 Laptop · hostname `cyy` · `D:\MI` | conda **`cyy`** | **本机请用这里的 bat** |
| （共享根）`experiment_game/*.bat` | 5060 等 · 常 `D:\cyy\MI` | 仓库根 **`.venv`** | **不要为本机去改这些文件** |

新增机位时：复制 `5070_laptop/`，改 `machine.json` / README / `_resolve_python.bat`，勿覆盖共享根入口。
