# experiment_game · 按机位分目录

> **换机 / 新笔记本**：先读 [`../docs/其他电脑移机攻略_20260829.md`](../docs/其他电脑移机攻略_20260829.md)（逐步清单）；背景见 [`../docs/换机最小运行与可移植性重构建议_20260828.md`](../docs/换机最小运行与可移植性重构建议_20260828.md)。

| 目录 | 机位 | Python | 说明 |
|------|------|--------|------|
| [`5070_laptop/`](./5070_laptop/) | RTX 5070 Laptop · hostname `cyy` · `D:\MI` | conda **`cyy`** | **本机请用这里的 bat** |
| [`_template/`](./_template/) | 新机复制源 | `%USERPROFILE%` 探测 | 无硬编码用户名 |
| （共享根）`experiment_game/*.bat` | 5060 等 · 常 `D:\cyy\MI` | 仓库根 **`.venv`** | **不要为本机去改这些文件** |

新增机位时：复制 [`_template/`](./_template/)，改 `machine.json` / README / 环境名，勿覆盖共享根入口。  
已验证机位表：[`VERIFIED.md`](./VERIFIED.md)。  
依赖预检：`python -m experiment_game.tools.preflight`（各机位 `check_deps.bat` 同入口）。
