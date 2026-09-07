# MI_model 离线训练

## 依赖安装

```bash
.venv\Scripts\python.exe -m pip install -r MI_model\requirements.txt
```

## Stage 1：前后二分类（front / back）

当前唯一质量可靠的二分类为 **前进想象 vs 后退想象**。

- 参与训练：`front1.csv` + `back.csv` 各 **1 个 session**（每类 CSV 数量对齐；`back1.csv` 暂不用）
- 自动排除：文件名以 `_no` 结尾的 CSV（如 `front_no.csv`、`right_no.csv`）

```bash
cd MI_model
python scripts/run_stage1.py
```

或分步：

```bash
python scripts/01_inspect_data.py --stage 1
python scripts/02_build_dataset.py --stage 1
python scripts/03_train_csp_svm.py --stage 1
python scripts/04_eval_offline.py --stage 1
```

## 数据命名约定

| 模式 | 示例 | 含义 |
|------|------|------|
| 正常 session | `back.csv`, `front1.csv` | 参与训练（若属于当前 Stage 类别） |
| 质量标记 | `front_no.csv`, `left1_no.csv` | **永不参与训练** |

## 训练 / 测试划分

- 步骤 2 按 **group** 固定留出 **20%** 为测试集（`test_idx.npy`），避免滑窗泄漏
- 步骤 3 仅在 **80% 训练集** 上 CV + 训练模型
- 步骤 4 仅在 **测试集** 上评估（主指标 `test_set_accuracy`）

## 输出

| 路径 | 内容 |
|------|------|
| `reports/stage1/` | 检查报告、CV、**测试集**混淆矩阵 |
| `dataset/stage1/` | `X/y/groups`, `train_idx`, `test_idx`, `meta.json` |
| `models/stage1/` | `csp.pkl`, `svm.pkl`, `pipeline.pkl` |

## 标签（Stage 1）

| 标签 | 类别 | 想象内容 |
|------|------|----------|
| 0 | front | 持续想象前进 |
| 1 | back | 持续想象后退 |
