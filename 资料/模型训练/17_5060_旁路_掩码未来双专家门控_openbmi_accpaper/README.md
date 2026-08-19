# 17 · 5060 旁路 · 掩码未来双专家门控 · OpenBMI Acc_paper



> 方案正文：[`资料/模型方案/掩码未来表征预测_双专家门控_在线MI/`](../../模型方案/掩码未来表征预测_双专家门控_在线MI/)  

> 代码包：[`code/train_lab/src/step/5060_mask_future_dual_expert_accpaper/`](../../../code/train_lab/src/step/5060_mask_future_dual_expert_accpaper/)  

> **全量五折亦可在本机跑**（建议 mem_guard / 少开后台）；5090 姊妹包见 [`../17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/`](../17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/)  

> **正式结果表**：[`阶段结果登记_主线五折.md`](./阶段结果登记_主线五折.md)  
> **详细结果报告**：[`资料/Lejepa_shallow模型方案/.../实验报告_主线与U系列_5060_OpenBMI_AccPaper.md`](../../Lejepa_shallow模型方案/掩码未来表征预测_双专家门控_在线MI/实验报告_主线与U系列_5060_OpenBMI_AccPaper.md)



## 设备



NVIDIA RTX 5060 Laptop · ~16GB RAM · 低内存默认（batch 128/256 · workers=0）



## 一键启动



```powershell

cd code/train_lab/src/step/5060_mask_future_dual_expert_accpaper

python _smoke_local.py

python chain_all.py

# 或 run_chain_detached.bat

```



A1+ 前置数据（三类 · protocol_version≥3）：



```powershell

cd code/preprocess_lab

python -m src.datasets.openbmi_pf1000.batch --reset

# 输出 out/openbmi_2s_hop100_pf1000/

```



默认链：`A0_ref → A0 → A1 → P0 → A2 → P1 → P2`。  

完整 B/C：`python chain_all.py --full-chain`。  

后续结构升级（U1–U3）见实验方案；**五折已完成**，详报见上「详细结果报告」。



## 结果登记（主线 + U · 五折 · 2026-08-19）



| 臂 | Test Acc_paper mean±std | run 路径 | 备注 |

|----|-------------------------|----------|------|

| A0_ref | 0.5403±0.0256 | `shallow_A0_ref_.../run_20260817_133104/three` | braindecode |

| A0 | 0.5342±0.0242 | `20260817_151010_A0` | 自写 500pt |

| A1 | 0.5717±0.0236 | `20260818_015137_A1` | |

| P0 | 0.5672±0.0230 | `20260818_050022_P0` | |

| A2 | 0.5656±0.0239 | `20260818_072842_A2` | |

| P1 | 0.5643±0.0217 | `20260818_092335_P1` | |

| **P2（主）** | **0.5707±0.0112** | **`20260818_133939_P2`** | 定稿+Decoder |

| U1 | 0.5722±0.0180 | `20260818_194132_U1` | vs P2 +0.15 pp |

| U3 | 0.5708±0.0180 | `20260818_235303_U3` | vs P2 ≈0 |

| U2 | 0.5665±0.0236 | `20260819_034744_U2` | vs P2 −0.42 pp |



逐折与附报见 [`阶段结果登记_主线五折.md`](./阶段结果登记_主线五折.md)。



权重根目录：`code/train_lab/out/5060_mask_future_dual_expert_accpaper/`


