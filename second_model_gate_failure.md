# 第二模型（π0）：预注册门槛未通过

**结论**：π0 的两次 LoRA 微调（分别从 `pi0_base` 和 `pi0_libero` 初始化）以及一次使用
base 归一化统计的变体，均未达到预注册的英语能力门槛（每组 `en` 与 `en_para` ≥ 70%）。
因此，第二模型的 420 回合语言对比**不进入主结果**；论文保持单模型案例研究定位。
本文件记录完整的负结果、控制实验与复现路径。

## 1. 预注册设计（与主实验逐格配对）

- 2 组（akita 双碗交付、soda 汽水交付）× 3 个任务层级 × 5 个初始状态（0–4）= 30 个单元；
- 7 个条件（`en, en_para, es, zh, ar, hi, sw`）× 2 轮 = **420 回合**；
- 指令逐字取自 `instructions_v5.json`；
- 能力门槛：每组 `en` 与 `en_para` ≥ 70%（≥21/30）；未过门槛不解释语言对比。

## 2. 运行记录

| 运行 | 初始化 | 训练 | 评测 | `en` | `en_para` | 结论 |
|---|---|---|---|---:|---:|---|
| A | `pi0_base` | LoRA 30k | 420 回合 | 0/60 | 0/60 | 全部失败 |
| B | `pi0_libero` | LoRA 30k | 420 回合 | 0/60 | 0/60 | 全部失败 |
| C | `pi0_libero` + base norm stats | LoRA 30k | 7 回合 smoke | 0/7 | — | 门槛未过，未跑 420 |

- 运行 A 的 Step 0 loss = 0.1219；运行 B 的 Step 0 loss = 0.1513、param_norm = 1384.93，
  确认 `pi0_libero` 初始化已加载。
- 运行 B 的 420 回合完整分析：`second_model_results_pi0libero_v2/analysis/report.md`，
  每组、每条件均为 0/30。
- 另有一次 from-base 重复运行（因脚本未透传 `PI0_BASE_PARAMS`，实验名误写为
  `_pi0libero`），结果与 A 相同，未用于分析。

## 3. 控制实验：微调破坏了 base 能力

标准 LIBERO `libero_spatial`，每任务 1 回合：

| 策略 | 成功 |
|---|---:|
| 未微调的 `pi0_libero` | **7/10** |
| 运行 B（`pi0_libero` + LoRA 30k） | **0/10** |
| 运行 C（base norm stats） | **0/10** |

未微调的 π0 检查点在标准 LIBERO 上正常，说明 **π0 的 serving 管线、动作变换与客户端正确**；
LoRA 在 LIBERO-Safety 数据上微调 30k 后，标准 LIBERO 能力从 70% 降到 0%。换用 base 的
norm stats 也没有恢复，说明问题不是归一化尺度，而是 LoRA 适配本身把动作头带偏。

## 4. 为什么不上报「语言效应未复现」

预注册规定：未过能力门槛时，`en` 为 0 不能解释为「非英语退化消失」或「效应不存在」。
本轮 π0 的英语任务本身失败，语言对比的分母没有意义。我们的结论是：

> 在该 LoRA 配方下，π0 没有达到可解释的语言对比所需的能力门槛；因此不把本轮结果
> 作为对句框效应的跨模型检验。

## 5. 复现

```bash
# 训练（运行 B）
cd /home/sang/offline-bundle/second_model/config
EXP=pi0_safety_20261023_pi0libero_v2
PI0_BASE_PARAMS=/openpi_assets/checkpoints/pi0_libero/params \
  ./train_pi0.sh train "$EXP"

# 评测（420）
cd /home/sang/offline-bundle/second_model/eval
OUT_DIR=/home/sang/offline-bundle/second_model_results_pi0libero_v2 \
  ./run_420.sh run

# 分析
cd /home/sang/offline-bundle/second_model/analysis
python merge_analysis.py \
  --pi0-results /home/sang/offline-bundle/second_model_results_pi0libero_v2 \
  --out /home/sang/offline-bundle/second_model_results_pi0libero_v2/analysis
```

## 6. 文件

- `second_model_results/`：运行 A（from-base 30k）的 420 回合；
- `second_model_results_pi0libero_v2/`：运行 B（`pi0_libero` + LoRA 30k）的 420 回合与 `analysis/`；
- `second_model_results_basestats/`：运行 C 的 7 回合 smoke；
- `second_model/预注册_第二模型.md`：预注册、偏离记录与门槛规则；
- `second_model/README.md`：训练、评测与故障排查。

## 7. 对论文的意义

- 主发现（π0.5 上的句框绑定）仍是**单策略案例研究**；本文不主张跨架构普适性。
- 预注册门槛发挥了作用：它阻止了把「模型没学会任务」误读为「语言效应未复现」。
- 该负结果对读者有用：它说明在 LIBERO-Safety 上做第二模型时，LoRA 微调可能破坏
  base 的精确动作能力；若要复现，建议预留能力门槛检查，并优先考虑全量微调或
  更小的 LoRA 学习率/秩。
