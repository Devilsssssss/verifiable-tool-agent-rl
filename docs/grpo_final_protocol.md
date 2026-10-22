# 最终 GRPO 对照协议

状态：**协议已冻结，训练与评测已完成。** 这是一轮有明确停止条件的最终 RL 对照；结果未达到预注册门槛，因此不宣称 GRPO 增益。

## 研究问题

在已经通过审计的 `return-boundary-v1` SFT adapter 上，加入包含终态、目标订单检查和业务政策的 GRPO 奖励，是否能在全新冻结的退货 holdout 上稳定超过匹配的 SFT baseline？

## 固定条件

- 基座：`Qwen3-1.7B`，初始化为 `return-boundary-v1` seed45/seed46 的匹配 SFT adapter；
- 训练任务：`data/return-boundary-v1/tasks_train.jsonl`，300 个任务，900 个 SFT 视图对应的已审计业务分布；
- GRPO 训练任务：由 `scripts/create_final_grpo_data.py` 生成，严格使用 `strict-v3-reason-code` 和 `strict_shaped`；
- 两个 RL seed：45、46；其它学习率、generation 数、训练步数和解码预算固定，不根据中间结果挑选 checkpoint；
- 评测：全新任务实体、ID、措辞和退货年龄的冻结 holdout，另保留旧确认集作为回归检查；
- 主要指标：严格成功率；诊断指标：终态满足、目标订单检查、政策合规、理由码正确、工具失败和奖励组内方差。

## 奖励定义

`strict_shaped` 只作为训练信号，报告仍以 `success` 为唯一主指标：

- 严格成功：`+1.0`；
- 正确终态：`+0.35`；
- 检查目标订单：`+0.25`；
- 无政策违规：`+0.20`，有违规：`-0.50`；
- 未成功时按正确前缀给最多 `+0.20` 进度分；
- 超出效率预算扣 `0.05`。

因此“终态正确但没有检查目标订单”或“终态正确但有退货政策违规”不能获得严格成功奖励。`strict-v3-reason-code` 还要求过期退货和已发货订单的升级理由码匹配。

## 冻结与停止条件

1. 先生成并审计新 holdout，再运行任何候选 adapter；holdout 文件哈希进入结果报告。
2. 先做匹配 SFT baseline 的两个 seed 评测，再训练两个 GRPO seed；不能用 holdout 选模型。
3. 任一 seed 的奖励组内方差长期为零，或训练后严格成功率没有超过匹配 SFT baseline，均记录为 RL 阴性，不增加训练步数。
4. 只有两个 seed 都在新 holdout 上超过匹配 SFT baseline，且目标订单检查率和政策合规率均达到 100%，才允许写“本环境下的 RL 增益”；否则结论为未证实。

## 最终结果

新冻结 holdout 共 120 条，使用 `strict-v3-reason-code` 与 `strict_shaped`，GRPO 与对应 SFT 起点逐 seed 配对：

| seed | 匹配 SFT 严格成功 | GRPO 严格成功 | SFT 目标检查 / 合规 | GRPO 目标检查 / 合规 | 判断 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 45 | 56/120 (46.7%) | 56/120 (46.7%) | 91.7% / 47.5% | 91.7% / 46.7% | 未超过 baseline |
| 46 | 52/120 (43.3%) | 51/120 (42.5%) | 87.5% / 50.0% | 86.7% / 50.0% | 低于 baseline |

旧 `position-confirmation` 回归集上，GRPO seed45/46 分别为 72/120 (60.0%) 和 71/120 (59.2%)；它只用于回归，不是盲测证据。训练过程中多次出现组内奖励方差为 0，说明本轮奖励在实际 rollout 上没有提供稳定区分度。

**结论：本轮 RL 增益未证实，停止继续增加 GRPO 步数或挑选 checkpoint。** 这不是训练脚本失败，而是两个 seed 在新确认集上都没有超过匹配 SFT，且目标检查率、政策合规率均未达到 100%。逐条失败归因见 [grpo_failure_analysis.md](grpo_failure_analysis.md)：四份 run 的 240 条过期退货任务全部失败，主因是 strict-v3 的理由码没有在初始化 SFT 的可观测输入中显式提供，工具拒绝后又反复调用。下一步应先修正输入/动作契约、退货位置 2 和目标订单检查，重新做双 seed SFT；不应直接进入新的 RL 调参。

## 复现入口

```powershell
python scripts/create_final_grpo_data.py
python scripts/create_return_holdout.py `
  --seed 20261030 --id-offset 70000000 `
  --output data/grpo-final-v1/holdout/tasks_test.jsonl `
  --summary artifacts/grpo-final-v1-holdout-audit.summary.json `
  --heldout data/position-confirmation/tasks_test.jsonl
```

服务器训练命令、adapter 哈希、原始轨迹和逐 seed 评测结果写入服务器 `artifacts/grpo-final-v1-*`；已拉回并纳入版本控制的聚合摘要包括 `*-holdout.summary.json` 和 `*-confirmation.summary.json`。模型权重和逐条轨迹不进入 Git。
