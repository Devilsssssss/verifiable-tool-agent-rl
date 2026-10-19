# Return Boundary v1 Results

更新日期：2026-10-19

## 结论

`return-boundary-v1` 是一次监督数据分布对照，不是 RL 或 GRPO 实验。它保留
300 个训练任务和 900 个训练视图，只改变退货目标年龄、干扰订单年龄和目标订单位置，
用于检查 v10 复验中集中的退货边界失败。

两个训练 seed 均完成 113 个优化步，使用相同的 Qwen3-1.7B + LoRA 配置和锁定的训练依赖。
结果显示边界数据会改变退货策略，但训练 seed 之间差异很大，不能解释为稳定收益：

| seed | 新退货 holdout | 旧 v10 确认集回归 | holdout 目标检查率 | 回归目标检查率 |
|---:|---:|---:|---:|---:|
| 45 | 78/120 (65.0%) | 87/120 (72.5%) | 111/120 (92.5%) | 108/120 (90.0%) |
| 46 | 98/120 (81.7%) | 103/120 (85.8%) | 108/120 (90.0%) | 111/120 (92.5%) |

旧确认集配对结果进一步说明 seed sensitivity：seed45 相对匹配的 v10 基线
`105/120` 净下降 18 个任务（gained/lost = `5/23`，bootstrap 95% CI
`[-0.2333, -0.0667]`）；seed46 相对 `101/120` 只增加 2 个任务
(`17/15`，CI `[-0.0750, 0.1083]`）。因此不能把 seed46 的小幅净增写成稳定泛化提升。

## Holdout 分层

| seed | 期限内退货 | 过期退货 | 位置 1 | 位置 2 |
|---:|---:|---:|---:|---:|
| 45 | 57/60 | 21/60 | 41/60 | 37/60 |
| 46 | 54/60 | 44/60 | 60/60 | 38/60 |

seed45 的过期退货仍是主要失败来源，且训练后旧确认集回归明显恶化。seed46 在过期
退货上改善较大，但位置 2 只有 `14/30`，位置 1 为 `30/30`；holdout 的位置差为
`-36.7` 个百分点。年龄单元格和完整业务族统计保存在对应的 `holdout-analysis`
摘要中，不把开发集中的年龄 1/60 单元格误写成训练边界单元格结果。

## 审计与复现

- 数据审计：`artifacts/return-boundary-v1-training-audit.summary.json`
- 训练摘要：`artifacts/qwen3-sft-return-boundary-v1-seed45-training.summary.json`、
  `artifacts/qwen3-sft-return-boundary-v1-seed46-training.summary.json`
- 每个 seed 的流水线状态：`artifacts/return-boundary-v1-seed{45,46}-pipeline.summary.json`
- holdout 分析、轨迹重放和 paired 摘要：`artifacts/return-boundary-v1-seed{45,46}-holdout-*`
- 旧确认集回归、轨迹重放和 paired 摘要：`artifacts/return-boundary-v1-seed{45,46}-regression-*`

两组流水线都完成了模型生成、strict-v2 评分、离线轨迹重放和配对分析；重放摘要报告
`replay_verified=120` 且 `original_scores_unchanged=true`。

## 决策

边界对照完成，但目标订单检查门槛仍未解决，且一个 seed 产生显著回归、另一个 seed
只产生不确定的小幅变化。因此当前不启动 GRPO。下一步若继续，应先明确目标检查和位置
2 的预注册门槛，再设计独立确认集；不能把这次监督分布干预的结果当作 RL 增益证据。
