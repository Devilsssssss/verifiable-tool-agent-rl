# 实验索引与决策记录

这份索引用于快速理解项目为什么逐步停止某些路线，以及每个结论对应哪份协议和结果。

| 阶段 | 核心问题 | 结果/决策 | 入口 |
|---|---|---|---|
| 环境与评分 | 工具调用是否可执行、奖励是否可重放 | SQLite 状态、10 工具、strict-v2 已冻结 | `docs/format_control.md`、`docs/project_qa.md` |
| v8 位置探针 | 模型是否只会选第一项订单 | 60/120，成功全在位置 1；暴露捷径 | `docs/order_position_probe.md` |
| v10 位置平衡 SFT | 训练分布能否缓解位置捷径 | seed45/46/47 有改善，但目标检查稳定性门槛未过 | `docs/position_v10_protocol.md`、`docs/position_v10_seed_replication_protocol.md` |
| return-v11 | 退货年龄分布干预是否稳定泛化 | seed-sensitive，旧确认集回归；不作为 RL 证据 | `docs/return_v11_results.md` |
| return-boundary-v1 | 退货临界天数是否是主要失败来源 | holdout 提升不稳定，位置 2 和目标检查仍失败 | `docs/return_boundary_training_protocol.md`、`docs/return_boundary_v1_results.md` |
| GRPO gate | 是否满足进入 RL 的前置条件 | 未满足，GRPO 暂缓 | `docs/project_summary.md`“当前结论的边界” |

## 当前状态

已授权的监督诊断和边界对照均已完成。当前没有应被包装成“已解决”的开放实验；如果继续，应该先冻结独立确认集，明确目标订单检查和位置 2 的成功门槛，再设计下一轮数据或 RL 对照。不要用专家演示、旧确认集或评分器修订替代模型结果。

## 面试讲法

先用[项目简介](project_brief.md)讲问题和结论，再用[项目总结](project_summary.md)讲证据，最后打开具体协议和结果证明审计链路。最稳妥的表述是：完成了可验证环境、严格评分、策略捷径诊断和多 seed 监督对照，并据结果阻止了没有前置证据的 GRPO 实验。
