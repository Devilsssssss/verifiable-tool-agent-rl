# 实验索引与决策记录

这份索引用于快速理解项目为什么逐步停止某些路线，以及每个结论对应哪份协议和结果。

| 阶段 | 核心问题 | 结果/决策 | 入口 |
|---|---|---|---|
| 环境与评分 | 工具调用是否可执行、奖励是否可重放 | SQLite 状态、10 工具、strict-v2 已冻结 | `docs/format_control.md`、`docs/project_qa.md` |
| v8 位置探针 | 模型是否只会选第一项订单 | 60/120，成功全在位置 1；暴露捷径 | `docs/order_position_probe.md` |
| v10 位置平衡 SFT | 训练分布能否缓解位置捷径 | seed45/46/47 有改善，但目标检查稳定性门槛未过 | `docs/position_v10_protocol.md`、`docs/position_v10_seed_replication_protocol.md` |
| return-v11 | 退货年龄分布干预是否稳定泛化 | seed-sensitive，旧确认集回归；不作为 RL 证据 | `docs/return_v11_results.md` |
| return-boundary-v1 | 退货临界天数是否是主要失败来源 | holdout 提升不稳定，位置 2 和目标检查仍失败 | `docs/return_boundary_training_protocol.md`、`docs/return_boundary_v1_results.md` |
| final GRPO protocol | 在修正奖励和全新 holdout 后是否仍能观察 RL 增益 | seed45 持平、seed46 回归；目标检查/合规门槛未过，RL 增益未证实；失败主因是理由码契约未显式可观测 | `docs/grpo_final_protocol.md`、`docs/grpo_failure_analysis.md`、`artifacts/grpo-final-v1-failure-analysis.summary.json` |

## 当前状态

已授权的监督诊断、边界对照和最终 GRPO 对照均已完成。最终 GRPO 在全新 holdout 上未超过匹配 SFT，两个 seed 的目标检查/政策合规门槛也未通过，因此按协议收尾为“RL 增益未证实”。不要用专家演示、旧确认集或评分器修订替代模型结果。

## 面试讲法

先用[项目简介](project_brief.md)讲问题和结论，再用[项目总结](project_summary.md)讲证据，最后打开具体协议和结果证明审计链路。最稳妥的表述是：完成了可验证环境、严格评分、策略捷径诊断、多 seed 监督对照和一次预注册的双 seed GRPO 复验；GRPO 未超过匹配 SFT，因此据证据停止继续调参。

## 文件维护

- 对外入口是 `README.md`、`docs/project_brief.md`、`docs/experiment_index.md` 和 `docs/project_summary.md`。
- `scripts/` 中的生成、训练、评测、重放和审计脚本按实验版本保留；历史脚本仍对应协议或已提交摘要，不作为当前未完成任务。
- `artifacts/` 只保留小型审计和聚合摘要；模型权重、逐条轨迹和生成数据由 `.gitignore` 排除。
- 已完成的边界对照不会被删除或改写成成功结果；后续实验必须新建协议和独立确认集。
