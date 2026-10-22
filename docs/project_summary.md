# 项目总结：可验证的 Tool-Agent 后训练实验

## 一句话介绍

在一个带 SQLite 状态和 10 个工具的零售环境中，训练 Qwen3-1.7B + LoRA 完成多轮工具调用；用数据库终态、工具调用顺序和业务政策共同评分，而不是只看模型文本。

## 核心实验结果

| 实验 | 测试集 | 严格成功 | 结论 |
| --- | --- | ---: | --- |
| v8 位置探针 | 120 条 | 60/120 | 60 个成功全部来自目标订单第一位，暴露位置捷径 |
| v10 seed45 | 旧位置确认集 | 105/120 | 位置平衡 SFT 改善了第二位泛化 |
| v10 seed46/47 | 旧位置确认集 | 101/120、101/120 | 总体提升复现，但未满足预注册的目标检查稳定性门槛 |
| v11 seed45 | 新冻结退货 holdout | 56/120 | 相比匹配 v10 的 58/120 略有下降 |
| v11 seed46 | 新冻结退货 holdout | 81/120 | 相比匹配 v10 的 54/120 提升，但只在一个训练 seed 上出现 |
| v11 seed45/46 | 旧确认集回归 | 96/120、96/120 | 两个 seed 都低于匹配 v10，存在回归 |
| return-boundary-v1 seed45 | 新退货 holdout / 旧确认集 | 78/120 / 87/120 | holdout 有提升但旧集显著回归，目标检查门槛未过 |
| return-boundary-v1 seed46 | 新退货 holdout / 旧确认集 | 98/120 / 103/120 | holdout 提升、旧集基本持平；位置 2 和目标检查仍未达门槛 |

v11 的结果不是 RL 增益证据，而是一次预先固定的监督数据分布干预：它改变了退货目标和干扰订单的年龄分布，同时保持训练视图数量和优化配置不变。完整配对结果和失败模式见 [return_v11_results.md](return_v11_results.md)。

## 主要工作

- 设计可复核的有状态 Tool-Agent 环境：覆盖取消、拒收升级、换货、期限内退货和过期退货 5 类任务；通过 10 个工具和独立 SQLite 状态约束“先查询、再修改、精确退款”等策略。
- 构建 300 条训练任务、60 条验证任务和 120 条冻结确认任务，完成专家轨迹重放、跨划分实体隔离、隐藏 ID 捷径和真实 SFT mask 审计；共检查 980 条专家任务，实体重叠和捷径成功均为 0。
- 针对首步用户邮箱提取失败，保存原始生成并定位到解析前后均出现的 `For account` 参数污染；设计显式邮箱字段输入协议，避免把提示干预误报成模型增益。
- 在同一显式字段协议下比较基座模型与 v8 LoRA adapter：validation-60 从 `0/60` 提升到 `60/60`，冻结确认集从 `2/120` 提升到 `120/120`，五个业务族均通过，全部轨迹重放一致。
- 新建按任务族平衡的订单位置探针；同一批保存轨迹在 `strict-v2` 下，基座为 `1/120`、v8 为 `60/120`，但 v8 的 60 个成功全部来自目标第一位，目标第二位为 `0/60`。
- 预注册等预算的位置平衡v10 SFT，只改变正常/修复视图中的订单位置；在训练前冻结的新实体确认集上从v8的`60/120`提升到`105/120`，目标第二位从`0/60`提升到`47/60`，目标订单检查率达到`120/120`。随后使用同一数据和确认集做 seed46/47 复验，分别达到`101/120`，但目标第二位的订单检查分别为`55/60`和`48/60`，未通过预注册稳定性门槛。
- 从失败轨迹定位出模型始终选择列表第一项，并发现旧评分器会把 7 条“检查错误订单后升级”的轨迹判成功；保留旧结果后新增目标订单检查约束并离线重放，避免用评分器修订掩盖原始结果。
- 对两个等预算 factorial SFT 候选进行止损验证：原始提示协议下均为 `0/20`，因此停止继续扩充同类 SFT，未用奖励改写、解析器修补或确认集反向选模掩盖负结果。
- 对 return-v11 两个训练 seed 做冻结 holdout 与旧确认集的 task-level gained/lost 分析；新增 `scripts/analyze_failure_deltas.py`，将正确终态但策略违规、未检查目标订单和反复调用已关闭退货接口分开统计。
- 针对 v10 seed 复验暴露的退货分支失败，冻结 `return-boundary-v1` 监督对照：保留 300 个训练任务和 900 个视图，只将退货目标年龄放在 29/30 天与 31/32 天边界，并平衡目标订单位置；训练前语义审计覆盖 600 个任务，专家成功、实体隔离和捷径检查均通过。
- 完成 return-boundary-v1 seed45/46 的匹配训练、退货 holdout、旧确认集回归、轨迹重放和配对分析；结果为 holdout `78/120、98/120`，旧确认集 `87/120、103/120`。一个 seed 明显回归，另一个 seed 只有不确定的小幅净增，因此维持 GRPO 阻断。
- 增加向后兼容的 `strict-v3-reason-code` 评测协议，验证升级理由与目标订单政策匹配；strict-v2 历史轨迹和结果保持不变。
- 冻结最终 GRPO 对照协议：基于已审计的 `return-boundary-v1` SFT，使用 `strict_shaped` 训练信号、`strict-v3-reason-code`、全新 120 条退货 holdout 和两个 RL seed；训练前不读取候选结果，未通过严格成功/目标检查/政策合规门槛就不宣称增益。
  - 完成最终 GRPO 双 seed 复验：新 holdout 上 seed45 为 56/120，与匹配 SFT 持平；seed46 为 51/120，低于匹配 SFT 的 52/120。目标订单检查率为 91.7%/86.7%，政策合规率为 46.7%/50.0%，均未达到门槛；因此 RL 增益未证实，停止继续增加步数或挑选 checkpoint。
  - 对四份逐条 rollout 做失败归因：240/240 条过期退货任务失败，聚合出现 710 次理由码不匹配、525 次重复请求已关闭退货和 494 次 step-limit。根因是 strict-v3 后加入的机器理由码没有进入 boundary-v1 SFT 的可观测提示/动作契约；下一轮应先做双 seed SFT 修复，再决定是否重启 RL。

## 实验结论说明

v8在旧确认集达到120/120，但位置探针显示它第一位60/60、第二位0/60，而且始终选择第一项。等预算的位置平衡v10只改变训练订单位置，seed45在旧确认集达到105/120，后续 seed46/47 均为101/120；这支持“位置平衡 SFT 能缓解位置捷径”，但目标检查稳定性仍未达到预注册门槛。

return-v11 将问题推进到新冻结退货边界：seed45 为56/120，seed46 为81/120，但两个 seed 在旧确认集都回退到96/120。seed46 在过期退货且目标第一位时达到29/30，却在目标第二位时为0/30。当前最稳妥的结论是：数据分布干预会改变策略表现，但收益依赖训练 seed，并可能损害已有能力；没有可信的 RL 增益。

`return-boundary-v1` 已完成两次与 v10 对齐的监督训练和评测，但它仍不是 RL 结果。seed45 在旧确认集从 105/120 回退到 87/120，seed46 从 101/120 变为 103/120；holdout 的提升具有明显 seed sensitivity，且两个 seed 的目标订单检查率都低于全量门槛。因此不能把它写成稳定泛化或 RL 先验证据。

## 当前结论的边界

- 结果主要证明固定输入协议下的 SFT 能力；v10 的总体提升未满足目标订单检查稳定性门槛，v11 又表现出 seed-sensitive 收益和旧集回归，因此不能声称数据干预或 RL 带来了稳定提升。
- 三个训练seed均在同一合成确认集上改善了v8的第一位置捷径，但该集合的seed45结果已用于决定开展复验；复验不构成第二份独立盲测，不能外推为通用泛化。
- seed45失败15/120，seed46/47各失败19/120，全部位于退货分支；复验还暴露出5条和12条错误订单检查后升级的轨迹，不能把这些终态正确的任务算作严格成功。
- 没有可信的 RL 增益证据；位置平衡 SFT 的训练 seed 复验未通过预注册门槛，边界对照又显示一个 seed 明显回归、另一个 seed 只有小幅不确定净增。后续应先处理退货位置 2 和目标检查，再决定是否进入 RL。
- 新退货 holdout 同时改变了实体、措辞和年龄分布，不能把结果解释成单独的年龄因果效应；旧确认集已经参与开发分析，只作为回归检查。
- 所有严格成功都要求终态正确且没有策略违规；`terminal_goal_completion` 不能替代严格成功。
- 边界对照已完成，目标检查门槛仍未在监督复验中稳定解决。最终 GRPO 对照已执行完毕：两个 seed 均未超过匹配 SFT，目标检查/政策合规也未达 100%，实验按“RL 增益未证实”收尾。

## 关键复现实验

- 原始输出诊断：`artifacts/v8-raw-first-call.summary.json`
- v9 两候选 SFT：`artifacts/qwen3-sft-factorial-v9-high-validation-20.summary.json`、`artifacts/qwen3-sft-factorial-v9-low-validation-20.summary.json`
- 字段协议 validation-60 对照：`artifacts/base-v8-field-v1-validation-60-comparison.summary.json`
- 字段协议冻结确认集对照：`artifacts/base-v8-field-v1-confirmation-120-comparison.summary.json`
- 订单位置 strict-v2 对照：`artifacts/base-v8-position-probe-120-strict-v2-comparison.summary.json`
- v10位置平衡确认对照：`artifacts/v8-v10-position-v10-confirmation-comparison.summary.json`
- v10失败归因：`artifacts/v10-position-v10-confirmation-failure-analysis.summary.json`
- v10 seed复验汇总：`artifacts/v10-seed-replication-comparison.summary.json`
- v11 结果报告：`docs/return_v11_results.md`
- v11 失败差异分析：`artifacts/return-v11-*-failure-deltas.summary.json`
- 退货边界训练集审计：`artifacts/return-boundary-v1-training-audit.summary.json`
- 退货边界生成脚本：`scripts/create_return_boundary_training.py`
- 退货边界协议：`docs/return_boundary_training_protocol.md`
- 退货边界结果：`docs/return_boundary_v1_results.md`
- strict-v3 理由校验协议：[reason_code_protocol.md](reason_code_protocol.md)
- 完整方案和实验口径：[factorial_v9_protocol.md](factorial_v9_protocol.md)
- v10预注册方案与结果：[position_v10_protocol.md](position_v10_protocol.md)
- v10 seed复验方案与结果：[position_v10_seed_replication_protocol.md](position_v10_seed_replication_protocol.md)
