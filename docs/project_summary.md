# 项目总结：可验证的 Tool-Agent 后训练实验

## 一句话介绍

在一个带 SQLite 状态和 10 个工具的零售环境中，训练 Qwen3-1.7B + LoRA 完成多轮工具调用；用数据库终态、工具调用顺序和业务政策共同评分，而不是只看模型文本。

## 主要工作

- 设计可复核的有状态 Tool-Agent 环境：覆盖取消、拒收升级、换货、期限内退货和过期退货 5 类任务；通过 10 个工具和独立 SQLite 状态约束“先查询、再修改、精确退款”等策略。
- 构建 300 条训练任务、60 条验证任务和 120 条冻结确认任务，完成专家轨迹重放、跨划分实体隔离、隐藏 ID 捷径和真实 SFT mask 审计；共检查 980 条专家任务，实体重叠和捷径成功均为 0。
- 针对首步用户邮箱提取失败，保存原始生成并定位到解析前后均出现的 `For account` 参数污染；设计显式邮箱字段输入协议，避免把提示干预误报成模型增益。
- 在同一显式字段协议下比较基座模型与 v8 LoRA adapter：validation-60 从 `0/60` 提升到 `60/60`，冻结确认集从 `2/120` 提升到 `120/120`，五个业务族均通过，全部轨迹重放一致。
- 新建按任务族平衡的订单位置探针；同一批保存轨迹在 `strict-v2` 下，基座为 `1/120`、v8 为 `60/120`，但 v8 的 60 个成功全部来自目标第一位，目标第二位为 `0/60`。
- 预注册等预算的位置平衡v10 SFT，只改变正常/修复视图中的订单位置；在训练前冻结的新实体确认集上从v8的`60/120`提升到`105/120`，目标第二位从`0/60`提升到`47/60`，目标订单检查率达到`120/120`。随后使用同一数据和确认集做 seed46/47 复验，分别达到`101/120`，但目标订单检查率为`55/60`和`48/60`，未通过预注册稳定性门槛。
- 从失败轨迹定位出模型始终选择列表第一项，并发现旧评分器会把 7 条“检查错误订单后升级”的轨迹判成功；保留旧结果后新增目标订单检查约束并离线重放，避免用评分器修订掩盖原始结果。
- 对两个等预算 factorial SFT 候选进行止损验证：原始提示协议下均为 `0/20`，因此停止继续扩充同类 SFT，未用奖励改写、解析器修补或确认集反向选模掩盖负结果。

## 实验结论说明

“我的实验没有把失败包装成成功。v8在旧确认集达到120/120，但位置探针显示它第一位60/60、第二位0/60，而且始终选择第一项。我因此预注册了等预算的位置平衡v10，只改变训练订单位置；新冻结集上v10达到105/120，第二位从0/60提升到47/60，并在120题中都检查了目标订单。不过15个失败全部在退货分支，而且实验只有一个训练seed。结论是受控数据干预改善了位置泛化，但退货策略边界和多seed稳定性仍未解决，也没有可信的RL增益。”

## 当前结论的边界

- 结果主要证明固定输入协议下的 SFT 能力；seed46/47 的总体成功率保持在`101/120`，但预注册要求的目标订单检查`60/60`未复现，因此不能声称已通过多训练 seed 稳定性复验。
- v8的完全位置捷径已由v10位置平衡训练显著改善，但只完成一个训练seed和一个新确认集，不能外推为稳定的通用泛化。
- v10仍有15/120失败，全部属于期限内/过期退货策略判断或失败恢复，而不是订单选择错误。
- 没有可信的 RL 增益证据；位置平衡 SFT 的训练 seed 复验未通过预注册门槛，后续应先处理退货策略边界和目标检查，再决定是否进入 RL。
- 所有严格成功都要求终态正确且没有策略违规；`terminal_goal_completion` 不能替代严格成功。

## 关键复现实验

- 原始输出诊断：`artifacts/v8-raw-first-call.summary.json`
- v9 两候选 SFT：`artifacts/qwen3-sft-factorial-v9-high-validation-20.summary.json`、`artifacts/qwen3-sft-factorial-v9-low-validation-20.summary.json`
- 字段协议 validation-60 对照：`artifacts/base-v8-field-v1-validation-60-comparison.summary.json`
- 字段协议冻结确认集对照：`artifacts/base-v8-field-v1-confirmation-120-comparison.summary.json`
- 订单位置 strict-v2 对照：`artifacts/base-v8-position-probe-120-strict-v2-comparison.summary.json`
- v10位置平衡确认对照：`artifacts/v8-v10-position-v10-confirmation-comparison.summary.json`
- v10失败归因：`artifacts/v10-position-v10-confirmation-failure-analysis.summary.json`
- v10 seed复验汇总：`artifacts/v10-seed-replication-comparison.summary.json`
- 完整方案和实验口径：[factorial_v9_protocol.md](factorial_v9_protocol.md)
- v10预注册方案与结果：[position_v10_protocol.md](position_v10_protocol.md)
- v10 seed复验方案与结果：[position_v10_seed_replication_protocol.md](position_v10_seed_replication_protocol.md)
