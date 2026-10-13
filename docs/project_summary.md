# 项目总结：可验证的 Tool-Agent 后训练实验

## 一句话介绍

在一个带 SQLite 状态和 10 个工具的零售环境中，训练 Qwen3-1.7B + LoRA 完成多轮工具调用；用数据库终态、工具调用顺序和业务政策共同评分，而不是只看模型文本。

## 主要工作

- 设计可复核的有状态 Tool-Agent 环境：覆盖取消、拒收升级、换货、期限内退货和过期退货 5 类任务；通过 10 个工具和独立 SQLite 状态约束“先查询、再修改、精确退款”等策略。
- 构建 300 条训练任务、60 条验证任务和 120 条冻结确认任务，完成专家轨迹重放、跨划分实体隔离、隐藏 ID 捷径和真实 SFT mask 审计；共检查 980 条专家任务，实体重叠和捷径成功均为 0。
- 针对首步用户邮箱提取失败，保存原始生成并定位到解析前后均出现的 `For account` 参数污染；设计显式邮箱字段输入协议，避免把提示干预误报成模型增益。
- 在同一显式字段协议下比较基座模型与 v8 LoRA adapter：validation-60 从 `0/60` 提升到 `60/60`，冻结确认集从 `2/120` 提升到 `120/120`，五个业务族均通过，全部轨迹重放一致。
- 对两个等预算 factorial SFT 候选进行止损验证：原始提示协议下均为 `0/20`，因此停止继续扩充同类 SFT，未用奖励改写、解析器修补或确认集反向选模掩盖负结果。

## 实验结论说明

“我的实验最后没有把原始提示下的失败包装成成功。模型在原提示中会把 `For account` 一起复制进邮箱，导致第一步 lookup 失败。于是我把问题拆成输入协议和训练能力两个因素：在请求中显式列出用户已经提供的邮箱，再对比基础模型和 LoRA adapter。字段协议下基座在冻结集是 2/120，adapter 是 120/120，说明 adapter 在这个协议上确实学到了后续工具策略；但原始协议下仍是 0/20，所以我的结论限定为‘固定输入协议下的后训练效果’，不是泛化到任意用户表达，也没有宣称 RL 已经带来增益。”

## 当前结论的边界

- 结果主要证明固定输入协议下的 SFT 能力，不证明多训练 seed 的统计稳定性。
- 确认集仍保持目标订单排在第一，尚未证明订单位置泛化。
- 没有可信的 RL 增益证据；后续若做 RL，应先处理输入协议和位置偏差，再使用新的验证协议。
- 所有严格成功都要求终态正确且没有策略违规；`terminal_goal_completion` 不能替代严格成功。

## 关键复现实验

- 原始输出诊断：`artifacts/v8-raw-first-call.summary.json`
- v9 两候选 SFT：`artifacts/qwen3-sft-factorial-v9-high-validation-20.summary.json`、`artifacts/qwen3-sft-factorial-v9-low-validation-20.summary.json`
- 字段协议 validation-60 对照：`artifacts/base-v8-field-v1-validation-60-comparison.summary.json`
- 字段协议冻结确认集对照：`artifacts/base-v8-field-v1-confirmation-120-comparison.summary.json`
- 完整方案和实验口径：[factorial_v9_protocol.md](factorial_v9_protocol.md)
