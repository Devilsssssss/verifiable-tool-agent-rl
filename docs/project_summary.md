# 项目总结：可验证的 Tool-Agent 后训练实验

## 一句话介绍

在一个带 SQLite 状态和 10 个工具的零售环境中，训练 Qwen3-1.7B + LoRA 完成多轮工具调用；用数据库终态、工具调用顺序和业务政策共同评分，而不是只看模型文本。

## 主要工作

- 设计可复核的有状态 Tool-Agent 环境：覆盖取消、拒收升级、换货、期限内退货和过期退货 5 类任务；通过 10 个工具和独立 SQLite 状态约束“先查询、再修改、精确退款”等策略。
- 构建 300 条训练任务、60 条验证任务和 120 条冻结确认任务，完成专家轨迹重放、跨划分实体隔离、隐藏 ID 捷径和真实 SFT mask 审计；共检查 980 条专家任务，实体重叠和捷径成功均为 0。
- 针对首步用户邮箱提取失败，保存原始生成并定位到解析前后均出现的 `For account` 参数污染；设计显式邮箱字段输入协议，避免把提示干预误报成模型增益。
- 在同一显式字段协议下比较基座模型与 v8 LoRA adapter：validation-60 从 `0/60` 提升到 `60/60`，冻结确认集从 `2/120` 提升到 `120/120`，五个业务族均通过，全部轨迹重放一致。
- 新建按任务族平衡的订单位置探针；同一批保存轨迹在 `strict-v2` 下，基座为 `1/120`、v8 为 `60/120`，但 v8 的 60 个成功全部来自目标第一位，目标第二位为 `0/60`。
- 从失败轨迹定位出模型始终选择列表第一项，并发现旧评分器会把 7 条“检查错误订单后升级”的轨迹判成功；保留旧结果后新增目标订单检查约束并离线重放，避免用评分器修订掩盖原始结果。
- 对两个等预算 factorial SFT 候选进行止损验证：原始提示协议下均为 `0/20`，因此停止继续扩充同类 SFT，未用奖励改写、解析器修补或确认集反向选模掩盖负结果。

## 实验结论说明

“我的实验没有把失败包装成成功。模型在原提示中会污染邮箱参数，所以我固定显式字段协议后比较基座和 LoRA，旧确认集从 2/120 到 120/120。但我又做了订单位置平衡测试，发现 adapter 在目标第一位是 60/60、第二位是 0/60，而且始终选择第一项。我还从轨迹中发现旧评分器对升级任务有 7 个假阳性，于是保留 v1 结果并新增要求检查目标订单的 strict-v2。最终结论是模型在固定输入和固定位置上学到了工具策略，但没有订单位置泛化，也没有可信的 RL 增益。”

## 当前结论的边界

- 结果主要证明固定输入协议下的 SFT 能力，不证明多训练 seed 的统计稳定性。
- 订单位置探针已证明 v8 存在完全的位置捷径；尚未完成位置随机化训练与新种子复验。
- 没有可信的 RL 增益证据；后续若做 RL，应先处理输入协议和位置偏差，再使用新的验证协议。
- 所有严格成功都要求终态正确且没有策略违规；`terminal_goal_completion` 不能替代严格成功。

## 关键复现实验

- 原始输出诊断：`artifacts/v8-raw-first-call.summary.json`
- v9 两候选 SFT：`artifacts/qwen3-sft-factorial-v9-high-validation-20.summary.json`、`artifacts/qwen3-sft-factorial-v9-low-validation-20.summary.json`
- 字段协议 validation-60 对照：`artifacts/base-v8-field-v1-validation-60-comparison.summary.json`
- 字段协议冻结确认集对照：`artifacts/base-v8-field-v1-confirmation-120-comparison.summary.json`
- 订单位置 strict-v2 对照：`artifacts/base-v8-position-probe-120-strict-v2-comparison.summary.json`
- 完整方案和实验口径：[factorial_v9_protocol.md](factorial_v9_protocol.md)
