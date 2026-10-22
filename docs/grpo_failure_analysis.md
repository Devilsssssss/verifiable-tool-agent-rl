# 最终 GRPO 失败归因

这份报告分析最终双 seed GRPO 在全新退货 holdout 上的逐条 rollout，不把训练日志中的平均奖励当成模型能力。

## 证据范围

- 数据：`data/grpo-final-v1/holdout/tasks_test.jsonl`，120 条，实体、措辞、ID 和退货年龄均已冻结；
- 运行：匹配 SFT seed45/46 与 GRPO seed45/46，共 480 条逐条 rollout；
- 协议：`strict-v3-reason-code`，训练信号为 `strict_shaped`；
- 聚合结果：`artifacts/grpo-final-v1-failure-analysis.summary.json`。

## 主要发现

| 现象 | 聚合数量 | 解释 |
|---|---:|---|
| `return_expired` 任务失败 | 240/240 | 瓶颈集中在过期退货分支，而不是所有工具调用 |
| 缺失或错误 `reason_code` | 142 个失败任务；1,012 次缺失参数调用 | 模型没有学到 strict-v3 的机器理由码契约 |
| `escalation_reason_code_mismatch` | 710 次 | 升级工具拒绝了不匹配的理由码 |
| 重复请求已关闭退货 | 98 个失败任务；525 次 `return_window_closed` | 模型在工具拒绝后没有切换到升级分支 |
| 失败后触发 step limit | 247 个失败任务；494 次 | 错误恢复变成了重复调用，而不是终止或升级 |
| 目标订单未检查 | 51 个失败任务 | 仍有位置和目标识别问题，但不是本轮的唯一主因 |

seed45 和 seed46 的 GRPO 失败模式与对应 SFT 基线基本一致：GRPO 没有改变过期退货分支的决策结构。`return_eligible` 仍有 215/240 条成功，说明问题不是模型完全不会使用工具，而是特定策略分支的任务契约不完整。

## 根因判断

`return-boundary-v1` SFT 初始化的训练样本中，过期退货专家计划只调用：

```text
escalate_to_human(reason=...)
```

最终 strict-v3 数据生成脚本后来把 `reason_code=return_window_expired` 加入了隐藏的 `goal_json` 和专家轨迹，但系统提示和自然语言用户请求没有公开这个枚举。于是模型必须从未见过的环境字段推断一个精确机器值；工具拒绝后，模型又没有明确的失败转移规则，导致重复调用直到 step limit。

这说明本轮结果不能简单解释为“GRPO 没学会理由码”。更准确的结论是：**GRPO 初始化与 strict-v3 的输入/动作契约不匹配，奖励无法修复缺失的可观测任务信息。**

## 下一轮边界

下一轮只允许先做匹配的双 seed SFT 修复：

1. 在系统提示和 `escalate_to_human` 工具 schema 中显式说明 `reason_code` 是必需字段，并公开 `return_window_expired`、`shipped_order_cannot_cancel` 等有限枚举；
2. 为过期退货加入“检查订单 -> 一次升级 -> 停止”的正例，以及工具拒绝后不得重复请求退货的负例；
3. 保持基座、任务数量、位置平衡、解码和评测集不变，先验证 SFT 是否同时修复理由码和目标订单检查；
4. 只有两个 SFT seed 在新的独立确认集上达到预设门槛，才重新设计 RL 奖励并考虑 GRPO。

不应通过删除 strict-v3、把错误升级算成功、或增加 GRPO 步数来掩盖这个接口问题。
