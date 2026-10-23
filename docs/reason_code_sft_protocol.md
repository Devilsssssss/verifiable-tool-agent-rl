# reason-code-v1 SFT 修复协议

状态：**训练数据已生成并通过离线审计，尚未开始服务器训练。**

## 目的

最终 GRPO 的失败归因显示，strict-v3-reason-code 要求精确的机器理由码，
但 return-boundary-v1 的初始化 SFT 没有把这个动作契约放进模型可观测的
系统提示和工具 schema。本轮只修复输入/动作契约，暂不修改奖励、环境状态、
任务分布或 GRPO 超参数。

## 变量隔离

| 变量 | 历史默认 | 本轮修复 |
| --- | --- | --- |
| 评分协议 | strict-v2-target-inspection 或显式 strict-v3-reason-code | strict-v3-reason-code |
| 输入契约 | legacy | reason-code-v1 |
| 工具动作 | reason_code 可选 | reason_code 必填，枚举为 shipped_order_cannot_cancel、return_window_expired |
| 训练预算 | 300 个任务、900 个视图 | 完全相同 |
| 验证预算 | 60 个任务、120 个视图 | 完全相同 |

评分协议和输入契约是两个独立开关。历史 strict-v3 结果继续使用旧提示和旧
schema；只有显式传入 --input-contract reason-code-v1 才启用新契约。

## 数据审计

生成命令：

~~~powershell
python scripts/create_reason_code_sft_data.py
~~~

当前冻结输入摘要为：

- 训练源：return-boundary-v1/tasks_train.jsonl，300 个独立任务；
- 训练视图：正常 300、参数修复 300、lookup 300，共 900；
- 验证源：tasks_validation.jsonl，60 个独立任务；
- 验证视图：正常 60、参数修复 60，共 120；
- strict-v3 专家重放：训练 300/300，验证 60/60；
- 数据摘要：artifacts/reason-code-sft-v1-training-audit.summary.json。

训练前还必须运行真实 SFT mask 审计，且所有 mismatch、空监督和截断计数为零。

## 训练与评测

两个 seed 使用相同的基座、数据、LoRA 配置和优化预算。训练命令可写成一行：

~~~powershell
python scripts/train_sft.py --model-id models/Qwen3-1.7B --train data/reason-code-sft-v1/sft_train.jsonl --train-extra data/reason-code-sft-v1/sft_repair_lookup_train.jsonl --validation data/reason-code-sft-v1/sft_validation.jsonl --validation-extra data/reason-code-sft-v1/sft_repair_validation.jsonl --output-dir outputs/qwen3-1.7b-tool-sft-reason-code-v1-seed45 --epochs 1 --learning-rate 2e-4 --max-length 4096 --batch-size 1 --gradient-accumulation 8 --seed 45 --input-contract reason-code-v1
~~~

seed46 只替换 output-dir 和 seed 46。训练完成后，先审计两个 adapter
的 mask，再在同一份冻结确认集上使用：

~~~text
--scoring-protocol strict-v3-reason-code --input-contract reason-code-v1
~~~

不得用 holdout 选择 seed、checkpoint 或解码参数。

## 决策门槛

本轮是 SFT 修复，不是 RL 结果。只有两个 seed 都在独立确认集上同时满足：

1. 严格成功率超过原 return-boundary-v1 匹配 SFT；
2. 目标订单检查率达到 100%；
3. 政策合规率达到 100%；
4. 过期退货分支不再出现缺失/错误理由码和重复请求已关闭退货；

才允许继续设计新的 GRPO 对照。任一门槛失败，就保留失败归因，停止继续调
GRPO，转而处理任务契约、订单位置或评测覆盖。

