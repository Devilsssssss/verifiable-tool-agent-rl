# Verifiable Tool-Agent RL

一个面向 Agentic RL / 大模型后训练实习的研究型项目：在有状态的本地零售环境中训练小模型完成多轮工具调用，并用数据库终态与策略约束给出可执行、可复核的奖励。

## 研究问题

在相同 SFT 初始化和生成预算下，过程奖励是否比只有任务成败的稀疏奖励更能提升：

- 端到端任务成功率；
- 策略合规率；
- 有效工具调用率；
- 未见用户、商品、订单标识和提示措辞上的泛化能力。

正式实验使用 `Qwen3-1.7B + LoRA`。`Qwen2.5-Coder-0.5B-Instruct` 只用于低成本验证训练、解析和评测链路，不作为最终结论。

## 环境与任务

每个 episode 都创建独立的内存 SQLite 数据库。模型只能通过 10 个工具观察和修改状态，不能直接读取隐藏 ID 或目标答案。

```text
用户请求
  -> lookup_user -> list_orders -> get_order
  -> 根据订单状态与政策选择操作
  -> cancel / exchange / return / escalate
  -> 必要时 issue_refund
  -> 数据库终态 + 完整调用轨迹 -> reward
```

当前包含五类任务：处理中订单取消、已发货订单拒绝并升级、低价换货并退差价、期限内退货、过期退货拒绝并升级。每条任务还包含干扰订单和干扰商品，避免只凭位置猜目标。

环境强制检查：先查询再使用标识、修改前检查订单、换货前检查库存、只向原支付方式退还精确金额、超过退货期或系统能力时保持业务状态不变并升级人工。

## 可验证奖励

`sparse` 仅使用最终任务成功 `1/0`。`shaped` 同时考虑任务结果、策略合规、有效工具调用率、正确计划前缀进度和步骤效率：

```text
task outcome + policy compliance + valid tool-call rate + correct-prefix progress + efficiency
```

成功不是由模型自报，而是由数据库终态和违规轨迹共同决定。直接使用隐藏 ID、空操作和绕过检查的捷径都不能获得成功奖励。

当前严格成功指标会把 `invalid_tool_call`（例如多传参数）也记为违规，即使模型随后纠正了调用并达到目标，严格成功仍为 0。为分析恢复行为，额外报告 `terminal_goal_completion`（仅检查目标数据库终态）和 `format_recovery_rate`（只有格式错误、最终达到目标的任务数 / 出现格式错误的任务数）。存在业务政策违规或超预算的轨迹不能计入格式恢复成功。这两个诊断指标不改变严格成功或训练奖励。

## 数据与审计

固定种子生成 `300/60/120` 条训练、验证和测试任务。三个划分使用不同的实体和提示模板；同一实体不会跨划分出现。`scripts/audit_tasks.py` 会重放所有专家轨迹并检查重复、ID 泄漏、跨划分实体重叠、空操作和直接捷径。

当前审计结果：

| 检查 | 结果 |
| --- | ---: |
| 总任务数 | 480 |
| 专家轨迹成功 | 480 / 480 |
| 空操作成功 | 0 / 480 |
| 隐藏 ID 捷径成功 | 0 / 480 |
| 跨划分实体重叠 | 0 |

完整摘要见 `artifacts/task_audit_summary.json` 和 `artifacts/environment_baselines_summary.json`。

## 实验矩阵

| 组别 | 初始化 | 奖励 | 目的 |
| --- | --- | --- | --- |
| A | 原始模型 | 无训练 | 零样本基线 |
| B | LoRA SFT | 专家轨迹 | 建立可用工具策略 |
| C | B | 稀疏 GRPO | 测试只有终态奖励的效果 |
| D | B | 过程奖励 GRPO | 测试合规与过程信号的增益 |
| E | 参数错误反馈 SFT | 专家轨迹 + 执行报错后的纠正轨迹 | 测试模型能否从工具参数错误中恢复 |

所有模型在同一测试集、相同最大轮数和生成预算下评测。主要指标是任务成功率；合规率、有效调用率、平均步数和各任务族结果用于解释差异。

### 0.5B 链路试验

20 条独立测试任务上的低成本诊断：

| 模型 | 任务成功率 | 合规率 | 有效工具调用率 |
| --- | ---: | ---: | ---: |
| 原始 0.5B | 0% | 10% | 0% |
| 0.5B LoRA SFT | 0% | 20% | 40.3% |

SFT 已让小模型学会前三个公共查询步骤，但它在订单状态分支和后续参数绑定上仍失败。因此这组结果只证明训练链路有效，也支持使用 1.7B 进行正式实验；不能据此声称任务能力提升。

### 1.7B 诊断结果

初始 SFT adapter 在 20 条未见实体、未见措辞的测试上为 `0/20` 严格成功，平均有效工具调用率约 `67.9%`。两组相同预算的 GRPO（30 步、4 个 generations、温度 `1.2`）在 validation-20 和 test-20 上仍为 `0/20`，但 test 的正确计划前缀约为 `64%`。训练中组内奖励几乎没有方差，当前结果不能证明 GRPO 带来了增益。

错误反馈 SFT 的各版结果如下。test-20 均使用同一批前 20 条任务，每个任务族 4 条；test-120 包含这些任务，不能把两批当成独立实验。

| 版本 | 训练口径 | 评测 | 严格成功 | 目标终态完成 | 格式恢复成功 / 格式错误任务 |
| --- | --- | --- | ---: | ---: | ---: |
| repair-v1 | 300 条修复全轨迹，错误调用也受到监督，1 epoch | test-20 | 7/20 | 7/20 | 0/12 |
| repair-v1 | 同一 adapter | test-120 | 42/120 | 50/120 | 4/74 |
| repair-v3 | 300 条正常轨迹 + 300 条修复轨迹，仅监督修复边界后的正确输出，1 epoch | test-20 | 0/20 | 12/20 | 8/20 |
| repair-v3 | 同一 adapter | validation-20 | 0/20 | 0/20 | 无格式错误任务，恢复率未定义 |

v3 的 12 条终态完成中，8 条只有格式错误；另外 4 条还曾违反退货政策，因此不能计入格式恢复成功。取消任务会反复给退款工具传入多余的 `order` 参数；换货任务会选中干扰订单。validation-20 的 20 条任务都发生了错误邮箱查询。v3 尚未扩测 test-120。

v1 的 35% 是旧版全轨迹训练的探索性结果，不能归因于“正确屏蔽错误调用”。只使用修复后片段的 v2 在冒烟检查中未正常发起工具调用，没有完成正式评测。v3 补回正常轨迹后可发起调用，但严格成功退化。各版训练样本数、损失口径和种子不同，目前没有完成等预算、单变量、多种子的对照。

`scripts/audit_sft_masks.py` 使用实际 SFTTrainer 预处理和 collator 检查 600 条训练样本：提示全部被屏蔽，无空监督、无截断，生成前缀与训练序列一致。同时发现 600 条训练/推理提示存在工具 schema 键顺序差异，300 条修复提示在 Arrow 转换后多出 `null` 参数字段。这些是待控制的格式差异，尚不能证明它们造成了性能退化。审计见 `artifacts/qwen3-sft-repair-v3-mask-audit.summary.json`。

`scripts/audit_rollouts.py` 重放已保存的调用轨迹，核对每步结果与原有严格分数，再统计终态和恢复；重放不会生成新的模型回答。详细结果见 `artifacts/*-replay.summary.json`。

## 复现

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"

python scripts/generate_tasks.py
python scripts/audit_tasks.py
python scripts/run_baselines.py
pytest
ruff check .
```

设置本地模型目录后运行 SFT：

```bash
MODEL_DIR=/path/to/Qwen3-1.7B
python scripts/train_sft.py \
  --model-id "$MODEL_DIR" \
  --output-dir outputs/qwen3-1.7b-tool-sft

python scripts/train_sft.py \
  --model-id "$MODEL_DIR" \
  --train data/generated/sft_train.jsonl \
  --train-extra data/generated/sft_repair_train.jsonl \
  --validation data/generated/sft_validation.jsonl \
  --validation-extra data/generated/sft_repair_validation.jsonl \
  --output-dir outputs/qwen3-1.7b-tool-sft-repair-v3 \
  --epochs 1 --learning-rate 2e-4 \
  --batch-size 1 --gradient-accumulation 8 --seed 45

python scripts/audit_sft_masks.py \
  --model-id "$MODEL_DIR" \
  --data data/generated/sft_train.jsonl data/generated/sft_repair_train.jsonl \
  --output artifacts/qwen3-sft-repair-v3-mask-audit.summary.json
```

混合版共有 600 个训练视图，来自 300 个独立任务；验证集 120 个视图来自 60 个独立任务。正常视图监督完整正确调用序列，修复视图的错误调用和执行报错仅作上下文。当前命令复现 v3 的训练口径，不复现历史 v1 的全轨迹损失。

分别从同一个 SFT adapter 运行两组 GRPO：

```bash
python scripts/train_grpo.py \
  --model-id "$MODEL_DIR" \
  --sft-adapter outputs/qwen3-1.7b-tool-sft \
  --reward-mode sparse \
  --output-dir outputs/qwen3-1.7b-grpo-sparse

python scripts/train_grpo.py \
  --model-id "$MODEL_DIR" \
  --sft-adapter outputs/qwen3-1.7b-tool-sft \
  --reward-mode shaped \
  --output-dir outputs/qwen3-1.7b-grpo-shaped
```

复现 v3 的固定 20 条测试诊断：

```bash
python scripts/evaluate_model.py \
  --model-id "$MODEL_DIR" \
  --adapter outputs/qwen3-1.7b-tool-sft-repair-v3 \
  --tasks data/generated/tasks_test.jsonl \
  --output artifacts/qwen3-1.7b-tool-sft-repair-v3-test-20.jsonl \
  --run-name qwen3-1.7b-tool-sft-repair-v3-test-20 \
  --limit 20 --max-new-tokens 512 --temperature 0 --seed 42

python scripts/audit_rollouts.py \
  --tasks data/generated/tasks_test.jsonl \
  --rollouts artifacts/qwen3-1.7b-tool-sft-repair-v3-test-20.jsonl \
  --output artifacts/qwen3-1.7b-tool-sft-repair-v3-test-20-replay.summary.json
```

模型权重、adapter、生成数据和逐条轨迹默认不进入 Git。可公开的摘要保存在 `artifacts/*summary.json`。

## 当前限制

- 当前测试集验证未见实体和未见措辞，不等同于未见任务类型的组合泛化。
- 测试结果已用于诊断和设计后续训练数据，因此属于开发阶段的探索性评测；后续确认实验需要另设未用于调参的评测集。
- 当前严格奖励把可恢复的参数格式错误和业务政策违规都视为失败；应先明确奖励设计与恢复能力的研究目标，再开展新的 RL 对照。
- SQLite 工具环境比真实客服系统更干净，结论不能直接外推到生产环境。
- 过程奖励可能被利用，因此必须同时报告终态成功与合规审计，不能只报告平均奖励。
