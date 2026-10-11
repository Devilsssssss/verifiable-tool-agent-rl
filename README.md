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

所有模型在同一测试集、相同最大轮数和生成预算下评测。主要指标是任务成功率；合规率、有效调用率、平均步数和各任务族结果用于解释差异。

### 0.5B 链路试验

20 条独立测试任务上的低成本诊断：

| 模型 | 任务成功率 | 合规率 | 有效工具调用率 |
| --- | ---: | ---: | ---: |
| 原始 0.5B | 0% | 10% | 0% |
| 0.5B LoRA SFT | 0% | 20% | 40.3% |

SFT 已让小模型学会前三个公共查询步骤，但它在订单状态分支和后续参数绑定上仍失败。因此这组结果只证明训练链路有效，也支持使用 1.7B 进行正式实验；不能据此声称任务能力提升。

### 1.7B 诊断结果

初始 SFT adapter 在 20 条未见实体、未见措辞的测试上为 `0/20` 成功，平均有效工具调用率约 `67.9%`。第一版 GRPO 使用 8 条训练样本时，环境奖励恒为 `1.4`，组内标准差为 0，因而没有有效梯度。修正 TRL 工具适配、加入正确前缀进度奖励，并将采样温度设为 `1.2` 后，20 条 validation 任务的 5 步诊断出现奖励标准差 `0.35--0.40`；测试成功率仍为 `0/20`，说明这只是“奖励链路可学习”的诊断，不是最终能力提升。正式实验还需要扩大训练样本、固定采样预算并在独立测试上复核。

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
```

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

使用隐藏测试集评测任一 adapter：

```bash
python scripts/evaluate_model.py \
  --model-id "$MODEL_DIR" \
  --adapter outputs/qwen3-1.7b-tool-sft \
  --tasks data/generated/tasks_test.jsonl \
  --output artifacts/qwen3-sft-test.jsonl \
  --run-name qwen3-sft-test
```

模型权重、adapter、生成数据和逐条轨迹默认不进入 Git。可公开的摘要保存在 `artifacts/*summary.json`。

## 当前限制

- 当前测试集验证未见实体和未见措辞，不等同于未见任务类型的组合泛化。
- SQLite 工具环境比真实客服系统更干净，结论不能直接外推到生产环境。
- 过程奖励可能被利用，因此必须同时报告终态成功与合规审计，不能只报告平均奖励。
