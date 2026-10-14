# 订单位置泛化探针

这个实验检验模型是否真正读取 `list_orders` 的返回值，还是依赖“目标订单总在
第一位”的数据捷径。它不修改模型、奖励、工具定义或评测器。

## 固定协议

- 使用 seed `20261014` 生成 120 条新确认任务，与训练、验证、旧测试和旧确认集
  的实体隔离。
- 使用 seed `20261015` 在每个业务族内随机分配目标订单的位置；每族 24 条中，
  12 条目标在第一位，12 条在第二位。
- 订单 ID、订单项外键、目标状态和专家轨迹中的所有 `order_id` 同步交换，任务
  语义保持不变。
- 只比较冻结的原始 Qwen3-1.7B 与冻结的 v8 adapter。两者都使用 greedy、
  512 tokens、seed 42 和同一 `explicit-email-field-v1` 输入协议。
- 任务不进入训练、提示词修改或 checkpoint 选择。看到结果后不重跑挑种子。

生成阶段必须满足：120/120 专家成功、0 条空操作成功、0 条直接捷径成功，且
五个任务族的位置计数均为 12/12。模型结果按总体、位置和业务族/位置交叉分层
报告；单次固定解码不能代表训练随机种子的方差。

## 复现命令

```bash
python scripts/create_confirmation.py --seed 20261014 --count 120 \
  --output-dir data/position-confirmation \
  --summary artifacts/position-confirmation-audit.summary.json
python scripts/create_position_shifted.py \
  --input data/position-confirmation/tasks_test.jsonl \
  --output artifacts/position-shifted-tasks.jsonl \
  --summary artifacts/position-shifted-tasks.summary.json --seed 20261015

python scripts/evaluate_model.py --model-id models/Qwen3-1.7B \
  --tasks artifacts/position-shifted-tasks.jsonl \
  --input-intervention explicit-email-field-v1 --limit 120 \
  --max-new-tokens 512 --temperature 0 --seed 42 \
  --run-name base-position-probe-120 \
  --output artifacts/base-position-probe-120.jsonl
python scripts/evaluate_model.py --model-id models/Qwen3-1.7B \
  --adapter outputs/qwen3-1.7b-tool-sft-repair-v8 \
  --tasks artifacts/position-shifted-tasks.jsonl \
  --input-intervention explicit-email-field-v1 --limit 120 \
  --max-new-tokens 512 --temperature 0 --seed 42 \
  --run-name v8-position-probe-120 \
  --output artifacts/v8-position-probe-120.jsonl

python scripts/audit_rollouts.py --tasks artifacts/position-shifted-tasks.jsonl \
  --rollouts artifacts/base-position-probe-120.jsonl \
  --output artifacts/base-position-probe-120-replay.summary.json
python scripts/audit_rollouts.py --tasks artifacts/position-shifted-tasks.jsonl \
  --rollouts artifacts/v8-position-probe-120.jsonl \
  --output artifacts/v8-position-probe-120-replay.summary.json
python scripts/analyze_position_probe.py \
  --tasks artifacts/position-shifted-tasks.jsonl \
  --baseline artifacts/base-position-probe-120.jsonl \
  --candidate artifacts/v8-position-probe-120.jsonl \
  --output artifacts/base-v8-position-probe-120-comparison.summary.json
```

## 结果与评分器修订

预注册的 `strict-v1` 结果为基座 `1/120`、v8 `67/120`。逐条检查后发现，v8
目标第一位为 `60/60`，第二位为 `7/60`；但第二位的 7 条成功都来自
`cancel_shipped`，且模型检查的仍是第一条干扰订单。旧评分器对
`escalate_unchanged` 只检查业务状态未改变和发生一次升级，因此产生假阳性。

这是看到结果后的评分器修订。项目保留上述 v1 摘要，同时新增
`strict-v2-target-inspection`，要求成功轨迹实际检查目标订单。使用
`scripts/rescore_rollouts.py` 对完全相同的保存轨迹离线重放，没有重新调用模型。

| 模型 | strict-v2 总体 | 目标第一位 | 目标第二位 | 目标订单选择率 |
| --- | ---: | ---: | ---: | ---: |
| 原始 Qwen3-1.7B | 1/120 | 1/60 | 0/60 | 5/120 |
| v8 adapter | 60/120 | 60/60 | 0/60 | 60/120 |

v8 相比基座增加 59 个成功、损失 0 个，任务级 bootstrap 差值区间为
`[0.400, 0.583]`。但增益全部来自目标第一位；v8 在目标第二位的五个业务族
均为 `0/12`，且目标订单选择为 `0/60`。这证明当前 adapter 依赖列表第一项，
不能声称订单位置泛化。单次模型训练和固定 greedy 解码仍不能量化训练种子方差。

离线复评分命令：

```bash
python scripts/rescore_rollouts.py --tasks artifacts/position-shifted-tasks.jsonl \
  --rollouts artifacts/base-position-probe-120.jsonl \
  --output artifacts/base-position-probe-120-strict-v2.jsonl
python scripts/rescore_rollouts.py --tasks artifacts/position-shifted-tasks.jsonl \
  --rollouts artifacts/v8-position-probe-120.jsonl \
  --output artifacts/v8-position-probe-120-strict-v2.jsonl
```
