# v9 有界开发实验方案

本轮先记录方案再训练。它是开发集驱动的诊断，不能报告为盲测。

## 依据

v8 原始生成把 `For account` 放进了验证邮箱参数，解析前后完全一致。
20 条验证任务若由外部提供正确的第一次 lookup，后续模型调用可完成 20/20；
这是辅助成绩，不能替代独立成绩。训练任务上的完整交叉探针分别达到
79/80（四种结构）和 99/100（新增陌生前缀）。因此不能断言所有陌生实体
都失败，也不能断言单靠随机邮箱增强即可修复。

## 数据和预算

从固定的 v8 训练任务构建视图，验证、旧测试和冻结确认集只用于实体隔离
审计，不进入训练。仍为 300 个独立任务。

- 300 条完整轨迹：每个业务族内部轮转分配 4 个问法 × 5 个邮箱结构，
  每个业务族的每个单元恰有 3 条。
- 300 条对应恢复轨迹：只监督执行错误之后的正确续写。
- 每族选 4 个训练任务，逐任务展开全部 20 个单元，得到 400 条首调用。
  首调用前缀保持原始系统提示和十个工具，不使用外部邮箱提取器。
- 每个完整轨迹和首调用所依赖的任务都重放专家；随后审计真实 Trainer 掩码。

合计 1000 视图。两个候选均从原始 Qwen3-1.7B 初始化，LoRA r16/alpha32，
单 epoch，batch1、累积8、seed45、最大4096 tokens，预期125个更新步。
候选仅学习率不同：2e-4 与 5e-5。这是本轮内部等预算对照；v8 使用900视图，
所以 v8→v9 的差异不能全部归因于数据覆盖。

## 判定与止损

先对两个候选评测同一 validation 前20条，greedy、最多512新tokens、原始
严格奖励、原始任务提示，不给 oracle、不修改解析或邮箱。所有调用保存并重放。
只有首20条至少10条严格成功且五个族均有成功的候选才进入完整 validation-60。
完整60条达到至少50%且每族都有成功，才满足后续RL启动门槛。

本轮最多两个训练候选，不加新训练轮次、不根据冻结确认集改模型。
若两者都失败，保留负结果与原始生成，停止同类SFT扩充并评估任务范围或
基础模型；不能把 oracle 成绩、训练loss、探针成绩包装为完整任务增益。

## 复现命令

以下在项目根目录运行，`MODEL_DIR` 指向原始本地 Qwen3-1.7B。
v8源数据按README中的增强与首调用课程命令先生成。

```bash
python scripts/diagnose_sft.py --model-id "$MODEL_DIR" \
  --adapter outputs/qwen3-1.7b-tool-sft-repair-v8 \
  --tasks data/generated/tasks_validation.jsonl --mode oracle --limit 20 \
  --output artifacts/v8-oracle-first-20.jsonl
python scripts/diagnose_sft.py --model-id "$MODEL_DIR" \
  --adapter outputs/qwen3-1.7b-tool-sft-repair-v8 \
  --tasks data/diverse-v8/tasks_train.jsonl --mode factorial --limit 5 \
  --output artifacts/v8-factorial-prefix-first-5.jsonl
python scripts/create_factorial_training.py \
  --source data/diverse-v8/tasks_train.jsonl \
  --heldout data/generated/tasks_validation.jsonl data/generated/tasks_test.jsonl \
    data/confirmation/tasks_test.jsonl \
  --output-dir data/factorial-v9 \
  --summary artifacts/factorial-v9-training-audit.summary.json
python scripts/audit_sft_masks.py --model-id "$MODEL_DIR" \
  --data data/factorial-v9/sft_train.jsonl data/factorial-v9/sft_repair_lookup_train.jsonl \
  --output artifacts/qwen3-sft-factorial-v9-mask-audit.summary.json
python scripts/train_sft.py --model-id "$MODEL_DIR" \
  --train data/factorial-v9/sft_train.jsonl \
  --train-extra data/factorial-v9/sft_repair_lookup_train.jsonl \
  --validation data/generated/sft_validation.jsonl \
  --validation-extra data/generated/sft_repair_validation.jsonl \
  --output-dir outputs/qwen3-1.7b-tool-sft-factorial-v9-low \
  --epochs 1 --learning-rate 5e-5 --batch-size 1 --gradient-accumulation 8 --seed 45
python scripts/evaluate_model.py --model-id "$MODEL_DIR" \
  --adapter outputs/qwen3-1.7b-tool-sft-factorial-v9-low \
  --tasks data/generated/tasks_validation.jsonl --limit 20 \
  --max-new-tokens 512 --temperature 0 --seed 42 \
  --run-name qwen3-sft-factorial-v9-low-validation-20 \
  --output artifacts/qwen3-sft-factorial-v9-low-validation-20.jsonl
python scripts/audit_rollouts.py --tasks data/generated/tasks_validation.jsonl \
  --rollouts artifacts/qwen3-sft-factorial-v9-low-validation-20.jsonl \
  --output artifacts/qwen3-sft-factorial-v9-low-validation-20-replay.summary.json
```

高学习率候选只将 `5e-5` 改为 `2e-4`，并将输出名称中的 `low` 改为 `high`。
原80调用诊断对应新增陌生前缀前的四种结构；现脚本固定生成100调用，
原始80调用轨迹单独保存，避免将不同诊断版本混淆。

## 实际执行结果（2026-10-12）

两个候选均从基础模型训练1000视图、1 epoch / 125更新步，数据哈希一致。
高学习率耗时595.44秒、训练loss 0.02911；低学习率耗时597.90秒、loss 0.07881。
980条专家任务审计全部通过，空操作捷径和实体重叠均为0；1000条真实训练掩码
检查中的提示泄漏、监督空缺、前缀不匹配和截断均为0。这些是实现检查。

| 原提示开发验证 | 严格成功 | 终态完成 | 错误邮箱任务 | 轨迹重放 |
| --- | ---: | ---: | ---: | ---: |
| v9 high，2e-4 | 0/20 | 0/20 | 20/20 | 20/20一致 |
| v9 low，5e-5 | 0/20 | 0/20 | 20/20 | 20/20一致 |

两个候选各自的前5条原始生成均继续把 `For account` 写入邮箱，解析前后相同。
与v8及两候选之间的逐题比较均无新增成功，也无丢失成功。全零差值的bootstrap
区间退化为[0,0]，不能据此宣布总体等效或所有SFT方案无效。

按预设止损条件停止本轮同类SFT扩充；未扩测完整validation-60、未启动新GRPO、
未使用冻结确认集评测或选版本。负结果只适用于当前模型、训练预算和开发子集。

训练manifest、原始生成诊断、评测及重放摘要保存于`artifacts/`，模型权重仅在
训练服务器。adapter SHA256：

- high：`d517941700facaece9f1d7c07d4d1de9af07b0878de515a5c5cf02cffffdface`
- low：`f377ab8aeef7309faa9d302de6ab4914ce7b80f96778c4686d1d95cbf1afec34`

## 单独的提示格式诊断

固定v8 adapter和相同20条开发任务，只将请求中已有的邮箱显式列成字段：

```text
Customer email: <用户文本中的邮箱>
Request: <完整原请求>
```

不注入oracle调用、不修正输出、不修改环境或评分；首调用及后续调用全部由
模型生成。结果严格成功20/20、五族各4/4、工具有效率与合规率均为100%、
平均5步；20条轨迹重放一致，错误邮箱0条。原提示下同一adapter为0/20。

此结果标记`input_intervention=explicit-email-field-v1`，只能说明当前开发子集
在输入格式干预下改善，不能归因于v9训练或RL，也不能声称未见措辞泛化。
基础模型在此协议下的成绩、完整验证和请求扰动尚未评测。后续若研究该方案，
应先固定新协议并补这些对照，不沿用原协议的门槛声明。比较脚本拒绝把此类
干预轨迹混入原提示模型比较。

```bash
python scripts/diagnose_sft.py --model-id "$MODEL_DIR" \
  --adapter outputs/qwen3-1.7b-tool-sft-repair-v8 \
  --tasks data/generated/tasks_validation.jsonl --mode email-field --limit 20 \
  --output artifacts/v8-explicit-email-field-20.jsonl
python scripts/audit_rollouts.py --tasks data/generated/tasks_validation.jsonl \
  --rollouts artifacts/v8-explicit-email-field-20.jsonl \
  --output artifacts/v8-explicit-email-field-20-replay.summary.json
```

正式扩测可使用 `evaluate_model.py --input-intervention explicit-email-field-v1`。
该选项同时支持不加载 adapter 的基础模型和 v8 adapter；比较时必须保持相同的
输入干预、任务文件、seed、温度和生成预算。字段协议的结果不能与原始提示协议
直接做 SFT 增益比较。

## 字段协议的 validation-60 对照（2026-10-12）

在同一 `validation-60`、seed 42、greedy、512 新 token 和 `canonical-v4` 下，
只比较显式邮箱字段协议内的基座模型与 v8 adapter：

| 模型 | 严格成功 | 五族 | 轨迹重放 | 错误邮箱 |
| --- | ---: | ---: | ---: | ---: |
| 基座 Qwen3-1.7B | 0/60 | 各0/12 | 60/60一致 | 0 |
| v8 adapter | 60/60 | 各12/12 | 60/60一致 | 0 |

配对比较为60条全增益、无损失，探索性任务 bootstrap 区间为 `[1.0, 1.0]`。
这只能说明在这个**固定的新输入协议**和当前开发验证集上，v8 adapter 比基座
多完成了任务；不等价于跨训练种子或新分布的因果结论。原始提示协议下 v8
仍为0/20，因此不能把字段协议下的60/60改写为原协议的SFT增益。

为检查字段位置，v8 将字段放到完整请求之后，在 validation-20 上仍为20/20；
但这不是新的独立训练或泛化集。完整结果保存在 `artifacts/`。
