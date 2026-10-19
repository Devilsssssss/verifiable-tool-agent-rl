# Verifiable Tool-Agent RL

一个研究 Agentic RL 与大模型后训练的实验项目：在有状态的本地零售环境中训练小模型完成多轮工具调用，并用数据库终态与策略约束给出可执行、可复核的奖励。

最新进展：位置平衡 SFT 的 v10 基线在新冻结的 120 条退货泛化任务上为 **58/120、54/120、46/120**。预注册的 return-v11 等视图年龄分布干预在 seed45/46 上分别得到 **56/120、81/120**，但在旧确认集回归上都低于匹配 v10（96/120 对 105/120、101/120）。随后完成了 return-boundary-v1 的 seed45/46 监督边界对照：holdout 为 **78/120、98/120**，但旧确认集回归为 **87/120、103/120**，一个 seed 明显回归、一个 seed 只是不确定的小幅变化，目标订单检查率也未达到全量门槛。因此不能声称稳定的 Agentic RL 或后训练增益，当前不启动 GRPO。详见[边界对照结果](docs/return_boundary_v1_results.md)、[基线结果与限制](docs/return_holdout_baseline_results.md)、[v11 结果报告](docs/return_v11_results.md)和[边界训练协议](docs/return_boundary_training_protocol.md)。

对外快速入口：[一页项目说明](docs/project_brief.md)；实验路线与决策记录见[实验索引](docs/experiment_index.md)。

边界数据生成、语义审计、Qwen3-1.7B seed45/46 训练、120 条 holdout、旧确认集回归、轨迹重放和配对分析均已完成。冻结产物位于 `data/return-boundary-v1/`，审计摘要为 `artifacts/return-boundary-v1-training-audit.summary.json`，结果见 `docs/return_boundary_v1_results.md`。训练使用远端锁定依赖，不能把本地专家演示替代模型结果。

边界对照已经完成，但两个 seed 的目标订单检查率仍为 90.0%--92.5%，且旧确认集出现 seed-sensitive 回归；在目标检查和位置 2 门槛解决前不进入 GRPO。

面试复核入口：先看 [项目总结](docs/project_summary.md)，再运行下面的失败差异分析（需要已生成的评测产物）：

```bash
python scripts/analyze_failure_deltas.py \
  --tasks data/return-holdout/tasks_test.jsonl \
  --baseline artifacts/return-holdout-seed46.jsonl \
  --candidate artifacts/return-v11-seed46-holdout.jsonl \
  --output artifacts/return-v11-46-holdout-failure-deltas.summary.json
```

该分析只读取保存的轨迹，不参与训练、选 checkpoint 或修改评分协议。

## 最小演示

这条命令不需要 GPU、模型权重或外部服务，会运行一个脚本专家并展示工具调用、数据库终态和严格评分：

```bash
python scripts/demo.py --family return_eligible
```

安装开发依赖后即可运行：

```bash
python -m pip install -e ".[dev]"
```

演示使用的是专家策略，不代表模型评测；模型训练和评测需要额外的训练依赖、数据文件与本地模型权重。

## 自动检查

仓库提交会自动运行 `pytest` 和 `ruff`。本地可用下面的命令复现：

```bash
python -m pytest
python -m ruff check src scripts tests
```

## 研究问题

在相同 SFT 初始化和生成预算下，过程奖励是否比只有任务成败的稀疏奖励更能提升：

- 端到端任务成功率；
- 策略合规率；
- 有效工具调用率；
- 未见用户、商品、订单标识和提示措辞上的泛化能力。

当前训练实验使用 `Qwen3-1.7B + LoRA`。`Qwen2.5-Coder-0.5B-Instruct` 只用于低成本验证训练、解析和评测链路，不作为最终结论。目前尚未获得可信的 RL 增益证据。

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

当前包含五类任务：处理中订单取消、已发货订单拒绝并升级、低价换货并退差价、期限内退货、过期退货拒绝并升级。训练集和旧评测集中的目标订单始终排在第一；另设的订单位置探针在每个业务族内将目标位置严格平衡为 `12/12`，专门检验位置捷径。

环境强制检查：先查询再使用标识、修改前检查订单、换货前检查库存、只向原支付方式退还精确金额、超过退货期或系统能力时保持业务状态不变并升级人工。

## 可验证奖励

`sparse` 仅使用最终任务成功 `1/0`。`shaped` 同时考虑任务结果、策略合规、有效工具调用率、正确计划前缀进度和步骤效率：

```text
task outcome + policy compliance + valid tool-call rate + correct-prefix progress + efficiency
```

成功不是由模型自报，而是由数据库终态和违规轨迹共同决定。直接使用隐藏 ID、空操作和绕过检查的捷径都不能获得成功奖励。

当前默认评分协议为 `strict-v2-target-inspection`：除终态正确且没有违规外，轨迹还必须实际检查目标订单。此前保存的实验使用 `strict-v1-with-terminal-diagnostics`；两种协议在摘要中明确区分，旧轨迹需先用 `scripts/rescore_rollouts.py` 离线复评分，不能直接混合比较。

当前严格成功指标会把 `invalid_tool_call`（例如多传参数）也记为违规，即使模型随后纠正了调用并达到目标，严格成功仍为 0。为分析恢复行为，额外报告 `terminal_goal_completion`（仅检查目标数据库终态）和 `format_recovery_rate`（只有格式错误、最终达到目标的任务数 / 出现格式错误的任务数）。存在业务政策违规或超预算的轨迹不能计入格式恢复成功。这两个诊断指标不改变严格成功或训练奖励。

针对升级任务的理由校验已作为独立的 `strict-v3-reason-code` 协议实现，默认评分和历史结果仍是 strict-v2。v3 的设计、兼容性和离线复评分方式见 [reason_code_protocol.md](docs/reason_code_protocol.md)。

## 数据与审计

固定种子生成 `300/60/120` 条训练、验证和测试任务。三个划分使用不同的 ID、邮箱、商品名称和提示模板；用户显示姓名不在实体重叠审计范围内。`scripts/audit_tasks.py` 会重放所有专家轨迹并检查重复、ID 泄漏、跨划分实体重叠、空操作和直接捷径。

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

### 格式修复与训练覆盖诊断

`canonical-v4` 将工具参数存为稳定的 JSON 字符串，避免 Arrow 合并不同工具的参数字段；训练与推理共用参数转换和排序后的 schema。600 条训练视图的真实 SFTTrainer 审计中，提示差异、掩码错误、截断均为 0。GRPO 原生工具循环也统一 JSON 排序，3900 个助手生成位置与 SFT/推理前缀逐 token 一致。

| 实验 | 评测 | 严格成功 | 错误邮箱任务 |
| --- | --- | ---: | ---: |
| 原始 1.7B，canonical 推理格式 | validation-20 | 0/20 | 5/20 |
| v3，仅更换 canonical 推理格式 | validation-20 | 0/20 | 20/20 |
| v4，相同 600 视图 / 1 epoch / seed 45，修复训练格式 | validation-20 | 0/20 | 20/20 |
| 同一 v4，在训练任务上的诊断 | train-10 | 10/10 | 未单独统计 |
| v5，增强训练实体与三种问法，600 视图 / 1 epoch / seed 45 | validation-20 | 0/20 | 20/20 |
| v6，增加 300 首调用视图，900 视图 / 1 epoch / seed 45 | validation-20 | 0/20 | 20/20 |
| v7，定向补充首调用问法，900 视图 / 1 epoch / seed 45 | validation-20 | 0/20 | 20/20 |
| v8，混合训练邮箱结构，900 视图 / 1 epoch / seed 45 | validation-20 | 0/20 | 20/20 |

v4 会把 `For account` 一起复制进邮箱。这支持继续检查训练覆盖，而不能把格式审计通过当成行为能力改善。训练任务诊断不是泛化成绩。

`scripts/augment_training.py` 只变换训练集的邮箱、商品名称和三种训练措辞，保持订单 ID、金额、业务状态、目标和预算不变；输出仍为 300 个独立任务、600 个正常/修复视图。验证和测试文件不变。新数据与开发划分一起通过 480 条专家、空操作、隐藏 ID 和实体重叠审计。

新确认集由 `scripts/create_confirmation.py` 固定 seed 20261012 生成，120 条任务包含新 ID、邮箱、商品名称及每类一种新措辞。用户显示姓名仍沿用原测试任务；实体重叠审计检查 ID、邮箱和商品名称，不包括显示姓名。该确认集保持目标订单排在第一，不能证明订单位置泛化；不用于选择检查点或训练数据。具体对照与进入 RL 的门槛见 [实验口径](docs/format_control.md)。

`scripts/probe_email.py` 固定 10 个验证任务的邮箱、商品，只改变用户问法，单独测首个 `lookup_user` 调用。v5 在原问法下为 0/10，三组已见训练问法分别为 10/10。该诊断支持措辞敏感性，不能当成完整任务成功率。结果见 `artifacts/qwen3-sft-repair-v5-email-probe.summary.json`。

`scripts/create_lookup_curriculum.py` 因此增加 300 条训练用首调用监督，使用 12 种邮箱句式，保留原有 300 正常和 300 修复视图。v6 共 900 个视图，仍只有 300 个独立业务任务；这不是等预算的单变量实验。新增样本不包含验证或确认集任务，邮箱查询逐条重放核验。v6 的 900 个视图掩码审计通过，4200 个助手生成位置与 GRPO 原生模板逐 token 一致，但 validation-20 仍为 0/20。

v7 的训练课程定向补充四种 `For account + 邮箱` 句式（训练实体及请求），视图总数仍为 900。通用前缀与验证问法重叠，因此后续验证提升只能作为开发诊断；未见措辞的结论需要另用冻结确认集。

v7 的交叉首调用诊断使用 5 条开发验证任务：原问法配训练邮箱为 5/5；首调用训练问法配验证邮箱为 0/5；原问法配验证邮箱为 0/5；原始首调用训练样本为 5/5。三组先前的完整训练问法配验证邮箱则各为 5/5。结果支持邮箱拼写与问法交互，不能仅归因于问法，也不是完整任务成功率。商品问法未做完全交叉控制。

v8 将训练邮箱混合成短数字、长数字和哈希结构，覆盖 `example.test` 及三个训练域名，不使用开发或确认集的邮箱、保留前缀。保持 300 个独立任务及 900 个视图，从同一原始模型训练 1 epoch / seed 45；数据结构选择来自开发诊断，因此验证结果仍属探索性结果。

v8 的 900 视图掩码/格式审计全部通过，600 条跨划分专家审计通过，但 validation-20 仍为 0/20、错误邮箱 20/20；交叉首调用探针的七组计数与 v7 相同。严格评分与原始轨迹重放一致，未据此启动新 RL。详见 [失败复盘与下一轮设计](docs/sft_failure_review.md)。

2026-10-12 的原始输出诊断确认：前五条验证请求中，模型直接把 `For account`
生成进邮箱参数；解析前后内容一致。给前20条任务外部提供正确的一次 lookup 后，
模型后续调用达到 **辅助成功20/20**，重放通过；这是诊断，不能当成独立成功率。
只用训练任务状态、固定商品与请求的交叉首调用探针为79/80；新增陌生邮箱前缀后
为99/100。它们只评参数提取，不能代表业务任务成功或证明泛化已解决。

本轮有界SFT使用300个独立训练任务、1000视图，交叉覆盖问法与邮箱结构，并比较
2e-4和5e-5两个学习率。任务/实体审计及1000条真实训练掩码检查通过。方案与
止损条件及实际结果见 [v9实验方案与结果](docs/factorial_v9_protocol.md)：两个候选
均完成125个更新步，原提示下严格成功均为0/20、错误邮箱均为20/20，40条轨迹
重放一致。本轮停止原始提示协议下的同类SFT扩充，未启动新RL；显式字段协议的
validation-60与冻结确认集结果见下文。

另用v8进行提示格式诊断：只将用户请求中已出现的邮箱显式列成字段，全部工具
调用仍由模型生成，在相同20条开发任务上严格成功20/20、重放一致。这是
**提示干预结果**，不是SFT或RL增益，也不是冻结确认集的泛化成绩。

随后固定该字段协议扩测 validation-60：基座模型为0/60，v8 adapter为60/60，
五个业务族均为12/12，逐条重放一致；邮箱字段放在请求末尾时 v8 validation-20
仍为20/20。该对照只说明新输入协议下 v8 相对基座的开发集差异，不能改写为
原始提示协议的增益。

在不参与选模的冻结确认集 120 条任务上，同一字段协议下基座为2/120，v8
adapter为120/120，五个业务族均为24/24，逐条重放一致。配对比较为118条
增益、0条损失，任务级 bootstrap 区间为[0.958, 1.000]；这支持字段协议下
对新实体和新措辞的差异，但不改变原始提示协议 validation-20 为0/20的事实，
也不证明订单排列泛化。

随后冻结模型和字段协议，在另一批新实体任务上运行 120 条订单位置探针。每族
各有 12 条目标在第一位、12 条在第二位。初始 `strict-v1` 记为基座1/120、v8
67/120；轨迹审计发现其中7条拒绝/升级任务虽然检查了错误订单，仍因业务终态
未改变而被判成功。因此在看到结果后新增 `strict-v2-target-inspection`，要求实际
检查目标订单，并保留v1原始结果，不将修订冒充预注册指标。

按v2离线重放同一批轨迹，基座为1/120，v8为60/120；v8在目标第一位时60/60，
第二位时0/60，五个业务族的第二位置均为0/12。目标订单选择率也恰好是第一位
60/60、第二位0/60，说明模型始终选择列表第一项。这个负结果否定了位置泛化，
并把下一步从继续调奖励改为先构造位置随机化训练对照。完整协议和结果见
[订单位置泛化探针](docs/order_position_probe.md)。

随后预注册等预算的v10位置平衡SFT：仍使用300个独立训练任务、900个视图、
1 epoch和seed 45，只随机化600条正常/修复视图中的目标订单位置，300条首调用
视图与v8逐字节一致。训练前另用新实体和新位置seed冻结120条确认任务。相同
`strict-v2`评测下，v8为60/120，v10为105/120；目标第二位从0/60提升到47/60，
第一位为58/60。v10在120/120任务中都检查了目标订单，说明本轮不再是固定选择
第一项或第二项，但仍有15个退货任务失败。新增47条成功、损失2条，净增加45条；任务级配对
bootstrap区间为[0.283, 0.467]。随后固定数据复验seed46/47，均为101/120，第二位
分别为48/60和47/60，但第二位目标订单检查仅55/60和48/60，未达到预注册的60/60门槛，
因此暂不进入RL。全部轨迹重放一致，完整口径见
[v10位置平衡SFT](docs/position_v10_protocol.md)和
[seed复验方案与结果](docs/position_v10_seed_replication_protocol.md)。

实验概述与结论边界见 [项目总结](docs/project_summary.md)；设计与评测相关问题见 [项目问答](docs/project_qa.md)。

旧两组 30 步 GRPO 日志的奖励审计显示：稀疏奖励 29/30 步、过程奖励 28/30 步的组内奖励标准差为零。均值奖励很高也可能只是重复相同轨迹。`scripts/audit_training_logs.py` 保存原日志哈希和原生统计；`scripts/probe_reward_groups.py` 在训练任务上采样，预检两种奖励是否具有组内差异。推理预检与原生 GRPO 工具循环不同，不能互相替代。

`scripts/compare_runs.py` 对保存的逐条轨迹做配对比较，拒绝不同数据哈希、任务集合、生成预算、格式或评分口径的结果；任务级 bootstrap 不等于多训练种子的不确定性。

## 复现

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"
# 已验证的 Linux / Python 3.12 训练机直接依赖版本，并非完整传递依赖锁。
pip install -r requirements-training.lock

python scripts/generate_tasks.py
python scripts/audit_tasks.py
python scripts/run_baselines.py
pytest
ruff check .
python scripts/demo.py
python scripts/demo.py --family return_eligible --inject-format-error
```

两个演示命令已在不安装任何项目依赖的新 Python 虚拟环境中执行通过；使用脚本专家，不加载模型。训练环境的实际版本另由训练摘要记录。

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
  --output-dir outputs/qwen3-1.7b-tool-sft-repair-v4 \
  --epochs 1 --learning-rate 2e-4 \
  --batch-size 1 --gradient-accumulation 8 --seed 45

python scripts/audit_sft_masks.py \
  --model-id "$MODEL_DIR" \
  --data data/generated/sft_train.jsonl data/generated/sft_repair_train.jsonl \
  --output artifacts/qwen3-sft-repair-v4-mask-audit.summary.json
```

混合版共有 600 个训练视图，来自 300 个独立任务；验证集 120 个视图来自 60 个独立任务。正常视图监督完整正确调用序列，修复视图的错误调用和执行报错仅作上下文。当前训练代码采用 canonical-v4 格式，不复现历史 v1 的全轨迹损失或 v3 的 Arrow 参数结构。

生成训练覆盖增强数据及冻结的新确认集：

```bash
python scripts/augment_training.py
python scripts/create_confirmation.py

python scripts/train_sft.py \
  --model-id "$MODEL_DIR" \
  --train data/diverse/sft_train.jsonl \
  --train-extra data/diverse/sft_repair_train.jsonl \
  --validation data/generated/sft_validation.jsonl \
  --validation-extra data/generated/sft_repair_validation.jsonl \
  --output-dir outputs/qwen3-1.7b-tool-sft-repair-v5 \
  --epochs 1 --learning-rate 2e-4 \
  --batch-size 1 --gradient-accumulation 8 --seed 45
```

增加首调用课程后的训练命令：

```bash
python scripts/create_lookup_curriculum.py
python scripts/audit_sft_masks.py \
  --model-id "$MODEL_DIR" \
  --data data/diverse/sft_train.jsonl data/diverse/sft_repair_lookup_train.jsonl \
  --output artifacts/qwen3-sft-repair-v6-mask-audit.summary.json

python scripts/train_sft.py \
  --model-id "$MODEL_DIR" \
  --train data/diverse/sft_train.jsonl \
  --train-extra data/diverse/sft_repair_lookup_train.jsonl \
  --validation data/generated/sft_validation.jsonl \
  --validation-extra data/generated/sft_repair_validation.jsonl \
  --output-dir outputs/qwen3-1.7b-tool-sft-repair-v6 \
  --epochs 1 --learning-rate 2e-4 \
  --batch-size 1 --gradient-accumulation 8 --seed 45
```

复现 v7 定向补充时，先生成单独目录以保留 v6 的数据：

```bash
python scripts/create_lookup_curriculum.py --profile v7 \
  --output-dir data/curriculum-v7 \
  --summary artifacts/lookup-curriculum-v7.summary.json
# 使用上面的 v6 训练命令，并修改两个参数：
# --train-extra data/curriculum-v7/sft_repair_lookup_train.jsonl
# --output-dir outputs/qwen3-1.7b-tool-sft-repair-v7
```

复现 v8 邮箱结构覆盖：

```bash
python scripts/augment_training.py --profile v8 --output-dir data/diverse-v8 \
  --confirmation data/confirmation/tasks_test.jsonl \
  --summary artifacts/diverse-v8-training-audit.summary.json
python scripts/create_lookup_curriculum.py --profile v7 --data-dir data/diverse-v8 \
  --summary artifacts/lookup-curriculum-v8.summary.json
python scripts/audit_sft_masks.py --model-id "$MODEL_DIR" \
  --data data/diverse-v8/sft_train.jsonl data/diverse-v8/sft_repair_lookup_train.jsonl \
  --output artifacts/qwen3-sft-repair-v8-mask-audit.summary.json
python scripts/train_sft.py --model-id "$MODEL_DIR" \
  --train data/diverse-v8/sft_train.jsonl \
  --train-extra data/diverse-v8/sft_repair_lookup_train.jsonl \
  --validation data/generated/sft_validation.jsonl \
  --validation-extra data/generated/sft_repair_validation.jsonl \
  --output-dir outputs/qwen3-1.7b-tool-sft-repair-v8 \
  --epochs 1 --learning-rate 2e-4 --batch-size 1 --gradient-accumulation 8 --seed 45
python scripts/probe_email.py --model-id "$MODEL_DIR" \
  --adapter outputs/qwen3-1.7b-tool-sft-repair-v7 \
  --tasks data/generated/tasks_validation.jsonl --limit 5 \
  --lookup-data data/curriculum-v7/sft_lookup_train.jsonl \
  --output artifacts/qwen3-sft-repair-v7-lookup-controls.jsonl
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

复现 v3 的固定 20 条测试诊断：

```bash
python scripts/evaluate_model.py \
  --model-id "$MODEL_DIR" \
  --adapter outputs/qwen3-1.7b-tool-sft-repair-v3 \
  --tasks data/generated/tasks_test.jsonl \
  --output artifacts/qwen3-1.7b-tool-sft-repair-v3-test-20.jsonl \
  --run-name qwen3-1.7b-tool-sft-repair-v3-test-20 \
  --limit 20 --max-new-tokens 512 --temperature 0 --seed 42 \
  --input-format legacy-v3

python scripts/audit_rollouts.py \
  --tasks data/generated/tasks_test.jsonl \
  --rollouts artifacts/qwen3-1.7b-tool-sft-repair-v3-test-20.jsonl \
  --output artifacts/qwen3-1.7b-tool-sft-repair-v3-test-20-replay.summary.json
```

模型权重、adapter、生成数据和逐条轨迹默认不进入 Git。可公开的摘要保存在 `artifacts/*summary.json`。

## 当前限制

- 当前测试集验证未见实体和未见措辞，不等同于未见任务类型的组合泛化。
- v10的seed46/47复验保持了总体成功率，但未完全复现目标订单检查门槛；尚未证明跨训练seed、任务生成机制或真实系统的订单选择泛化。
- v10 seed45的15个失败全部在退货分支；seed46/47各失败19条，且分别有5条和12条第二位目标检查失败。退货策略与目标检查仍需处理。
- return-boundary-v1 的 seed45/46 已完成：退货 holdout 为 78/120、98/120，旧确认集回归为 87/120、103/120；seed45 出现明显回归，seed46 只有不确定的小幅净增，位置 2 和目标订单检查仍未达到预注册门槛。因此当前不启动 GRPO。
- 测试结果已用于诊断和设计后续训练数据，因此属于开发阶段的探索性评测；后续确认实验需要另设未用于调参的评测集。
- 当前严格奖励把可恢复的参数格式错误和业务政策违规都视为失败；应先明确奖励设计与恢复能力的研究目标，再开展新的 RL 对照。
- SQLite 工具环境比真实客服系统更干净，结论不能直接外推到生产环境。
- 过程奖励可能被利用，因此必须同时报告终态成功与合规审计，不能只报告平均奖励。
