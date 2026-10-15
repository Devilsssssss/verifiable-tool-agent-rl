# v10 位置平衡 SFT 多 seed 复验方案

本轮只检验 v10 结果对训练随机性的敏感程度。方案在 seed 46/47 训练前保存，
不修改训练数据、提示协议、超参数、确认任务或评分器，也不根据中间结果增加候选。

## 固定条件

- 从同一原始 `Qwen3-1.7B` 分别初始化 LoRA；
- 使用与 v10 seed 45 完全相同的 900 个训练视图及文件哈希；
- LoRA r16/alpha32/dropout0.05、all-linear、1 epoch、学习率 `2e-4`；
- batch 1、梯度累积 8、最大 4096 tokens，预期 113 个优化步；
- 新候选只使用训练 seed 46 和 47；
- 评测沿用同一 120 条位置平衡确认集、显式邮箱字段 v1、greedy、512 tokens、
  推理 seed 42 和 `strict-v2-target-inspection`。

训练 seed 45 的结果已经用于决定本轮复验，因此这 120 条确认任务不再是对整个
研究流程完全不可见的盲测集。本轮只能回答“保持方法不变时，训练随机性是否会
破坏已观察到的位置改善”，不能提供第二份独立分布泛化证据。

## 预注册判据

对 seed 46 和 47 分别读取：目标第二位严格成功、目标订单检查、总体严格成功和
五个任务族的分层结果。若两个新 seed 均满足以下条件，则记为通过本轮复验：

1. 目标第二位严格成功至少 `30/60`；
2. 目标第二位订单检查为 `60/60`；
3. 训练清单为 900 个样本、113 个优化步，且除训练 seed 外关键设置与数据哈希
   和 seed 45 一致；
4. 120 条保存轨迹全部通过数据库重放。

无论通过与否都报告三个 seed 的逐 seed 数字、均值和范围，不挑选最好 checkpoint。
任一新 seed 未达到门槛时，停止进入 RL；全部达到也只说明当前合成任务与输入协议
下的位置改善对这三个训练 seed 稳定，下一步仍应处理退货策略边界。

## 实际结果

两组复验使用了相同的 900 个训练视图、113 个优化步和冻结的 120 条确认任务。
训练清单中的关键设置、数据哈希与 seed45 一致；运行时间和训练指标另行保留。
seed46 的训练损失为 `0.03428`，seed47 为
`0.03208`。两组保存的 120 条轨迹均通过数据库重放，原始分数未改变，错误邮箱
任务均为 0。

| 训练 seed | 第一位严格成功 | 第二位严格成功 | 第二位目标订单检查 | 总体严格成功 |
| ---: | ---: | ---: | ---: | ---: |
| 45（原 v10） | 58/60 | 47/60 | 60/60 | 105/120 |
| 46 | 53/60 | 48/60 | 55/60 | 101/120 |
| 47 | 54/60 | 47/60 | 48/60 | 101/120 |

三个训练 seed 的总体严格成功均值为 `102.33/120`（85.28%），范围
`101–105/120`；第一位均值 `55/60`，范围 `53–58/60`；第二位均值
`47.33/60`，范围 `47–48/60`；第二位目标检查均值 `54.33/60`，范围
`48–60/60`。这些是三个 seed 的描述统计，不是 360 个独立测试任务或总体置信区间。

seed46 和 seed47 都达到第二位严格成功至少 `30/60`，但目标订单检查分别为
`55/60` 和 `48/60`，没有达到预注册的 `60/60`。因此本轮复验判定为**未通过**，
不能把 v10 的目标选择改善表述为已满足预设的稳定性门槛，也不进入 GRPO/RL。

两个复验 seed 各有 19 个失败，全部属于退货分支。其余三个业务族均为 `24/24`。

| 任务族 | seed45 第一位/第二位 | seed46 第一位/第二位 | seed47 第一位/第二位 |
| --- | ---: | ---: | ---: |
| 处理中订单取消 | 12/12、12/12 | 12/12、12/12 | 12/12、12/12 |
| 已发货订单升级 | 12/12、12/12 | 12/12、12/12 | 12/12、12/12 |
| 低价换货 | 12/12、12/12 | 12/12、12/12 | 12/12、12/12 |
| 期限内退货 | 10/12、8/12 | 5/12、11/12 | 6/12、11/12 |
| 过期退货升级 | 12/12、3/12 | 12/12、1/12 | 12/12、0/12 |

逐条轨迹呈现三种失败模式：

- 把 12 天内的目标退货误判为过期并升级：seed46 为 8 条，seed47 为 7 条。
- 目标在第二位时，只检查第一笔订单便升级：seed46 为 5 条，seed47 为 12 条，
  均属于过期退货。尽管终态和无修改行为满足升级任务，仍因没检查目标而严格失败。
- 检查正确的 45 天过期订单后调用一次 `request_return`，再升级：seed46 为 6 条，
  seed47 为 0 条。最终状态虽正确，`return_window_closed` 违规仍使严格成功为 0。

因此订单位置与退货决策仍有交互，不能说所有 seed 都已解决目标选择。下一轮应先做
退货边界与目标检查的独立数据对照，不改变本轮评分或预注册门槛，也不继续增加 RL 预算。

本轮汇总文件为 `artifacts/v10-seed-replication-comparison.summary.json`；逐组重放
摘要为 `artifacts/v10-seed46-position-v10-confirmation-120-replay.summary.json`
和 `artifacts/v10-seed47-position-v10-confirmation-120-replay.summary.json`。

## 复现命令

```bash
for SEED in 46 47; do
  python scripts/train_sft.py --model-id models/Qwen3-1.7B \
    --train data/position-v10/sft_train.jsonl \
    --train-extra data/position-v10/sft_repair_lookup_train.jsonl \
    --validation data/generated/sft_validation.jsonl \
    --validation-extra data/generated/sft_repair_validation.jsonl \
    --output-dir "outputs/qwen3-1.7b-tool-sft-position-v10-seed${SEED}" \
    --epochs 1 --learning-rate 2e-4 --batch-size 1 \
    --gradient-accumulation 8 --seed "${SEED}"

  python scripts/evaluate_model.py --model-id models/Qwen3-1.7B \
    --adapter "outputs/qwen3-1.7b-tool-sft-position-v10-seed${SEED}" \
    --tasks artifacts/position-v10-confirmation-tasks.jsonl \
    --output "artifacts/v10-seed${SEED}-position-v10-confirmation-120.jsonl" \
    --run-name "v10-seed${SEED}-position-v10-confirmation-120" \
    --input-intervention explicit-email-field-v1 \
    --limit 120 --max-new-tokens 512 --temperature 0 --seed 42

  python scripts/audit_rollouts.py \
    --tasks artifacts/position-v10-confirmation-tasks.jsonl \
    --rollouts "artifacts/v10-seed${SEED}-position-v10-confirmation-120.jsonl" \
    --output "artifacts/v10-seed${SEED}-position-v10-confirmation-120-replay.summary.json"
done

python scripts/analyze_seed_replication.py \
  --tasks artifacts/position-v10-confirmation-tasks.jsonl \
  --baseline artifacts/v8-position-v10-confirmation-120.jsonl \
  --candidate seed45=artifacts/v10-position-v10-confirmation-120.jsonl \
  --candidate seed46=artifacts/v10-seed46-position-v10-confirmation-120.jsonl \
  --candidate seed47=artifacts/v10-seed47-position-v10-confirmation-120.jsonl \
  --manifest seed45=artifacts/qwen3-sft-position-v10-training.summary.json \
  --manifest seed46=artifacts/qwen3-sft-position-v10-seed46-training.summary.json \
  --manifest seed47=artifacts/qwen3-sft-position-v10-seed47-training.summary.json \
  --replay seed45=artifacts/v10-position-v10-confirmation-120-replay.summary.json \
  --replay seed46=artifacts/v10-seed46-position-v10-confirmation-120-replay.summary.json \
  --replay seed47=artifacts/v10-seed47-position-v10-confirmation-120-replay.summary.json \
  --required-replication seed46 --required-replication seed47 \
  --position-2-threshold 30 \
  --output artifacts/v10-seed-replication-comparison.summary.json
```

模型、原始轨迹和生成数据不随 Git 提交；基座、v8 和 seed45 的训练/生成命令见
README 与 [v10 协议](position_v10_protocol.md)。本页汇总命令依赖这些已生成轨迹。
分析器同时校验 120 条任务的 60/60 平衡、训练配置和重放摘要，不能用未完成的重放
报告得出复验通过。
