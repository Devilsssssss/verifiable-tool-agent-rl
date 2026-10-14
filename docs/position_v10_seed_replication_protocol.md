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
done
```
