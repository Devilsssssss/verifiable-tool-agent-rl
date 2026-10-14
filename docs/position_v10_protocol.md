# v10 位置平衡 SFT 预注册方案

本轮由订单位置探针的失败驱动，属于开发实验。方案和新确认集哈希必须在训练前
保存；只训练一个候选，不根据结果增加轮次、学习率或随机种子。

## 单变量训练对照

对照模型为 v8，候选 v10 从同一原始 `Qwen3-1.7B` 初始化。两者保持以下设置：

- 300 个独立训练任务、900 个训练视图；
- 300 条正常轨迹、300 条错误恢复轨迹、300 条首调用轨迹；
- LoRA r16/alpha32/dropout0.05、all-linear、1 epoch；
- 学习率 `2e-4`、batch 1、梯度累积 8、seed 45、最大 4096 tokens；
- 同一 validation 数据和同一 SFT mask/格式审计。

唯一训练干预发生在包含订单信息的 600 条正常/修复轨迹：使用 seed
`20261016` 在每个业务族内将 60 个任务严格分成目标第一位 30 条、第二位
30 条，订单 ID、订单项外键、目标和专家轨迹中的引用一起交换。300 条首调用
视图不含订单信息，直接从 v8 文件逐字节复制，不重新抽样问法。

## 冻结评测

训练前使用 seed `20261017` 和独立的 `40000000` ID offset 生成另一批 120 条
新实体确认任务，再用 seed `20261018` 在每个业务族内平衡目标位置为 12/12。
该集合与旧确认集也要通过 ID、邮箱和商品实体隔离审计，不进入训练、提示词修改
或 checkpoint 选择。

v8 与 v10 均使用显式邮箱字段 v1、greedy、512 tokens、seed 42 和
`strict-v2-target-inspection`。主要指标按以下顺序读取：

1. 目标第二位严格成功率；
2. 目标订单选择率；
3. 总体严格成功率；
4. 五个业务族的位置分层结果。

只训练一个候选。无论结果好坏都保存逐条轨迹；若第二位低于 `30/60`，停止同类
SFT 扩充并分析商品匹配与订单选择机制，不启动 GRPO。即使达到门槛，也只能说明
单一训练 seed、固定输入协议和合成任务上的改善。

## 复现命令

```bash
python scripts/create_confirmation.py --seed 20261017 --count 120 \
  --id-offset 40000000 --output-dir data/position-v10-confirmation \
  --summary artifacts/position-v10-confirmation-audit.summary.json
python scripts/create_position_shifted.py \
  --input data/position-v10-confirmation/tasks_test.jsonl \
  --output artifacts/position-v10-confirmation-tasks.jsonl \
  --summary artifacts/position-v10-confirmation-tasks.summary.json --seed 20261018

python scripts/create_position_training.py --data-dir data/diverse-v8 \
  --output-dir data/position-v10 --seed 20261016 \
  --heldout data/generated/tasks_validation.jsonl data/generated/tasks_test.jsonl \
    data/confirmation/tasks_test.jsonl \
  --summary artifacts/position-v10-training-audit.summary.json

python scripts/audit_sft_masks.py --model-id models/Qwen3-1.7B \
  --data data/position-v10/sft_train.jsonl \
    data/position-v10/sft_repair_lookup_train.jsonl \
  --output artifacts/qwen3-sft-position-v10-mask-audit.summary.json

python scripts/train_sft.py --model-id models/Qwen3-1.7B \
  --train data/position-v10/sft_train.jsonl \
  --train-extra data/position-v10/sft_repair_lookup_train.jsonl \
  --validation data/generated/sft_validation.jsonl \
  --validation-extra data/generated/sft_repair_validation.jsonl \
  --output-dir outputs/qwen3-1.7b-tool-sft-position-v10 \
  --epochs 1 --learning-rate 2e-4 --batch-size 1 \
  --gradient-accumulation 8 --seed 45
```
