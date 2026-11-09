# 原有 Transformer 实验与比较边界

新增 RWKV 对照采用独立入口与实验协议，见 [RWKV 实验指南](rwkv_rosa_comparison.md)。本文数字仅属于原有 Transformer 实验，不可用作 RWKV 收益。

以下数字摘自本地保存的 `outputs/<run>/comparison.json`，用于追溯已有小规模实验。原始数据、日志和权重不随 Git 上传。这些实验不是多随机种子统计，也没有证明大模型上的同等收益。

## 对照模型

baseline 是从头训练的 Qwen 风格 Transformer；ROSA 侧采用相同主干，再加入检索注入。没有 RWKV 神经网络对照。Qwen 名称也用于 tokenizer，不表示加载 Qwen 预训练权重。

| Run | Train/val/test 文档数 | 参数量 baseline → ROSA | Test loss | Test PPL | Token accuracy |
| --- | --- | --- | --- | --- | --- |
| `minipile_doclocal_sam_qwen` | 4000/400/400 | 24.18M → 24.18M | 6.7436 → 6.3371 | 848.62 → 565.17 | 14.70% → 20.44% |
| `compare_small_online_v1_64docs` | 2000/200/200 | 15.959M → 15.967M | 7.2962 → 6.7363 | 1474.74 → 842.43 | 10.73% → 18.60% |
| `compare_small_online_v2_32docs` | 32/8/8 | 15.96M → 31.84M | 16.8664 → 16.5328 | 21134322 → 15139510 | 3.83% → 4.60% |

注意 V1 的目录名是历史名称，实际文档数以 JSON 为准。以上 seq_len=64，训练 1 epoch。ROSA 能读取当前文档更早的前缀，baseline 的 Transformer 只看当前 chunk，因此效果收益包含额外历史信息的帮助。

同配置 32/8/8 实验中，V1 → V2 准确率 4.42% → 4.60%，平均 step 35.50ms → 74.60ms；V2 参数量接近翻倍，尚需更大规模与 capacity-matched baseline 验证。

## 训练路径性能

同一 32/8/8 小配置的 stateful sync、fast sync、fast async，平均 step 分别为 95.35ms、59.36ms、35.50ms，测试 loss/accuracy 一致。这是不同 ROSA 实现的比较，不是相对普通 Transformer 的加速，也不是服务化推理吞吐。

## 本地 checkpoint 记录

| 路径（本地，Git 忽略） | 参数量 | 文件大小 |
| --- | --- | --- |
| `outputs/compare_small_online_v1_64docs/rosa_fused.pt` | 15.97M | 60.92 MiB |
| `outputs/minipile_doclocal_sam_qwen/rosa_fused.pt` | 24.18M | 92.27 MiB |
| `outputs/compare_small_online_v2_32docs/rosa_fused.pt` | 31.84M | 121.49 MiB |
| `outputs/qwen_vs_rosa/rosa_fused.pt` | 154.29M | 588.59 MiB |

以上权重为 FP32。154M 是早期实验，test loss 60.83 高于对应 baseline 58.57，不能按文件大小判断效果。训练恢复文件位于 `checkpoints/<model>/last.pt`，还包括优化器和训练进度。Git 不备份这些文件。
