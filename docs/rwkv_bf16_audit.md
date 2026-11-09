# RWKV-7 / ROSA BF16 专项数值复查

日期：2026-11-09。独立环境：`rosa-rwkv`，PyTorch 2.7.1+cu126。

## 结论先行

此前只检查 8 个 token 的证据不足。扩大验证后，BF16 不只是逐位不一致，确实会在一些位置改变下一 token 的 top-1。
发现并修复了一个 AMP 精度边界问题；但修复它并不能保证整体 BF16 分段预测等价。正式准确率对照配置现改为 FP32。

## 1. 实现问题与修复

上游参考 `RWKV7_OP` 将输入和循环状态 `.float()`，但在本项目训练入口的外层 BF16 autocast 中，
`state @ a @ b`、`v @ k`、`state @ r` 的实际矩阵乘法仍被下转换为 BF16。
因此，**缓存张量 dtype=FP32 不代表递归计算实际使用 FP32**。关闭 BF16 reduced-precision reduction 也不能代替关闭 autocast。

算子追踪的控制实验：固定相同 FP32 输入、12 token、单 head、head size=64，观察 48 次 bmm：

| 路径 | bmm 输入 / 输出 | 最终缓存 dtype | 相对真正 FP32 运算的缓存最大绝对差 |
| --- | --- | --- | ---: |
| 修复前，外层 BF16 autocast | BF16 / BF16 | torch.float32 | 0.000527836 |
| 修复后，递归内部明确禁用 autocast | FP32 / FP32 | torch.float32 | 0.000000000 |

修复位于 `rwkv_rosa/precision.py`，由 `load_runtime()` 安装；`third_party` 原始源码没有修改。
外层投影仍可使用 BF16，但 WKV 递归中的矩阵乘法、状态更新按 FP32 运算，最终输出再按 runtime dtype 转换。
测试直接追踪实际算子 dtype，并验证 AMP 内外的递归输出、缓存以及输入梯度一致，而非只检查返回张量 dtype。

## 2. 相同权重上的扩大验证

权重：`/hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth`。

SHA256：`d8f8ecd4af8d706f867ac8a74284f35c0e3b9f863e7b09bb4d4031480c8b28b2`。

三组都从同一份原始权重初始化，审计过程不做 optimizer 更新，避免不同训练轨迹干扰数值归因。
参数存储与此前冒烟一致：冻结主干 BF16、共同可训练部分和新版记忆参数 FP32；使用固定 seed 42。
关闭 TF32 和 BF16 reduced-precision reduction；对比 legacy AMP、FP32 WKV + BF16 投影、全 FP32 三种策略。

输入：

- SFT 模板样本：最多 64，实际 **62** 个输入 token（报告键 `sft64` 是长度上限）。
- 英文重复字段：256 token。
- 中文字段 batch=2：每条 64 token，共 128 个预测位置。
- 每个输入分别按二等分、不等长切分、prefill 后连续 16 次单 token decode 处理。

每组共 446 个不同输入位置，按三种切分重复比较，合计 **1338 次位置比较**；不是 1338 个独立随机样本。
以下翻转是“相同精度下，整段 vs 分段的下一 token argmax 不同”，不是任务正确率。

| 模型 | 修复前 BF16 翻转数 | 修复后 FP32 WKV + BF16 投影翻转数 | 全 FP32 翻转数 | FP32 最大 logit 绝对差 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 12 / 1338 | 13 / 1338 | 0 / 1338 | 0.00025368 |
| original | 14 / 1338 | 11 / 1338 | 0 / 1338 | 0.00024223 |
| improved | 11 / 1338 | 15 / 1338 | 0 / 1338 | 0.00029373 |

**不能宣称“给 WKV 加 FP32 边界就解决了整体 BF16 等价性”。**翻转数并未一致下降。修复消除了一个明确的递归运算降精度问题，
但其他投影和层间 BF16 舍入依然存在；误差互相抵消与放大的方式不同，局部更精确不保证每个样本的最终 argmax 更接近。
FP32 也不是逐位相同，只是本轮最大 logit 差小于 3e-4、相对 RMS 误差最大约 1.71e-6，所有这些位置的 top-1 均一致。

## 3. 排除状态或 ROSA 地址错误

- 同精度、同 batch 形状重复整段计算，三组各输入的 logits 最大差都为 0。
- 原版 / 新版 ROSA 的整段与分段地址、匹配长度、valid/special mask 逐位置完全一致。
- FP32 下，多种分段方式均通过；检查了最终 WKV cache、time-shift cache 和 tokens_seen。
- 不加 ROSA 的 baseline 也出现 BF16 top-1 翻转，不能归因于 ROSA 检索或门控。
- baseline 第 0 层 receptance 投影在 BF16 整段/分段间最大差为 0.001953125，
  在同一处 FP32 对照约为 4.77e-7。差异在记忆调用之外已出现，随后沿 32 层递归与残差堆叠放大。

这支持当前残余差异主要来自低精度投影与矩阵计算形状变化的数值路径，而不是丢失 ROSA 状态。
结论仅覆盖这些样本，不是对所有输入/全部实现的形式证明。部分翻转位置的参考 top-2 margin 大于 0.3，不能全部当成完全并列。

## 4. 代码与默认设置

- `rwkv_rosa/precision.py`：递归 FP32 边界；legacy 仅作为数值审计的反事实开关，不是训练默认。
- `rwkv_rosa/numerics.py`：可复现审计，记录 KL、NLL 差、argmax 翻转、cache 误差、逐层差异与实际算子 dtype。
- `verify_online()`：从最多 8 扩大为最多 64 token；单独记录 BF16 翻转，出现时打印警告；不把 FP32 检查通过写成 BF16 等价。
- `configs/rwkv_rosa/rwkv7_2_9b_sft.json`：正式对照默认改为 **float32**。
- `configs/rwkv_rosa/rwkv7_2_9b_fp32_smoke.json`：新增 FP32 正确性冒烟。
- 原 BF16 冒烟保留，并使用独立输出目录，显式标注 BF16 不能保证生成等价。
- manifest 记录 `wkv_precision_policy=fp32_autocast_disabled`；代码 hash 改变，修复前 checkpoint 不能静默作为修复后同一对照继续。

## 5. 修复后训练冒烟

指定 2.9B 上，三组分别重新完成 BF16 和 FP32 的单步前向/反向/更新/验证（仍非全参数训练，非质量 benchmark）：

| 模式 | baseline step 峰值 GiB | original step 峰值 GiB | improved step 峰值 GiB |
| --- | ---: | ---: | ---: |
| FP32 WKV + BF16 投影 | 9.70 | 9.69 | 10.57 |
| 全 FP32 | 14.86 | 14.86 | 15.74 |

本机约 48 GiB 显存可以容纳这次 FP32 partial-finetune 冒烟。以上是训练 step 阶段，不是所有验证步骤的全程峰值。
尚未验证真实业务数据、完整 SFT、2.56 万 token 长上下文，不能声称所有场景 FP32 都绝不翻转。

全仓回归：**168 项通过，Python 总行覆盖率 95.93%（6296 / 6563）**，包括 CPU/CUDA AMP 实际算子 dtype、递归输出/缓存及 CPU 梯度回归。

## 6. 复现与原始产物

```bash
conda activate rosa-rwkv
cd /mnt/data/Codes/ROSA
python -m rwkv_rosa.numerics \
  --checkpoint /hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth \
  --out-dir /mnt/data/Codes/ROSA/outputs/rwkv_bf16_audit_rerun
```

- 扩大审计：`outputs/rwkv7_g1k_2_9b_bf16_audit/precision_report.json`。
- 修复后 BF16 训练：`outputs/rwkv7_g1k_2_9b_rosa_bf16_guarded_smoke/comparison.json`。
- 修复后 FP32 训练：`outputs/rwkv7_g1k_2_9b_rosa_fp32_stability_smoke/comparison.json`。
- 输出目录不覆盖既有文件；复现时请换新目录。
