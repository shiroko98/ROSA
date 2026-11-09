# ROSA 输出投影后归一化、匹配本体 RMS（2026-11-09）

## 实现和开启

新增独立模式 `--rosa_output_norm rms_match`。保留旧的 `none` 和只做输入值
归一化的 `rms`，默认值不变，不静默改变生产任务和旧 checkpoint。

```bash
--rosa_position post_norm --rosa_output_norm rms_match \
--rosa_scale_mode logit_gap --rosa_dynamic_max 0.2 \
--rosa_dynamic_margin 0.5 --rosa_dynamic_min_match 4 --rosa_dynamic_chunk 128
```

本次没有改生产启动脚本、重启/切换长训练、重写旧 checkpoint 或原始数据。

实际公式（逐 token，epsilon=1e-6）：

```
u = W_value RMSNorm(value)
rho = stopgrad(sqrt(mean(z**2)))
u_aligned = rho * u * rsqrt(mean(u**2) + epsilon)
a = detached_logit_gap_controller(z, u_aligned, candidate)
delta = valid * a * learned_gate * u_aligned
head_input = FP32(z) + delta
logits = FP32_head(head_input)
```

投影后归一化发生在 gate 和 gain **之前**。不归一化 gate/gain 后的残差，不归一化
`z+delta`，不额外乘固定0.05。方向改善量使用对齐后的 u_aligned 计算。
因此 `RMS(delta)/RMS(z)` 约等于最终增益 `a*gate`（epsilon只会轻微缩小该比例）。
零投影/零hidden保持有限；投影主要学方向，其整体幅度不再作为独立增益。

FP32输出 head、CPU导出/流式推理和GPU独立评测均识别此模式。
checkpoint 的 output_norm 严格区分 rms/rms_match，另加 `_output_alignment_version`
buffer 防止 weights-only 元数据遗漏时静默错载。跨公式不允许原样resume。

监控新增 `rosa_residual_relative_rms_mean/max`。另修正动态分支首batch合法零增益时
的审计：不会把“值表没有更新”误报为失败或谎报已更新；后续观察到变化才记为true。
fixed旧模式仍保留首更新的严格检查。

## 两组真实3B GPU短跑

执行脚本及完整报告：

- `/mnt/data/Codes/ROSA/outputs/side_output_alignment_short_20261109/run_short.sh`
- `/mnt/data/Codes/ROSA/outputs/side_output_alignment_short_20261109/report.json`

使用 pinned upstream new_sft 的原生 Trainer、MyDataset 和 CUDA fused masked CE；
原始2.9B冻结、ROSA CUDA FP32，head参数存储不变/仅计算FP32。
数据来自用户192K数据，但**只取64个1024-token监督尾部窗口**，保留原mask；
56 train/8 held-out，固定评估4个窗口/4096监督token。不是完整192K训练。

两组各20更新，batch1，LR3e-5，warmup4，gain上限.5、margin.5、min_match4。
训练batch哈希、初始值表哈希、学习率序列、base最终哈希逐项一致。
均正常结束、无OOM，本体无梯度且字节哈希不变，ROSA值表确实更新。

| 指标 | 仅值归一化 rms + 动态gain | 输出对齐 rms_match + 动态gain |
|---|---:|---:|
| step0 eval loss | 0.961749 | 0.968443 |
| step10 eval loss | 0.961735 | 0.968251 |
| step20 eval loss | 0.961709 | 0.967814 |
| 梯度范数中位数 | 0.09658 | 0.54853 |
| 普通更新耗时中位数，排除预热/评估步 | 0.10803s | 0.10833s |
| 峰值allocated | 9.17974GiB | 9.18951GiB |
| 平均残差RMS | 0.06718 | 0.38453 |

输出对齐版相对残差RMS均值约1.331%、最大约49.875%，与有效动态增益的记录吻合。
这些统计包括有效寻址但增益为0的位置，不是“每个位置均注入相同比例”。

**结论：数值/梯度/冻结/训练路径跑通，输出修正更强，但20步内验证loss仍略高于旧
动态版，没有证明质量提升。** 两组各自下降量不能脱离初始值直接作为优劣结论。
控制器未封顶时会反向补偿方向幅度变化；封顶位置及梯度几何变化才会产生差异。
匹配错误、门控未校准和输入ChatML兼容性问题没有因此自动解决。

短跑没有上传W&B，没有保存训练checkpoint，不包含NIAH/自由生成结果。
日志、数据manifest、运行脚本和测试记录保存在上述独立输出目录，不影响生产输出。

最终回归：123项，117通过、6跳过、0失败；新增输出对齐专项5项全部通过。
真实GPU短跑还验证了FP32原生fused CE的前向/反向与ROSA优化器更新，非仅CPU模拟。
单测日志在独立短跑目录的unit_tests.log；数据与执行源码哈希一并保留。

当前新实验默认动态上限已改为0.2；上文历史短跑仍为0.5，不改写历史结果。
已有checkpoint显式保存的上限不变；改变上限不能冒充完全一致的续训。
