# post_norm ROSA 动态 scale（实验功能，2026-11-09）

## 开启

在新的 native SFT 实验命令后追加以下参数，使用新的输出目录/W&B run：

```bash
--rosa_position post_norm --rosa_output_norm rms \
--rosa_scale_mode logit_gap --rosa_dynamic_max 0.2 \
--rosa_dynamic_margin 0.5 --rosa_dynamic_chunk 128 --rosa_dynamic_min_match 6
```

默认仍为 `--rosa_scale_mode fixed`，保持旧训练及自动续训的公式不变。
`logit_gap` 下 `rosa_scale` 不再参与增益计算；`dynamic_max` 才是上限。
本次没有切换/重启生产训练，也没有改生产启动脚本或数据。

## 公式与安全边界

- 本体 norm 后特征 z，检索候选 c。
- 未乘门控/scale 的输出方向 u = W_value RMSNorm(v)。
- 原始 base logits = W_head z（带 head bias 时也计入）。
- j 是 base 中分数最高的非 c token。
- gap = base[j] - base[c] + margin。
- improvement = (W_head[c] - W_head[j]) · u。
- improvement > 1e-6 且数值有限时：
  `a = clamp(max(gap,0) / improvement, 0, dynamic_max)`；否则 a=0。
- 最终 residual = a * learned_gate * u，再按地址 valid_mask 过滤。

原始匹配长度不足6（可配置）的地址不做动态放大；这是保守筛选，不是正确性概率。
控制器只看当前特征、检索候选、base 输出和修正方向，**不读取训练标签**。
控制器计算 no_grad/detach，避免通过分母反向学习出奇异放大；方向与 gate 仍接受
普通交叉熵梯度。候选已足够领先时增益为0；无效地址/错误方向不做动态注入。

这里只校准相对原始最强竞争者的分差，**不保证最终 top1**：其他 token 也会被
修正，而且 learned_gate 是现有的可学习门控，尚未校准为检索正确概率。
错误候选若方向“有效”仍可能被放大，因此必须同时检查正确/错误候选两组。
这不是强制 copy，也不能拿 teacher-forced 首 token 变化冒充自由生成准确率。

## 内存、计算与冻结

控制器按 token 分块做一遍额外的 frozen head 前向，并只对候选/竞争者的 head 行
计算方向点积，不额外展开一遍全词表方向 logits。
默认128行×65536词×FP32约32MiB的词表分块；不保存整个192K×词表的logits。
还有FP32 head临时副本及普通特征内存，实际峰值和速度仍需CUDA验证。

head 通过弱引用绑定，不注册进 ROSA，也不进入适配器优化器/checkpoint；
本体/head参数字节不变。输出仍走归一化修复后的 FP32 相加与分块 head/loss。

## 保存、加载和监控

完整训练checkpoint、CPU转换导出、CPU独立评测请求与GPU加载器同步记录/读取：
`scale_mode / dynamic_max / dynamic_margin / dynamic_chunk / dynamic_min_match`。
缺失字段的旧checkpoint按fixed默认值解释；跨模式或阈值变化的resume被拒绝。
新版权重带 `_dynamic_scale_version` buffer，防止sidecar遗漏字段时静默错载。

训练统计加入 `rosa_dynamic_scale_mean`、`rosa_dynamic_scale_max`（含gate后的有效值），
`rosa_dynamic_eligible_fraction`（通过匹配长度门槛的比例），
及 `rosa_dynamic_useful_fraction`（有效候选中改善方向为正的比例）。

## 验证范围

`tests/test_rwkv_dynamic_scale.py` 覆盖：分差计算/上限、非正方向和无效地址过滤、
非有限值处理、分块与整体一致、AMP禁用、梯度/冻结/无head重复注册、metadata恢复
拒绝，以及导出/CPU流式推理。另运行短3B CPU首token诊断，结果在
`outputs/side_dynamic_scale_smoke_20261109/report.json`。

短诊断是显式使用旧step200张量做新公式的内存内对照，不是旧优化器续训、不是
新动态版训练结果，也不是完整生成/NIAH benchmark；正式训练仍须先处理输入
ChatML保留token与base不兼容的问题。本补丁不修改该数据问题。

## 本次实测记录

- RWKV相关回归共118项：112通过、6跳过、0失败（隐藏CUDA运行）；动态scale专用7项通过。
- 3B CPU FP32，256-token文档，`rwkv`普通文本模板，原step200张量仅做公式对照。
- 未加最小匹配门槛时，QA错误候选只有1-token匹配，门控仍约0.899，有效scale约0.450；
  正确首token概率从base 0.269%降至0.071%。原始结果保存在report_unfiltered.json。
- 增加默认min_match=4后，该案例scale=0，保留base输出。
- 两个exact-prefix案例候选正确，base首token已是top1，概率分别91.47%/80.52%；
  动态控制器均输出scale=0，不做额外放大。
- 因此这3个案例验证了安全筛选及无需补分时的零增益行为，**没有证明动态版优于
  固定scale，也没有新增答对的首token**。未进行自由生成或192K CUDA训练。

当前新实验默认动态上限已改为0.2；上文历史短跑仍为0.5，不改写历史结果。
已有checkpoint显式保存的上限不变；改变上限不能冒充完全一致的续训。

当前新实验默认最小匹配长度已改为6 token；历史短跑仍为4，不改写旧记录。
