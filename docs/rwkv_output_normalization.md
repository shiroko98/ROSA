# post_norm ROSA 输出归一化修复（2026-11-09）

## 开启方式及兼容边界

在现有 native SFT 启动命令后显式追加：

```bash
--rosa_position post_norm --rosa_output_norm rms --rosa_scale 0.05
```

请使用**新的输出目录和 W&B run**。本次没有修改生产启动脚本、数据或运行中的训练，
没有启动新的正式训练。默认 `--rosa_output_norm none` 严格保留旧行为，避免主线程的
自动续训/评测在不知情时更换公式。block 注入不允许开启这个选项。

`rms` 同时开启以下两项，不能只归一化而继续用 BF16 输出加法：

1. 在值表 lookup 后、输出投影前，对输出值独立做**无仿射参数的 RMS 归一化**：
   `v_out = v * rsqrt(mean(v²) + 1e-6)`。
   门控的 key normalization 和匹配长度公式不变；不归一化整个本体 hidden，
   也不在输出投影之后再次归一化，保留投影调整方向及幅度的能力。
2. `delta = scale * gate * W_value(v_out)`，以 FP32 计算 `z.float() + delta`，
   输出 head GEMM 和分块交叉熵也走 FP32；RWKV blocks 仍是原生 BF16。

`scale=.05` 保留为固定输出增益，不是学习率，也不表示本体输出的5%。
归一化后它控制的是 RMS 标定后的值分支强度。没有新增可学习 gain，也没有修改
门控初始化；进一步的增益/门控调参应作为独立消融，不能与修复混为一谈。

## 冻结、显存与 checkpoint

- 本体与 head 参数仍冻结。head 仅在计算时转成 FP32 view/copy，存储仍是原来的
  BF16 参数，因此原始 frozen-base 的字节哈希、外部 base 还原检查不变。
- 不分配完整 `192K × 65536` logits，继续使用原生分块 CE。FP32 输出会增加
  hidden/head 临时张量与计算开销；**尚未做修复版 192K CUDA 容量或速度测试**。
- checkpoint/CPU导出/独立评测请求保存 `output_norm`；缺少此项按历史 `none`。
  resume 严格拒绝跨 `none/rms` 续训。新版权重另带 `_output_norm_version` buffer，
  即使 weights-only sidecar 漏字段，也会在严格加载时失败，而不是悄悄错用旧公式。
- CPU 推理、流式推理、CPU 独立 needle 和 GPU 独立 needle 均识别该模式。
  GPU 评测对 base/trained 两组使用相同的 FP32 head，避免把 head 精度差异计为收益。
  生成 GPU weights-only sidecar 时，必须从训练 checkpoint 复制 `output_norm`。

## 验证与未解决问题

新增 `tests/test_rwkv_output_normalization.py` 检查：
旧路径完全兼容、BF16 小残差不再消失、值分支归一化、投影幅度仍可学习、AMP 下
输出保持 FP32、冻结权重哈希/梯度、原生 fused CE 的 FP32 参数路径、metadata拒绝
错误续训、CPU导出/流式一致性，以及独立评测请求携带模式。

这些测试不证明训练质量提升，也不代替真实 CUDA fused CE 的数值/梯度测试。
数据的 ChatML 保留 token 与当前 base 的兼容性是**另一项已复现的问题**，本补丁
不修改 token 数据或监督 mask，不能宣称单靠归一化就修复完整192K的高loss。

本次最终验证命令（隐藏 CUDA，不抢占生产 GPU）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/mnt/data/anaconda3/envs/rosa-rwkv/bin/python -m unittest discover -s tests -p 'test_rwkv*.py' -q
```

结果：111 项，105 通过、6 跳过、0 失败；新增归一化专用测试7项全部通过。
vendored 43 个源文件的 local SHA256 全部匹配；src/model.py 对应的 upstream diff
和 local hash 同步更新，反向重建原始源码的 SHA 与 pinned upstream 一致。
