# RWKV + ROSA 初次验证记录（精度修复前历史）

> 后续扩大 BF16 检查发现了实际 top-1 翻转及 WKV autocast 边界问题。本文保留初次运行原值，不代表当前版本精度已验收。当前结论、修复、168 项测试及修复后重跑见 [BF16 专项复查](rwkv_bf16_audit.md)。

日期：2026-11-09。分支：`codex/rwkv-rosa-compare`。

## 环境

- Linux；独立 conda 环境 `rosa-rwkv`（Python 3.11.18）。
- PyTorch 2.7.1+cu126、NumPy 1.26.4、Jinja2 3.1.6。
- 单张 NVIDIA GeForce RTX 4090 D，本机报告总显存约 48 GiB。
- `pip check`、Python 编译检查、Bash 语法检查通过。

## 单元与回归测试

- `python -m coverage run -m unittest discover -s tests`：**158 项通过**。
- Python 总行覆盖率 **96.13%**（6082 / 6327 行），继续满足 95% 门槛。
- 固定来源的 `third_party/` 不计入本仓自写 Python 行覆盖率；原生 C++ 仍不计入。
- 新测试覆盖：上游 baseline 数值一致性、严格权重加载、共同初始化不别名、
  原版/新版零 scale、因果性、prefill/decode 双状态与快照恢复、分层梯度、稀疏分片、
  全局梯度裁剪、容量控制、相同历史数据与 mask、异步/同步相同更新、逐步恢复与连续训练逐参数一致。
- 修正旧 profiling 回归依赖的被忽略私有语料，改用公开示例；训练路径一致性统计不再把 PAD 位置的原始匹配元数据当作语义差异，并增加真实 token 不得被忽略的回归。

## 完整离线 toy epoch

`scripts/run_rwkv_rosa_compare.sh configs/rwkv_rosa/toy.json --out-dir outputs/rwkv_rosa_toy_verified`

三组均完成 1 epoch、16 个 optimizer step，保存可恢复的 trainable delta。
用途仅为检查 CLI、SFT、分层稀疏值表、完整训练循环和 checkpoint，不评价真实模型质量。

## 用户指定 2.9B 权重的真实 GPU 冒烟

基础权重：`/hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth`。

SHA256：`d8f8ecd4af8d706f867ac8a74284f35c0e3b9f863e7b09bb4d4031480c8b28b2`。

严格加载 32 层、hidden=2560、head size=64 的真实 RWKV-7 权重，无随机主干替代。
使用上游可微 PyTorch recurrence；BF16 存储冻结权重、FP32 可训练参数，BF16 autocast，
关闭 BF16 低精度部分归约。三组共同训练最后一个 block、输出 LayerNorm 和 head。

6 条合成 SFT 记录，train/val/test = 4/1/1；各组只更新 1 step，测试仅 6 个监督 token。
默认未保存这次大模型冒烟的 optimizer/checkpoint；正式配置支持保存与恢复。

| 组别 | 总参数 | 可训练参数 | Step 秒数 | 训练 step 阶段峰值 GiB | Test loss | Test token accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2,948,065,280 | 249,418,240 | 0.353 | 9.70 | 18.28534 | 50.00% |
| original | 2,948,065,280 | 249,418,240 | 0.324 | 9.69 | 18.25930 | 50.00% |
| improved | 3,128,949,762 | 430,302,722 | 1.903 | 10.57 | 18.15277 | 50.00% |

**这些不是效果或吞吐 benchmark。**只有 1 step、极少监督 token；值表新增参数与反向范围不同，
first-step optimizer 初始化也会影响耗时。尚不能认为新版比原版好。
模板/新增特殊标记尚需真实 SFT 适配，loss 很高，不能声称业务适配已完成。

## 指定权重上的在线状态检查

每组在训练后检查 8 个真实 tokenizer 输入 token：

| 组别 | FP32 整段/分段最大 logit 差 | FP32 相对 RMS 误差 | BF16 最大 logit 差 | BF16 top-1 一致率 | 关闭 ROSA 与上游主干差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.0001011 | 9.68e-07 | 0.25000 | 100.00% | 0.0 |
| original | 0.0000553 | 6.64e-07 | 0.34375 | 100.00% | 0.0 |
| improved | 0.0000992 | 1.37e-06 | 0.31250 | 100.00% | 0.0 |

FP32 检查阈值 atol=0.002、rtol=0.0002，三组通过；不声称 BF16 逐位一致。
验证时临时将冻结参数提升 FP32，额外峰值不包含在上表的训练 step 显存中。

## 产物与未完成事项

- 原始汇总：`outputs/rwkv7_g1k_2_9b_rosa_comparison_smoke/comparison.json`。
- 配置/数据/权重来源：同目录 `manifest.json`。
- GPU 控制台日志：`outputs/rwkv7_g1k_2_9b_rosa_comparison_smoke.console.log`。
- 所有 outputs 均被 Git 忽略；本文保留可追溯的实际结果摘要。
- 尚未开展真实业务数据上的完整 SFT、多种子、2.9B capacity-matched 对照或全参数训练。
- 未做多 GPU 验证；当前新入口明确拒绝 torchrun。
- 未推送或运行远程 CI。
