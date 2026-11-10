# ROSA：在线检索记忆与语言模型融合

本仓库实现纯 ROSA 非参数预测，以及将 ROSA 检索结果注入 Qwen/LLaMA 风格 Transformer 和 RWKV-7 的训练与评测流程。重点包括在线后缀自动机（SAM）、可训练记忆融合和训练性能优化。

**原有实验是 Transformer vs Transformer + ROSA**，使用自建主干从头训练；Qwen tokenizer 不代表 Qwen 预训练权重。**本分支新增真实 RWKV-7 的 baseline / 原版 ROSA 最小融合 / 新版 ROSA 对照训练**，支持严格加载 RWKV `.pth`；详见 [RWKV 实验指南](docs/rwkv_rosa_comparison.md)。`train_pure_rosa_rwkv_baseline.py` 仍只是 tokenizer + 纯 ROSA/bigram 评测，不是新 RWKV 神经模型入口。

## 功能

- 在线地址生成：统一 sequence / prefill / decode 状态，保留 reference 路径用于一致性回归。
- 记忆融合：共享 embedding（V1）、每层独立 ValueStore（V2）、上下文与匹配长度门控。
- 性能优化：fast SAM、可选 C++ CPU 扩展、异步地址预取、snapshot + replay、value hot cache。
- 数据与训练：JSON/JSONL、预分词 memmap、稀疏 value 更新、本地行分片、梯度累积、DDP/FSDP、W&B。

## 安装

建议 Python 3.10 或更新版本。在虚拟环境中安装：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

快速示例使用内置 byte tokenizer，不下载模型或数据，可在 CPU 运行；训练入口检测到 CUDA 时会自动使用 GPU。正式实验使用 Hugging Face tokenizer 时，再安装 `requirements-hf.txt`。W&B 是可选依赖：`python -m pip install wandb`。

GPU 环境请先安装与设备匹配的 PyTorch，再安装其余依赖。服务器安装与多卡入口见 [运行指南](docs/usage.md) 和 [服务器配置示例](env/server.local.sh.example)。C++ CPU 扩展需要本机编译工具，默认示例不依赖它。

## RWKV-7 原生 SFT 训练（当前推荐入口）

按用户指定的 `TimeMobius/RWKV-LM-V7/new_sft` 原生 `train.py`、Lightning Trainer、SFT loader/loss 和融合 CUDA 内核训练；默认 base 已改为用户的 `rwkv7-g1d-0.1b-20260129-ctx8192.pth`。冻结本体，只更新新版 ROSA，支持 `--rosa_device cuda` / `cpu`。真实 1K/8K/50K 各两组、每组 10 步已完成，GPU ROSA 在该硬件上快约 1.33× / 1.51× / 1.90×；SAM 寻址两组均在 CPU，非质量优劣结论。完整配置、结果、W&B 状态及命令见 [原生 SFT 指南](docs/rwkv_native_sft.md)。

旧 `train_rwkv_rosa_compare.py` 与后续参考版长度探测仍保留作正确性/历史实验，不作为原生内核性能依据。

### 当前 ROSA 扩展与验证边界

- 明确区分 block 注入与 `post_norm`（最终 norm 后、head 前），checkpoint 保存并校验位置语义。见 [post-norm 说明](docs/rwkv_post_norm_training.md)。
- 支持独立 GPU 大海捞针评测、RWKV state 和编译 SAM 增量更新；10 条三格式/模型配置快速对比已完成，未观察到旧 step 200 ROSA 相对 base 的提升。见 [stateful 实现与实测](docs/rwkv_gpu_stateful_needle.md)。
- 新增可选的输出归一化、投影后 RMS 对齐及有界 `logit_gap` 动态 scale；默认旧公式不变。这些功能已经过数值和短跑验证，**不代表已证明质量提升**。见 [动态 scale](docs/rwkv_dynamic_scale.md)、[输出对齐](docs/rwkv_output_alignment.md)。
- 原生路径支持八卡复制式 DP：密集梯度走 DDP，稀疏 ValueStore 梯度按全局触达行集合平均；不使用 ZeRO。每两小时保存，独立 CPU 评测不阻塞训练。见 [192K DP 训练](docs/rwkv_dp_192k_training.md)。

### ROSA-Relay：当前实现及实验归档

当前 2.9B 本体冻结，ROSA 在最终 norm 后、LM head 前注入，训练上下文为 192512。最终评测使用 step 1800、`logit_gap`、`min_match=6`、`dynamic_max=0.3`；训练已按要求停止。已完成 8K–256K 的短答案和长答案检索，以及 128K 的 GPU / CPU 混合 / 纯 CPU 推理速度对比。数据处理、代码入口、指标口径、结果和检查点位置集中在 [ROSA-Relay 状态与复现说明](docs/rosa_relay_status.md)。与 ROSA-Tuning 论文的区别见 [论文对照](docs/rosa_tuning_paper_comparison_2602_02499.md)。

## RWKV-7 对照实验（新增）

```bash
bash scripts/setup_rwkv_rosa_env.sh
bash scripts/run_rwkv_rosa_compare.sh configs/rwkv_rosa/toy.json
```

指定 RWKV-7 2.9B 权重的 GPU FP32 正确性冒烟配置：

```bash
bash scripts/run_rwkv_rosa_compare.sh configs/rwkv_rosa/rwkv7_2_9b_fp32_smoke.json
```

**精度边界：**参考运行时的 BF16 扩大审计观察到分段 top-1 翻转，已修复一处 WKV autocast 边界问题，但整体 BF16 等价仍无保证，因此参考版准确性对照默认 FP32。原生 SFT 则使用 BF16 本体和 FP32 ROSA；增量解码与整段前向的数值差异单独记录，不承诺逐位一致。见 [数值复查](docs/rwkv_bf16_audit.md)。

**冻结本体 + CPU ROSA：**新增 `--freeze-backbone --rosa-device cpu --variants improved`，仅训练 CPU 上的新版 ROSA；支持读取打包 SFT binidx，并提供完整反向传播的实测长度探测入口 `python -m rwkv_rosa.length_probe`。设备放置、真实预检结果及长上下文限制见 [CPU ROSA 训练说明](docs/rwkv_cpu_rosa_training.md)。

正式数据、相同历史范围、原版/新版定义、训练范围与恢复方法见 [RWKV 实验指南](docs/rwkv_rosa_comparison.md)。上游参考递归是可微 PyTorch 实现，并非其融合 CUDA 内核的性能替代。

## 离线快速验证

在仓库根目录执行以下命令（写成一行以兼容 Bash 和 PowerShell）：

```bash
python train_qwen_llama_vs_rosa_v2.py --data_path examples/toy_corpus.txt --split_mode line --arch_style qwen --seq_len 16 --stride 16 --dim 16 --n_layers 1 --n_heads 2 --n_kv_heads 2 --intermediate_size 32 --batch_size 4 --epochs 1 --rosa_recipe online_v1 --out_dir outputs/quickstart
```

命令会分别训练 baseline 与 ROSA，生成 `comparison.json`、`baseline.pt`、`rosa_fused.pt`。这是验证流程的小样本示例，不用于衡量模型质量。

纯 ROSA 示例不需要 PyTorch：

```bash
python train_pure_rosa_baseline.py --toy repeat --mode online --out_dir outputs/pure_rosa_demo
```

## 代码与文档导航

| 路径 | 用途 |
| --- | --- |
| `train_rwkv_rosa_compare.py`、`rwkv_rosa/` | 新增 RWKV-7 + 原版/新版 ROSA 对照训练 |
| `third_party/rwkv_lm_v7/` | 固定版本的上游 RWKV 参考主干、SFT 模板和 mask；保留许可证 |
| `train_qwen_llama_vs_rosa_v2.py` | 原有 Transformer 主训练入口 |
| `train_qwen_llama_vs_rosa.py` | 保留的早期 Transformer 对照入口 |
| `train_pure_rosa*_baseline.py` | 纯 ROSA 与 byte/RWKV tokenizer 实验 |
| `rosa_addressing.py`、`rosa_session.py` | 在线地址和状态生命周期 |
| `rosa_runtime.py`、`rosa_value_store.py` | 检索 payload、缓存与可训练 value |
| `rosa_memmap_*.py`、`build_rosa_memmap_dataset.py` | 预分词数据构建与加载 |
| `profile_rosa_online_baseline.py`、`scan_rosa_injection_layers.py` | 性能、一致性和注入层实验 |
| `scripts/server/` | 正式服务器脚本；`scripts/` 同名入口兼容旧命令 |
| `cpp_extensions/` | 可选 C++ SAM 地址实现 |
| `tests/` | 单元与回归测试 |
| `examples/` | 可公开、无需下载的示例语料 |
| [docs/usage.md](docs/usage.md) | 详细参数、训练与性能操作记录 |
| [docs/experiments.md](docs/experiments.md) | 已有实验结果与比较边界 |
| [docs/final_todolist.md](docs/final_todolist.md) | 合并路线图 |
| [docs/progress.md](docs/progress.md)、[docs/planning.md](docs/planning.md) | 历史进展和设计笔记 |
| [todolist.txt](todolist.txt) | 当前整理任务与验证状态 |

为兼容现有命令和导入，Python 模块继续保留在根目录。

## 服务器配置

代理和个人模型路径没有默认值。可直接导出环境变量，或复制示例后编辑本地配置：

```bash
cp env/server.local.sh.example env/server.local.sh
# 编辑 env/server.local.sh 中的数据、tokenizer 与输出路径
bash scripts/server/launch_8gpu_ddp_online_v1.sh
```

`env/server.local.sh` 被 Git 忽略。外部环境变量优先于示例中的默认设置。服务器默认模型规模较大，先检查显存和超参数，再启动多卡训练。详见 [服务器配置](docs/usage.md)。

## 测试与覆盖率

最近一次提交前 Linux CPU-only 全仓 pytest 回归：**272 项通过、6 项跳过**，无失败（另有 74 个 subtest 通过），包括真实双进程 DP 同步和续训测试。当前 Python 行覆盖率 **85.45%，尚未达到 CI 的 95% 门槛**；未降低阈值或排除新增自写 GPU 模块。历史记录及当前验证范围见 [验证记录](docs/testing.md)。GitHub Actions 已配置 Ubuntu / Windows 回归，push 后运行；本地通过不代表远程 CI 已完成。

```bash
python -m unittest discover -s tests
python -m pytest tests -q
python -m coverage run -m unittest discover -s tests
python -m coverage report
```

测试使用临时目录、离线 byte tokenizer 或 tokenizer 替身。覆盖率统计自写 Python 模块及命令行入口，固定来源的 `third_party/` 不计入；原生 C++ 和 `build_rosa_sam_cpu_extension.py` 构建配置不计入 Python 行覆盖率。真实多卡通信、CUDA 性能和编译器支持需要在对应环境另行验证。

## 数据与 checkpoint

`data/`、`outputs/`、`rosa_runs/`、模型权重、日志、缓存和本地配置均不上传 Git。训练完成的 `rosa_fused.pt` 是模型权重；`checkpoints/<model>/last.pt` 还含优化器及训练进度，用于恢复训练。**Git 提交不会备份 checkpoint，请单独保存。**

仓库保留原有 Git 历史；本文中的实验属于小规模研究验证，尚不代表成熟的大模型或生产推理服务。
