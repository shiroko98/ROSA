# ROSA：在线检索记忆与语言模型融合

本仓库实现纯 ROSA 非参数预测，以及将 ROSA 检索结果注入 Qwen/LLaMA 风格 Transformer 的训练与评测流程。重点包括在线后缀自动机（SAM）、可训练记忆融合和训练性能优化。

**当前对照实验是 Transformer vs Transformer + ROSA，不是 RWKV vs RWKV + ROSA。**模型在本仓库中从头训练；使用 Qwen tokenizer 不代表加载了 Qwen 预训练模型。`train_pure_rosa_rwkv_baseline.py` 使用 RWKV tokenizer，其 baseline 是 bigram，不是 RWKV 神经网络。

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
| `train_qwen_llama_vs_rosa_v2.py` | 当前主训练入口 |
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

最近一次本地验证：134 项测试通过，Python 总行覆盖率 95.96%。范围与环境边界见 [验证记录](docs/testing.md)。GitHub Actions 已配置 Ubuntu / Windows 回归，push 后运行。

```bash
python -m unittest discover -s tests
python -m coverage run -m unittest discover -s tests
python -m coverage report
```

测试使用临时目录、离线 byte tokenizer 或 tokenizer 替身。覆盖率统计 Python 模块及命令行入口；原生 C++ 和 `build_rosa_sam_cpu_extension.py` 构建配置不计入 Python 行覆盖率。真实多卡通信、CUDA 性能和编译器支持需要在对应环境另行验证。

## 数据与 checkpoint

`data/`、`outputs/`、`rosa_runs/`、模型权重、日志、缓存和本地配置均不上传 Git。训练完成的 `rosa_fused.pt` 是模型权重；`checkpoints/<model>/last.pt` 还含优化器及训练进度，用于恢复训练。**Git 提交不会备份 checkpoint，请单独保存。**

仓库保留原有 Git 历史；本文中的实验属于小规模研究验证，尚不代表成熟的大模型或生产推理服务。
