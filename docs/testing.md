# 测试与验证

## ROSA-Relay 整理提交前验证：2026-11-10

- Linux，conda `rwkv-lm-v7`，隐藏 CUDA 的全仓 pytest 回归：**272 项通过、6 项跳过**，无失败；另有 74 个 subtest 通过。总耗时约 67 秒。
- 命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python -m coverage run -m pytest tests -q`。
- pytest 会运行 unittest 用例，也会运行新增函数式 DP / 数据重处理测试；仅使用 `unittest discover` 不会发现这些函数式测试。开发依赖已包含 pytest。
- 包括真实双进程 Lightning / Gloo 稀疏行同步、加权全局 batch oracle、梯度累积和 bitwise 续训；普通 RWKV 文本格式、编译 tokenizer、mask / EOS / 打包 / 重建恢复；CPU 编译 WKV 的参考数值和递归状态；长短答案评分及采样规则。
- 全仓回归发现的旧 callback 测试替身缺少 `is_global_zero` 字段，已修复并纳入上述复跑。CUDA 专项在本轮 CPU-only 回归中跳过；既有八卡训练实测记录另见 DP 文档。
- Python 行覆盖率 **85.45%**，**未达到现有 95% 门槛**；coverage report 因门槛退出码为 2。未降低阈值，也未把新增训练或评测代码排除；远程 CI 的覆盖率检查预计仍会失败。
- 三个绘图脚本从已完成实验的 summary 在临时目录重新生成 PNG 成功。报告脚本的源码快照清单均指向仓库内存在的正式文件，不再依赖忽略的 `outputs/` 下临时脚本。
- 改动 Python 的 AST、Bash 的 `bash -n`、`git diff --check` 均通过；43 个 vendored 原生源文件哈希全部匹配，`train.py` 对上游的补丁已同步更新。
- 本次只整理代码、文档并运行回归，训练保持停止；数据、模型、实验日志和编译产物不进入 Git。最终配置、检查点和数据归档见 [ROSA-Relay 状态](rosa_relay_status.md)。

## RWKV 原生集成提交前验证：2026-11-09

- Linux，conda `rosa-rwkv`，隐藏 CUDA 执行全仓回归：262 项执行，256 项通过、6 项跳过，无失败。
- 命令：`CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests`。
- 同环境使用 coverage 重跑：262 项执行、6 项跳过、无测试失败；Python 行覆盖率 **91.60%（8301/9062）**，**未达到现有 95% 门槛**。本次不降低阈值，也不把新增自写 GPU 模块排除出统计；缺口主要在 GPU 评测及其调度/加载路径。远程 CI 仍需实际执行，当前覆盖率门槛预期会失败。
- 所有 RWKV Bash 启动脚本通过 `bash -n`；`git diff --check` 通过。
- vendored native 上游来源清单校验：43 个文件的本地 SHA256 全部匹配。许可证及完整修改补丁保留。
- 待提交内容经过大文件、二进制与硬编码凭据模式检查；模型、数据、编译产物及实验输出不进入 Git。
- GPU stateful WKV 专项此前在实际 CUDA 环境完成 3 项验证；本轮 CPU-only 回归的 skip 不等于再次跑过 GPU。
- 动态 scale、输出 RMS 对齐均为可选新公式，不静默改变历史 checkpoint；短跑质量边界见对应实验文档。

以下测试数量和覆盖率为对应历史版本的记录，不代表当前全部新增代码的覆盖率。

## BF16 专项复查：2026-11-09

全仓 **189 项通过**；另强制 CPU-only 回归 **189 项执行、3 项 GPU 测试跳过**，Python 总行覆盖率 **95.12%（7150/7517）**。
发现并修复 WKV 在外层 autocast 中被降精度的问题；新增 CPU/CUDA 实际算子 dtype、递归输出、缓存、梯度及多切分审计测试。
指定 2.9B 权重完成扩大数值审计，以及修复后的 BF16 / FP32 各三组单步 GPU 训练。BF16 仍有 top-1 翻转，正式对照默认 FP32。见 [专项记录](rwkv_bf16_audit.md)。
已完成指定 0.1B 权重的原生 SFT / 融合 CUDA 训练：1K、8K、50K 各 CPU/GPU ROSA 两组，每组 10 次参数更新；完整 epoch 检查点成功从 step 10 恢复并更新到 step 12。原生实现、精度边界与测速结果见 [原生 SFT 指南](rwkv_native_sft.md)。原参考版结论不能替代原生性能结果。

新增 binidx 解包/去重、冻结本体 + CPU ROSA 跨设备梯度/恢复，以及长度探测的 OOM/超时区分回归。真实 2.9B 已完成 CPU ROSA 两步参数更新；实际长度上限探测仍以输出目录的运行状态为准，不能从测试通过推断长序列已通过。详见 [CPU ROSA 训练说明](rwkv_cpu_rosa_training.md)。

以下 RWKV 初次记录和 Windows 整理记录为历史结果。

## RWKV 对照分支验证：2026-11-09

独立 Linux conda 环境中，全仓 **158 项通过，Python 总行覆盖率 96.13%**。
真实加载用户指定 RWKV-7 2.9B，完成 baseline / original / improved 三组单步 GPU 训练及 FP32 在线状态检查；BF16 数值误差独立记录。
另外三组完成离线 toy 全 epoch。范围、参数量、显存和结果见 [RWKV 验证记录](rwkv_rosa_validation.md)。
以下为此前仓库整理阶段的历史记录。

## 本轮验证：2026-11-09

- 平台：Windows；Python 3.10.15，PyTorch 2.5.1+cu121，NumPy 1.23.5，coverage.py 7.14.1。
- `python -m coverage run -m unittest discover -s tests`：134 项测试全部通过。
- `python -m coverage report`：Python 总行覆盖率 **95.96%**（5363/5589 行）。
- README 中的离线 Transformer + ROSA 快速示例已按原命令实跑，成功写出对比结果和两组模型权重。
- 服务器正式脚本与兼容入口通过 Bash 语法检查；配置测试验证默认无私人代理、环境变量覆盖以及本地配置加载。
- Markdown 本地链接、Python 语法、VS Code JSON、CI YAML、模型/数据忽略规则与待上传文件大小已检查。

## 运行方式

在仓库根目录执行：

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests
python -m coverage run -m unittest discover -s tests
python -m coverage report
```

`.coveragerc` 设置全仓 Python 总行覆盖率门槛 95%。统计包含当前与早期 Python 训练入口及运行模块，排除测试本身、忽略的本地构建/产物目录和原生扩展的 setuptools 构建配置。原生 C++ 行覆盖率不属于该数字；C++ 实现可用时会与 Python 地址结果做一致性回归。

## 新增回归范围

- 纯 ROSA 的 online / memory 评测，RWKV tokenizer 和 bigram baseline。
- 早期与当前 Transformer 命令行入口，baseline / ROSA 权重写出。
- epoch checkpoint 恢复、优化器状态恢复、checkpoint 保留与原子写入失败。
- reference、在线地址、缓存、snapshot、global memory、稀疏分片配方。
- memmap 从单数据源及显式切分构建，再用于真实小模型训练。
- profiling、注入层扫描、snapshot interval sweep。
- tokenizer 替身、输入格式与异常、缓存重置、可选原生扩展构建失败。
- 分布式包装和状态保存/恢复接口、metric reduction；通信边界使用替身。
- W&B 缺失、关闭、可选元信息失败；测试不连接外部 W&B 服务。

## 环境边界

本地通过不等于已完成多卡训练性能验证。新增 GitHub Actions 会在 push/PR 上执行 Ubuntu 与 Windows 测试，但本轮尚未 push，远程 CI 尚未运行。真实多卡通信、CUDA 吞吐和 FSDP 大模型恢复仍需在相应设备上验证。
