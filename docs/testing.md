# 测试与验证

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
