# ROSA 开发进度

## 当前状态

- 已有基础：`train_qwen_llama_vs_rosa_v2.py` 中已实现 ROSA 的 `sam` 检索版本，并有基础回归测试。
- 当前分支：`codex/online-rosa-p0-foundation`
- 当前阶段：P0 基础设施

## 本轮任务

- [x] 梳理改良文档与统一路线图
- [x] 建立本地规划/进度/操作文档
- [x] 实现 Online ROSA State
- [x] 抽象统一 AddressMeta 接口
- [x] 搭建最小在线注入闭环
- [x] 建立性能与正确性基线
- [x] P1-1: 引入 per-layer Value Store
- [x] 补逐 token 一致性测试
- [x] 完成自我验证并提交本轮 commit

## 本轮已完成

- 在 `train_qwen_llama_vs_rosa_v2.py` 中新增：
  - `AddressMeta`
  - `RosaStateSnapshot`
  - `OnlineRosaState`
  - `OnlineRosaBatchState`
  - `rosa_addressing_with_memory()`
- 现有 `naive` / `sam` 检索已复用统一地址抽象，保留旧的 tensor 返回接口。
- `naive` reference 与在线 state 路径已分离，后续可继续做独立一致性验证。
- `RosaFusedLM` 已支持 `init_online_state()` 与 `forward_online()`。
- 最小 online decode 注入闭环已跑通：prefill 后可逐步 `forward_online()`，并复用共享 embedding + match_len gate。
- 新增 `profile_rosa_online_baseline.py`，可对比：
  - baseline
  - rosa_reference
  - rosa_online
- `RosaFusedLM` 已支持 `--rosa_value_mode shared|per_layer`。
- 新增 `RosaValueStore`，`per_layer` 模式下每个注入层独立 lookup value。
- profiling 报告输出 `profile_report.json`，包含：
  - prefill 延迟 / tok/s
  - decode microbenchmark 延迟 / tok/s
  - 地址一致性
  - match / fire coverage
  - online vs reference logit diff
- 新增测试文件 `tests/test_rosa_online_state.py`。
- 新增测试文件 `tests/test_profile_rosa_online_baseline.py`。

## 自我验证记录

- `conda run -n model python -m unittest tests.test_rosa_online_state`
- `conda run -n model python -m unittest tests.test_profile_rosa_online_baseline`
- `conda run -n model python -m unittest discover -s tests`
- `conda run -n model python profile_rosa_online_baseline.py ...`（Qwen ckpt 小样本实跑）
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_value_mode per_layer`（P1-1 smoke）
- 结果：共 23 个测试，全部通过；profiling 脚本已在真实 checkpoint 上跑通。
- 参考报告：
  - `outputs/profile_qwen_online_baseline_smoke/profile_report.json`
  - `outputs/profile_smoke_online_baseline_postpatch/profile_report.json`
  - `outputs/profile_smoke_per_layer_value/profile_report.json`

## 下一步

- 继续完成 P1-2 `context-aware gate`。
- 完成后再补 P1 组合实验入口。

## 备注

- 本轮优先保证“语义正确 + 接口稳定”，性能优化与服务化调度放到后续阶段。
