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
- [x] P1-2: 引入 context-aware gate
- [x] P1-3: 把在线调度改成“地址先算、值后取”
- [x] P1-4: 加入异步预取与 staging buffer
- [x] P1-5: 层位扫描与插入策略搜索
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
- `RosaFusedLM` 已支持 `--rosa_context_gate`。
- gate 统计指标已接入输出：
  - `rosa_avg_gate`
  - `rosa_gate_coverage`
  - `rosa_gate_hit`
- 新增 `rosa_runtime.py`，承载运行时 payload 结构。
- `RosaFusedLM` 已支持：
  - `compute_rosa_address_batch()`
  - `build_rosa_injection_payload()`
  - `prepare_rosa_injection_payload()`
  - `forward(..., rosa_payload=...)`
- 新增：
  - `RosaPrefetcher`
  - `RosaStagingBuffer`
  - `init_prefetcher()`
  - `schedule_rosa_prefetch()`
  - `consume_rosa_prefetch()`
- 新增 `--rosa_inject_layer_ids`
- 新增 `scan_rosa_injection_layers.py`
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
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_value_mode per_layer --rosa_context_gate`（P1 组合 smoke）
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_prefetch`（P1-4 smoke）
- `conda run -n model python scan_rosa_injection_layers.py ... --scan_mode single --layer_candidates \"0,1\"`（P1-5 smoke）
- 结果：共 32 个测试，全部通过；层位扫描脚本已产出 ranked report。
- 参考报告：
  - `outputs/profile_qwen_online_baseline_smoke/profile_report.json`
  - `outputs/profile_smoke_online_baseline_postpatch/profile_report.json`
  - `outputs/profile_smoke_per_layer_value/profile_report.json`
  - `outputs/profile_smoke_p1_value_gate/profile_report.json`
  - `outputs/profile_smoke_prefetch_p1/profile_report.json`
  - `outputs/scan_p1_layers_smoke/layer_scan_report.json`

## 下一步

- 可以直接运行新的 launch 做 P1 特性组合实验。
- 如果要验证训练收益，优先跑 `P1 Value+Gate Smoke (Qwen)`，再看 `comparison.json` 和 profiling 报告。
- 下一步继续做 P1-6 热点地址缓存。

## 备注

- 本轮优先保证“语义正确 + 接口稳定”，性能优化与服务化调度放到后续阶段。
