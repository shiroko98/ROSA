# ROSA 开发进度

## 当前状态

- 已有基础：`train_qwen_llama_vs_rosa_v2.py` 中已实现 ROSA 的 `sam` 检索版本，并有基础回归测试。
- 当前分支：`codex/online-rosa-p0-foundation`
- 当前阶段：旧主线 P1 已收尾，准备进入在线主线重构

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
- [x] P1-6: 缓存与热点地址管理
- [x] 在线主线 Task-1: `AddressEngine.forward_seq()`
- [x] 在线主线 P0-1: 旧路径退位为 `reference/fallback`
- [x] 在线主线 P0-4: 训练前向接入 `forward_seq()`
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
- 新增 `--rosa_hot_cache_size`
- 新增 `RosaHotAddressCache`
- 新增 `RosaAddressEngine`
- 新增 `--rosa_seq_address_mode reference_backend|online_exact`
- 新增 `--rosa_train_mode online_seq|reference_precompute`
- profiling 报告输出 `profile_report.json`，包含：
  - prefill 延迟 / tok/s
  - decode microbenchmark 延迟 / tok/s
  - 地址一致性
  - match / fire coverage
  - online vs reference logit diff
- cache 相关统计现已接入：
  - `prefill_hot_cache_token_hit_rate`
  - `decode_hot_cache_token_hit_rate`
  - `rosa_hot_cache_active_entries`
  - `layer_stats.top_addresses`
- 新增测试文件 `tests/test_rosa_online_state.py`。
- 新增测试文件 `tests/test_profile_rosa_online_baseline.py`。
- 新增测试文件 `tests/test_rosa_runtime.py`。
- 训练主入口默认已切到 `online_seq`，`doc_local + sam` 不再默认产出 `rosa_precomputed_ids`。
- `DocChunkDataset` 在线主线路径现在默认提供 `full doc prefix` 左侧 memory，便于和 reference 路径对齐。
- 旧 `doc_local + sam precompute` 已通过 `--rosa_train_mode reference_precompute` 保留为显式回归/对照路径。
- `RosaFusedLM.forward_hidden()` 现已输出地址来源统计：
  - `rosa_address_source_precomputed`
  - `rosa_address_source_online_seq`
  - `rosa_address_source_reference_seq`
  - `rosa_address_source_online_step`
- 已补在线训练前向与训练 smoke，确认在不传 `rosa_precomputed_ids` 时：
  - 会实际调用 `AddressEngine.forward_seq()`
  - `loss.backward()` 可正常反传
  - `train_one_model()` 可稳定完成一轮训练

## 自我验证记录

- `conda run -n model python -m unittest tests.test_rosa_online_state`
- `conda run -n model python -m unittest tests.test_profile_rosa_online_baseline`
- `conda run -n model python -m unittest discover -s tests`
- `conda run -n model python profile_rosa_online_baseline.py ...`（Qwen ckpt 小样本实跑）
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_value_mode per_layer`（P1-1 smoke）
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_value_mode per_layer --rosa_context_gate`（P1 组合 smoke）
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_prefetch`（P1-4 smoke）
- `conda run -n model python scan_rosa_injection_layers.py ... --scan_mode single --layer_candidates \"0,1\"`（P1-5 smoke）
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_hot_cache_size 16`（P1-6 cache smoke）
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_min_match_len 1 --rosa_hot_cache_size 16`（P1-6 hit/tail smoke）
- `conda run -n model python profile_rosa_online_baseline.py ... --rosa_seq_address_mode online_exact`（在线主线 Task-1 smoke）
- `conda run -n model python -m unittest tests.test_train_qwen_llama_vs_rosa_v2`（在线主线 P0-1 入口切换）
- `conda run -n model python -m unittest tests.test_train_qwen_llama_vs_rosa_v2`（在线主线 P0-4 前向/反传 smoke）
- 结果：共 40 个测试，全部通过；P1 六项任务均已落地，在线主线 Task-1 也已完成。
- cache smoke 结论：`min_match_len=1` toy profile 上，prefill / decode hot cache token hit rate 约 `0.98 / 0.96`；端到端平均时延基本持平，说明当前收益主要体现在“减少重复 value fetch”，更适合后续 host memory / mmap 路径放大。
- AddressEngine 结论：`online_exact` 模式下，`forward_seq()` 与现有 reference 地址结果保持对齐，可作为后续切换训练主线的统一入口。
- 训练入口切换结论：当前默认训练主线已不再依赖 `rosa_precomputed_ids`；reference 预计算路径仍可通过显式 `train_mode` 单独回归。
- 在线训练前向结论：当前 teacher forcing 主训练前向在不传 precomputed 地址时，会稳定走 `forward_seq()` 并支持正常反传。
- 参考报告：
  - `outputs/profile_qwen_online_baseline_smoke/profile_report.json`
  - `outputs/profile_smoke_online_baseline_postpatch/profile_report.json`
  - `outputs/profile_smoke_per_layer_value/profile_report.json`
  - `outputs/profile_smoke_p1_value_gate/profile_report.json`
  - `outputs/profile_smoke_prefetch_p1/profile_report.json`
  - `outputs/scan_p1_layers_smoke/layer_scan_report.json`
  - `outputs/profile_p1_hot_cache_on/profile_report.json`
  - `outputs/profile_p1_hot_cache_on_m1/profile_report.json`

## 下一步

- 可以直接运行新的 launch 做 P1 特性组合实验，或单独验证热点缓存。
- 如果要验证 cache 行为，优先看 `prefill_hot_cache_token_hit_rate`、`decode_hot_cache_token_hit_rate` 和 p95。
- 已新增 `ROSA_在线主线重构_TODO.md`，旧 `doc_local/global precompute` 方案降级为 reference/fallback。
- 已在在线主线 TODO 中补充：
  - `ROSA-DocMemory`：外部检索文档作为 side memory
  - `ROSA × Engram`：文档 memory 与参数化 memory 的 hybrid 路线
- 下一步优先验证新的在线训练主线：`teacher forcing 并行主干 + AddressEngine.forward_seq() + 在线 side-branch 注入`。
- 下一步优先做在线训练地址与 `reference_precompute` 的逐位置一致性回归，并补 smoke 报告。

## 备注

- 本轮优先保证“语义正确 + 接口稳定”，性能优化与服务化调度放到后续阶段。
