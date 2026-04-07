# ROSA 改良规划

## 总目标

围绕 `ROSA_PLE_Engram_分析与改良方案_v2_工程版.docx` 与 `ROSA_统一路线图_TODO合并版.docx`，按 P0 -> P1 的顺序推进 ROSA 改良，优先完成在线化基础设施，再逐步接入 PLE 风格的 value store 与 Engram 风格的门控/调度。

## 当前迭代

- 分支：`codex/online-rosa-p0-foundation`
- 迭代主题：P0 基础设施
- 当前状态：已完成 P1-5 层位扫描基础能力，下一步进入热点缓存
- 对应路线图任务：
  - 把 ROSA 从离线/整段检索改成增量在线状态机
  - 抽象地址生成接口，解耦“匹配”和“取值”
  - 搭建最小在线注入闭环

## 本轮步骤

1. 建立本地规划、进度和操作文档。
2. 为在线 ROSA 定义统一地址结构 `AddressMeta`。
3. 实现可增量调用的 `OnlineRosaState`，支持 `reset()` / `prefill()` / `update_one()` / `snapshot()`。
4. 让在线地址结果与现有 reference 语义逐 token 对齐。
5. 补测试并完成一次自我验证。
6. 更新 `progress.md` 后提交 commit。

## 本轮结果

- 已新增 `AddressMeta` / `RosaStateSnapshot` / `OnlineRosaState`。
- 已新增 `OnlineRosaBatchState` 与 `build_online_rosa_batch_state()`。
- 已新增 `rosa_addressing_with_memory()`，把地址生成与 value 消费初步解耦。
- 已保留独立 `naive` reference 路径，并单独提供在线 state 路径用于逐 token 对齐验证。
- 已在 `RosaFusedLM` 中新增 `init_online_state()` / `forward_online()`，可在 decode 场景直接消费在线状态。
- 最小在线闭环仍沿用共享 `embed_tokens` 作为 value，并使用单层早期注入配置完成验证。
- 已补 `tests/test_rosa_online_state.py`，覆盖逐 token 对齐、special 过滤、snapshot/reset。
- 已补在线前向级测试，覆盖 prefill 后 decode、一段 prompt 从空状态在线前向、跨步状态推进。
- 已新增 `profile_rosa_online_baseline.py`，输出 prefill / decode microbenchmark / 地址一致性 / 覆盖率基线报告。
- 已新增 VS Code launch 配置，可直接跑 smoke 基线与 Qwen checkpoint 对比实验。
- 已在 `RosaFusedLM` 中新增 `RosaValueStore`，支持 `shared` / `per_layer` 两种 value 模式。
- `per_layer` 模式下每个注入层都有独立 value table，初始化时从共享词嵌入复制权重，便于平滑起步。
- 已在 `RosaFusedLM` 中新增可选 `context-aware gate`，使用当前 hidden state 与 memory value 的相互作用决定注入强度。
- gate 统计指标已接入模型输出：`rosa_avg_gate` / `rosa_gate_coverage` / `rosa_gate_hit`。
- 已将运行时数据结构拆到 `rosa_runtime.py`，并新增 `RosaAddressBatch` / `RosaInjectionPayload`。
- 已将 `RosaFusedLM` 拆成显式三段：`compute_rosa_address_batch()` -> `build_rosa_injection_payload()` -> `forward_hidden(..., rosa_payload=...)`。
- 已新增 `RosaPrefetcher` / `RosaStagingBuffer`，支持 payload 预取、staging 与统计。
- profiling 已支持 `--rosa_prefetch`，输出 `hit_rate` / `wait_ms` / `sync_fallbacks` 等预取指标。
- 已支持显式层位集合 `--rosa_inject_layer_ids`，并新增 `scan_rosa_injection_layers.py` 做 single/pair 扫描。
- 小型扫描实验已跑通，当前 toy smoke 上 `layer 0` 优于 `layer 1`。
- 已在 `model` 环境执行 `python -m unittest discover -s tests`，当前通过。

## 下一任务

1. 完成 P1-6：热点地址缓存与统计。
2. 为 cache 补 hit rate / tail latency 相关 smoke。
3. 收尾更新剩余 P1 进度与推荐实验入口。

## 自我验证清单

- 单元测试覆盖 `prefill`、`update_one`、`reset`、special target 过滤、逐 token 对齐。
- 在线状态输出与当前 reference 检索结果保持一致。
- 现有 ROSA / 数据加载 / dataset 测试不回归。
- 提交前检查 `git diff`，确认文档、代码、测试都已同步。

## 下一阶段预留

- 最小在线注入闭环：单层注入，先继续复用共享 `embed_tokens`。
- 性能基线：补 decode/prefill profiling 与 match coverage 指标。
- P1：per-layer value store + context-aware gate。
