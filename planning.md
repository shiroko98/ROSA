# ROSA 改良规划

## 总目标

围绕 `ROSA_PLE_Engram_分析与改良方案_v2_工程版.docx` 与 `ROSA_统一路线图_TODO合并版.docx`，按 P0 -> P1 的顺序推进 ROSA 改良，优先完成在线化基础设施，再逐步接入 PLE 风格的 value store 与 Engram 风格的门控/调度。

## 当前迭代

- 分支：`codex/online-rosa-p0-foundation`
- 迭代主题：在线主线重构准备
- 规模化专项：新增 `ROSA_规模化数据与模型_TODO.md`，把“大数据集 / 大模型 / 更重 memory backend”路线单独拆出，避免和在线主线任务混在一起
- 当前状态：已完成在线主线 `P1-3`、`P1-2`、`P1-4`、`P1-1` 与 `P1-5`；suffix automaton state、统一 session 生命周期、prefetch/cache 主线、在线训练 V1 配方和统一层位 sweep 协议都已接入
- 当前补充优化：已完成在线训练地址缓存，将 `online_seq + online_exact/online_sam` 的训练期地址构建前移到数据集阶段，优先缓解 `model_rosa_address` 瓶颈
- 当前策略调整：训练地址缓存不再作为默认主路径，后续以“异步地址支路 + 高性能 SAM sequence 实现”为优先优化方向
- 当前进展：训练地址异步预取已落地，可在不持久缓存整数据集的前提下，把 `online_seq` 地址准备与 GPU 主干训练做 overlap
- 当前进展：`online_sam` 的 sequence 快路径已落地，默认可通过 `--rosa_online_sam_impl fast` 走整段 `sam_rosa_predict`；`stateful` 保留为逐 token 回归实现
- 当前进展：`online_v2` 配方已落地，把 `per_layer ValueStore` 正式接入在线训练主线实验入口，训练 / profile / sweep 现在都能直接通过 recipe 复用这组配置
- 当前进展：训练期 `snapshot + 短 replay` 第一版已落地，当前可在文档级缓存稀疏 `RosaStateSnapshot`，并在 chunk 起点恢复在线状态而不必为每个 sample 复制整段地址表
- 当前进展：训练期地址异步预取 v2 已落地，当前支持可配置 `prefetch depth`，并把后台等待 / 准备 / 队列填充率接入 training timing
- 当前进展：`online_sam` 的编译型 CPU sequence 路径已落地，当前可通过 `--rosa_online_sam_impl compiled_cpu` 走 C++ 扩展
- 当前新任务：开始落地“预分词 + `memmap`/二进制数据集管线”，目标是让 `doc_local + online_seq` 在不把全部 token/sample 常驻 Python list 的前提下直接训练
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
- 已在 `rosa_runtime.py` 中新增 `RosaHotAddressCache`，支持按层 LRU 热点缓存、频次统计与 top-hot 地址报告。
- 已在 `RosaFusedLM` 中接入 `--rosa_hot_cache_size`，并将命中率、fill/evict、active entries 注入 profiling / eval 输出。
- 小型 cache smoke 已补齐；当前在 `min_match_len=1` 的 toy profile 上，prefill / decode token hit rate 约为 `0.98 / 0.96`，说明热点地址读取显著减少；由于 value backend 仍是本地 embedding，端到端平均时延基本持平。
- 已新增 `RosaAddressEngine`，统一 sequence-level `forward_seq()` 与 decode-level `forward_step()` 接口。
- 已新增 `--rosa_seq_address_mode reference_backend|online_exact`，当前可在不依赖预计算地址表的情况下跑整段 online sequence addressing。
- 已用单测与 smoke 验证 `online_exact` 与现有 reference 语义对齐。
- 已新增 `--rosa_train_mode online_seq|reference_precompute`，默认训练主入口切到 `online_seq`。
- `doc_local` / `global_train` 在线训练路径现在默认提供 `full doc prefix` 左侧 memory，不再默认依赖 `rosa_precomputed_ids`。
- 旧 `doc_local + sam precompute` 已退为显式 `reference_precompute` 回归路径。
- 已新增 `rosa_training_cache.py`，并让 `online_seq + online_exact/online_sam` 默认在训练期缓存整文档 sequence 地址。
- 已新增 `--disable_rosa_train_address_cache`，用于保留“逐 batch 现场构建地址”的对照路径。
- `profile_rosa_online_baseline.py` / `rosa_layer_sweep.py` 已同步兼容训练地址缓存数据集。
- 已在模型输出统计中加入地址来源标记：
  - `rosa_address_source_precomputed`
  - `rosa_address_source_online_seq`
  - `rosa_address_source_reference_seq`
  - `rosa_address_source_online_step`
- 已补在线训练 smoke：在不传 `rosa_precomputed_ids` 的情况下，`forward()` / `train_one_model()` 都能稳定前向、反传并完成一轮优化。
- 已在 `profile_rosa_online_baseline.py` 中新增 `train_path_consistency` 报告段，对比：
  - `online_seq` 训练样本地址路径
  - `reference_precompute` 训练样本地址路径
- 当前 smoke 结果显示：
  - `train path address agreement = 1.0`
  - `train path logit diff = 0.0`
- 已新增独立模块 `rosa_addressing.py`，承载：
  - 地址元数据
  - reference/fallback 地址实现
  - `RosaAddressEngine`
  - 真正的在线 suffix automaton state
- `OnlineRosaState` 现在已切到 SAM 状态结构；旧的 list-state 以 `ExactMatchRosaState` 形式保留为 fallback/reference。
- 已新增 `online_sam` sequence mode，并通过单测与 profiling smoke 验证：
  - `forward_seq()` 与 reference backend 地址结果逐位置一致
  - `forward_step()` 与 `forward_seq()` 在 memory prefill 后保持一致
  - `logit diff = 0.0`
- 已新增独立模块 `rosa_session.py`，提供：
  - `RosaBatchSession.prefill_seq()`
  - `RosaBatchSession.decode_step()`
  - `RosaBatchSession.snapshot()`
  - `RosaBatchSession.reset()`
- `RosaFusedLM` 已支持 `init_online_session(batch_size)`，profiling 的 online prefill/decode 路径现在复用同一 session API。
- session 回归已验证：
  - `prefill_seq()` 与旧 `forward_online()` 数值一致
  - `prefill + decode_step()` 与 memory reference 路径对齐
  - snapshot 能稳定反映 token 计数与状态生命周期
- `RosaBatchSession` 现已支持：
  - `schedule_decode_step()`
  - `prefetch_stats()`
  - 预取开启时的独立 schedule/runtime state 生命周期
- `profile_rosa_online_baseline.py --rosa_prefetch` 现已通过 `RosaBatchSession` 走主线，不再绕过 session 单独管理预取状态。
- 当前 smoke 报告：
  - `outputs/profile_p1_session_prefetch_smoke/profile_report.json`
  - decode `hot_cache token_hit_rate = 0.8`
  - 地址一致性与 logit diff 仍保持对齐
- 已新增独立模块 `rosa_recipes.py`，提供统一的 ROSA recipe/preset 入口。
- 当前已内置 `online_v1`，会固定在线训练主线的第一版推荐配置：
  - `online_seq`
  - `online_sam`
  - `shared value`
  - 单早层注入
  - `context gate + match_len prior`
- 已新增在线训练 V1 smoke 入口：
  - `outputs/p1_online_v1_smoke_qwen/comparison.json`
  - 当前小样本结果显示 ROSA test loss / token acc 相比 baseline 仍为正向
- 已新增独立模块 `rosa_layer_sweep.py`，统一承载层位集合解析、训练小样本 sweep、profile sweep 与合并排序。
- `scan_rosa_injection_layers.py` 现在只保留 CLI 包装，真正逻辑都收敛到 `rosa_layer_sweep.py`。
- 当前统一 sweep 支持：
  - `--experiment_mode profile`
  - `--experiment_mode train`
  - `--experiment_mode both`
- 当前 train+profile smoke 入口：
  - `outputs/scan_p1_train_profile_smoke/layer_scan_report.json`
  - 每个层位组合会同时产出 `profile_report.json` 和 `train_summary.json`
- 已在 `model` 环境执行 `python -m unittest discover -s tests`，当前通过。
- 已新增 `--rosa_online_sam_impl fast|stateful`，并把 `online_sam` 的训练/sequence 路径正式拆成：
  - `fast`：整段 array-backed `sam_rosa_predict`
  - `stateful`：逐 token `SuffixAutomatonRosaState.update_one()`
- `forward_step()` / session decode 仍保留真实在线 `stateful` 生命周期；只对 `forward_seq()` 做高性能下沉
- 已补 fast vs stateful 一致性回归，并验证：
  - helper 级地址元数据一致
  - `RosaAddressEngine.forward_seq()` 一致
  - `RosaFusedLM` address/logits 一致
- 纯地址 microbenchmark（`B=4, T=128, M=256`）当前结果：
  - `stateful ~4.734ms`
  - `fast ~1.836ms`
  - 地址层约 `2.58x` 加速
- 小型同步训练 smoke（关闭 async）当前结果：
  - `rosa_addr ~40.41ms -> ~39.56ms`
  - `step ~51.38ms -> ~50.60ms`
  - 端到端收益较温和，但训练指标与地址输出保持一致
- 已新增 `online_v2` 配方：
  - `online_seq`
  - `online_sam`
  - `per_layer value`
  - `context gate`
  - 单早层注入
- profiling 入口现已支持 `--rosa_recipe online_v2`
- 小型 MiniPile 实验（`32/8/8 docs`, `fast + async`）当前结果：
  - `online_v1`：`test loss 16.5346`，`token_acc 0.04419`，`step ~35.50ms`
  - `online_v2`：`test loss 16.5328`，`token_acc 0.04602`，`step ~74.60ms`
  - 说明 `per_layer` 在这次小实验上带来轻微效果增益，但参数量和训练开销明显上升，后续仍需更系统评估
- 已新增 `rosa_training_snapshot.py`，支持：
  - 文档级稀疏 state snapshot 构建
  - `AddressEngine.forward_seq_from_snapshots()`
  - batch 预处理阶段从 snapshot 恢复并做 chunk 起点短 replay
- 小型 MiniPile 同步训练实验（`32/8/8 docs`, `fast + sync`）当前结果：
  - full prefix：`rosa_addr ~27.00ms`，`step ~59.36ms`
  - snapshot interval 256：`rosa_addr ~21.08ms`，`step ~55.65ms`
  - test 指标保持一致（`loss 16.5346`，`token_acc 0.04419`）
  - 说明 snapshot + replay v1 已经在不改 full-history 语义的前提下，进一步压低了同步地址开销
- 小型 sweep 实验（`16/4/4 docs`, `online_v1 + fast`）当前结果：
  - sync:
    - no snapshot：`step ~64.04ms`，`rosa_addr ~30.50ms`
    - snapshot 64：`step ~48.56ms`，`rosa_addr ~15.60ms`
    - snapshot 128：`step ~48.83ms`，`rosa_addr ~15.77ms`
    - snapshot 256：`step ~50.47ms`，`rosa_addr ~16.96ms`
    - snapshot 512：`step ~50.90ms`，`rosa_addr ~17.41ms`
  - async:
    - no snapshot：`step ~36.51ms`，`rosa_addr ~0.22ms`
    - snapshot 64/128/256/512：`step ~37.38~38.55ms`，`rosa_addr ~0.23~0.28ms`
  - 当前结论：
    - snapshot 对 sync 路径收益明确，`64~128` 区间最好
    - async 已经把地址等待几乎完全隐藏，在这个小配置上再叠 snapshot 没有额外收益
- 小型 async depth 实验（`16/4/4 docs`, `online_v1 + fast + async`）当前结果：
  - depth 1：`step ~103.47ms`，`rosa_addr ~0.75ms`，`async_wait ~0.09ms`
  - depth 2：`step ~103.62ms`，`rosa_addr ~0.72ms`，`async_wait ~0.03ms`
  - depth 4：`step ~104.27ms`，`rosa_addr ~0.79ms`，`async_wait ~0.03ms`
  - 当前结论：
    - 更深的预取队列已经可用，但这组小配置里默认 `depth=1` 仍然最好
    - `depth>1` 的价值更偏向长前缀 / 更重地址支路 / 后台线程开始吃紧时的稳定性提升
- 编译型 CPU microbenchmark（`B=4, T=128, M=256`）当前结果：
  - `stateful ~3.984ms`
  - `fast ~2.723ms`
  - `compiled_cpu ~0.944ms`
  - 当前结论：
    - `compiled_cpu` 已经明显快于现有 Python `fast`
    - 这条路径现在适合作为下一阶段主推的高性能 sequence 实现候选
- 同配置训练 smoke（`16/4/4 docs`, `online_v1 + async depth=1`）当前结果：
  - `fast`：`step ~36.20ms`，`async_prep ~24.51ms`
  - `compiled_cpu`：`step ~35.22ms`，`async_prep ~10.55ms`
  - 当前结论：
    - 编译型 CPU 版已经显著压低了后台地址准备时间
    - 端到端 step 也出现了小幅但稳定的改善

## 下一任务

1. 在线主线 P1 已收束，后续可按新 TODO 进入 P2 的 `ROSA-DocMemory`。
2. 若继续做训练主线增强，优先把 `per-layer ValueStore` 作为在线训练默认实验对象之一。
3. 训练性能优化后续优先项：
   - 地址支路 CPU worker 前移 / next-batch overlap（继续作为默认主线优化；当前 queue depth 可调，但默认先保持 `1`）
   - `online_sam` sequence 快路径进一步下沉到 CUDA/Triton（C++ CPU v1 已完成）
   - 状态快照进一步轻量化 / 磁盘化 / 更细粒度间隔（更偏 sync / 无 async / 大数据场景）
   - memory window / bookmark（保留为可选工程折中，而非默认主线）
4. 当前已用 `--train_timing` 验证并完成两轮修复：
   - async overlap 把训练中等待地址的成本压到近零
   - `fast online_sam` 把纯 sequence 地址层成本压到 `stateful` 的约 `39%`

## 自我验证清单

- 单元测试覆盖 `prefill`、`update_one`、`reset`、special target 过滤、逐 token 对齐。
- 在线状态输出与当前 reference 检索结果保持一致。
- 现有 ROSA / 数据加载 / dataset 测试不回归。
- 提交前检查 `git diff`，确认文档、代码、测试都已同步。

## 下一阶段预留

- 最小在线注入闭环：单层注入，先继续复用共享 `embed_tokens`。
- 性能基线：补 decode/prefill profiling 与 match coverage 指标。
- P1：per-layer value store + context-aware gate。
