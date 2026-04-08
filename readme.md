# ROSA 本地操作备忘

## 环境

- Conda 环境：`model`
- 主训练脚本：`train_qwen_llama_vs_rosa_v2.py`
- 主测试入口：`tests/test_train_qwen_llama_vs_rosa_v2.py`
- 地址/状态模块：`rosa_addressing.py`
- 在线 session 模块：`rosa_session.py`

## 常用命令

```powershell
conda activate model
python -m unittest discover -s tests
```

```powershell
git status --short --branch
git log --oneline -5
```

## Online ROSA 最小闭环

- 入口能力：
  - `RosaFusedLM.init_online_state(batch_size)`
  - `RosaFusedLM.init_online_session(batch_size)`
  - `OnlineRosaBatchState.prefill(token_ids, pad_id=...)`
  - `RosaFusedLM.forward_online(input_ids, rosa_online_state=...)`
- 当前更推荐的统一生命周期接口：
  - `session = model.init_online_session(batch_size)`
  - `session.prefill_seq(prompt_ids)`
  - `session.decode_step(step_ids)`
  - `session.snapshot()`
- 当前版本仍使用共享 `embed_tokens` 作为 value，适合先验证在线路径与语义一致性。

## Per-Layer Value Store

- 训练/评测开关：`--rosa_value_mode shared|per_layer`
- `shared`：复用 `embed_tokens`
- `per_layer`：每个注入层使用独立 value table
- 当前初始化策略：`per_layer` value table 从共享词嵌入复制初值，方便和旧路径做平滑对比

## Context-Aware Gate

- 开关：`--rosa_context_gate`
- 作用：用当前 hidden state 与 memory value 的相互作用决定注入强度
- 当前输出统计：
  - `rosa_avg_gate`
  - `rosa_gate_coverage`
  - `rosa_gate_hit`
- `match_len` 仍然作为先验项参与 gate 计算；若关闭 `--rosa_disable_match_len_gate`，则只保留上下文项

## Runtime Payload

- 运行时数据结构在 `rosa_runtime.py`
- 调度拆分后的典型流程：
  - `compute_rosa_address_batch()`
  - `build_rosa_injection_payload()`
  - `forward(..., rosa_payload=...)`
- 这层接口是后续预取、缓存、层外调度的基础

## Address Engine

- 统一入口：`RosaAddressEngine`
- 训练主线开关：`--rosa_train_mode online_seq|reference_precompute`
- 当前支持两种 sequence 模式：
  - `--rosa_seq_address_mode reference_backend`
  - `--rosa_seq_address_mode online_exact`
  - `--rosa_seq_address_mode online_sam`
- 含义：
  - `online_seq`：新的训练主线，数据集默认只提供 `input_ids / labels / optional_memory`，由前向内部调用 `AddressEngine.forward_seq()`
  - `reference_precompute`：保留旧的 `doc_local + sam precompute` 回归路径
  - `reference_backend`：继续走现有整段 reference 地址逻辑
  - `online_exact`：按左上下文顺序扫描整段，生成 `[B, T]` 地址结果
  - `online_sam`：使用真正的在线 suffix automaton state 生成整段/逐步地址
- 当前用途：
  - 作为在线训练主线的第一版统一入口
  - 当前训练默认已切到 `online_seq + online_exact`
  - `doc_local` / `global_train` 在线模式默认提供 `full doc prefix` 左侧 memory，用于和旧 reference 路径对齐
  - 训练 / 评测输出里可直接看地址来源统计：
    - `rosa_address_source_precomputed`
    - `rosa_address_source_online_seq`
    - `rosa_address_source_reference_seq`
    - `rosa_address_source_online_step`
  - 当前 `OnlineRosaState` 已切到真正的在线 SAM 状态；旧 list-state 以 `ExactMatchRosaState` 保留作 fallback/reference

## Prefetch / Staging

- 入口能力：
  - `init_prefetcher()`
  - `schedule_rosa_prefetch()`
  - `consume_rosa_prefetch()`
- profiling 开关：
  - `--rosa_prefetch`
  - `--rosa_prefetch_pinned`
- 当前说明：
  - CPU 路径支持异步预取
  - GPU 路径先走安全同步回退，但统计接口一致

## Hot Address Cache

- 开关：`--rosa_hot_cache_size`
- 运行时实现：`rosa_runtime.py` 中的 `RosaHotAddressCache`
- 当前策略：
  - 按注入层独立 LRU
  - 仅在 `eval/profile` 路径启用，训练路径默认绕过，避免干扰梯度
  - profiling 报告会分别输出 `prefill` / `decode_micro` 两段 cache 命中率
  - 当前 toy smoke 上能稳定看到高 hit rate；但由于 value backend 仍是本地 embedding，端到端平均时延可能只小幅变化或基本持平
- 重点指标：
  - `prefill_hot_cache_token_hit_rate`
  - `decode_hot_cache_token_hit_rate`
  - `rosa_hot_cache_active_entries`
  - `hot_cache.layer_stats[].top_addresses`

## Injection Layer Scan

- 显式层位：`--rosa_inject_layer_ids`
- 扫描脚本：`scan_rosa_injection_layers.py`
- 典型用途：
  - 单层扫描：`--scan_mode single`
  - 双层扫描：`--scan_mode pair`
- 输出文件：`<out_dir>/layer_scan_report.json`

## Profiling 基线

- profiling 脚本：`profile_rosa_online_baseline.py`
- 输出文件：`<out_dir>/profile_report.json`
- 对比模式：
  - `baseline`
  - `rosa_reference`
  - `rosa_online`
- 当前还会额外输出：
  - `train_path_consistency`
  - 用于比较 `online_seq` 训练样本路径和 `reference_precompute` 训练样本路径的地址一致性与 logit diff
- 当启用热点缓存时，报告还会包含：
  - `hot_cache.prefill`
  - `hot_cache.decode_micro`
- 当前 decode 指标是无 KV cache 的单步 microbenchmark，适合比较 ROSA 分支路径开销与一致性，不等同于最终 serving 吞吐。

## VS Code Launch

- `ROSA v2 - Online Baseline Profile (Smoke)`：快速验证脚本链路。
- `ROSA v2 - Online Baseline Profile (Qwen ckpt compare)`：用现有 Qwen checkpoint 直接比较 reference vs online。
- `ROSA v2 - P1 Value+Gate Smoke (Qwen)`：直接训练一版 `per_layer + context_gate` 小实验。
- `ROSA v2 - Online Baseline Profile (P1 Value+Gate Smoke)`：快速看 P1 组合路径是否跑通。
- `ROSA v2 - Online Baseline Profile (P1 Prefetch Smoke)`：快速看 prefetch/staging 统计是否正常。
- `ROSA v2 - Online Baseline Profile (P1 Hot Cache Smoke)`：快速看热点缓存的命中率与 tail latency。
- `ROSA v2 - Injection Layer Scan (Smoke)`：快速扫描不同注入层位。

## 当前开发约定

- 新任务统一在 `codex/*` 分支上进行。
- 每完成一个可独立验收的任务，更新 `progress.md`，然后提交一版 commit。
- 规划与拆解写入 `planning.md`，执行过程中持续修订。
- 每一步完成后都要做自我验证：跑测试、看日志、核对输出，再决定是否进入下一步。
