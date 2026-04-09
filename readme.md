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

## 预分词 + `memmap` 数据集

- 构建脚本：`build_rosa_memmap_dataset.py`
- 运行时模块：`rosa_memmap_dataset.py`
- 训练入口：`--pretokenized_manifest`
- 当前用途：
  - 将原始 `text/json/jsonl` 语料离线预分词
  - 写成：
    - `train.tokens.bin`
    - `train.offsets.npy`
    - `train.lengths.npy`
    - `dataset_manifest.json`
  - 训练时按文档偏移动态切片，不再把全部 token/sample 展平到 Python list
- 当前已支持：
  - `doc_local + online_seq`
  - `global_train + online_seq`
- 当前尚未接入：
  - `reference_precompute`
  - 训练地址缓存
  - 训练状态快照

构建预分词数据集：

```powershell
conda run -n model python build_rosa_memmap_dataset.py `
  --train_data_path data/minipile/train-00000-of-00012-6fbcb5acda05b3c0.jsonl `
  --val_data_path data/minipile/validation-00000-of-00001-a2192e61a091cecb.jsonl `
  --test_data_path data/minipile/test-00000-of-00001-010a6231c4b54d31.jsonl `
  --data_format jsonl `
  --json_text_keys text `
  --max_train_docs 64 `
  --max_val_docs 16 `
  --max_test_docs 16 `
  --tokenizer_name_or_path D:/codes/Qwen3.5-0.8B `
  --out_dir outputs/minipile_memmap_qwen
```

直接从 manifest 训练：

```powershell
conda run -n model python train_qwen_llama_vs_rosa_v2.py `
  --pretokenized_manifest outputs/minipile_memmap_qwen/dataset_manifest.json `
  --tokenizer_name_or_path D:/codes/Qwen3.5-0.8B `
  --arch_style qwen `
  --seq_len 64 `
  --batch_size 2 `
  --epochs 1 `
  --dim 64 `
  --n_layers 2 `
  --n_heads 4 `
  --n_kv_heads 4 `
  --intermediate_size 128 `
  --rosa_recipe online_v1 `
  --out_dir outputs/minipile_memmap_online_v1
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
  - `session.schedule_decode_step(step_ids)`（启用 prefetch 时）
  - `session.prefetch_stats()`
  - `session.snapshot()`
- 当前版本仍使用共享 `embed_tokens` 作为 value，适合先验证在线路径与语义一致性。

## Per-Layer Value Store

- 训练/评测开关：`--rosa_value_mode shared|per_layer`
- `shared`：复用 `embed_tokens`
- `per_layer`：每个注入层使用独立 value table
- 当前初始化策略：`per_layer` value table 从共享词嵌入复制初值，方便和旧路径做平滑对比

## Sparse ValueStore Training

- 开关：`--rosa_sparse_value_training`
- 预设配方：`--rosa_recipe online_v2_sparse`
- 当前作用：
  - 仅对 `--rosa_value_mode per_layer` 生效
  - 将 per-layer value table 切成稀疏梯度 `Embedding(sparse=True)`
  - 训练时自动拆成：
    - 普通参数 -> `AdamW`
    - per-layer value table -> `SparseAdam`
- 当前输出统计：
  - `rosa_sparse_value_training`
  - `rosa_active_address_count`
  - `rosa_active_address_fraction`
- 当前定位：
  - 主要为大词表 / 大模型下的 optimizer state 和活跃行训练做准备
  - 不是分片 `ValueStore` 的最终形态
  - 目前尚未做跨卡分片 / host memory / all-to-all

## Sharded ValueStore

- 开关：`--rosa_value_shards N`
- 预设配方：`--rosa_recipe online_v2_sparse_sharded`
- 当前作用：
  - 仅对 `--rosa_value_mode per_layer` 生效
  - 将每层 value table 按词表行切成多个本地 shard
  - 每个地址会先映射到所属 shard，再到 shard 内部做 lookup
- 当前输出统计：
  - `rosa_value_shards`
  - `rosa_value_store_sharded`
  - `rosa_active_value_shards`
- 当前定位：
  - 这是“单机本地行分片 v1”
  - 适合继续放大词表 / per-layer table 的单机训练实验
  - 还不是跨卡 / host-memory / all-to-all 的最终规模化方案
  - 与 `FSDP` 的关系：
    - 这类“本地行分片”本身不会破坏 FSDP
    - 但它也不会自动变成跨卡 all-to-all 表
    - 当前首版 `FSDP` 暂不支持和 `--rosa_sparse_value_training` 一起用

## Scale Training Infra

- 开关：
  - `--activation_checkpointing`
  - `--grad_accum_steps`
  - `--save_every_epochs`
  - `--resume_from`
  - `--run_models baseline|rosa_fused|both`
- 当前作用：
  - 逐层 activation checkpointing，降低大模型训练显存占用
  - 梯度累积，支持小 batch micro-step + 大等效 batch
  - epoch 级 checkpoint/save-resume
  - 可只训练 `rosa_fused`，避免服务器上顺带再跑一遍 baseline

## Distributed Training

- 开关：
  - `--distributed_strategy ddp|fsdp`
  - `--distributed_backend gloo|nccl`
- 当前作用：
  - 支持用 `torchrun` 或显式 `RANK/WORLD_SIZE/MASTER_*` 环境变量启动
  - 自动接入 `DistributedSampler`
  - 训练/验证/测试指标做跨 rank 归约
  - DDP/FSDP 路径都支持 checkpoint/save-resume v1
- 当前边界：
  - `FSDP` 首版暂不支持 `--rosa_sparse_value_training`
  - 当前 FSDP checkpoint 仍是 full-state 保存，适合先跑通，不是最终高效形态

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
- `online_sam` 额外支持：
  - `--rosa_online_sam_impl fast`
  - `--rosa_online_sam_impl compiled_cpu`
  - `--rosa_online_sam_impl stateful`
- 含义：
  - `online_seq`：新的训练主线，数据集默认只提供 `input_ids / labels / optional_memory`，由前向内部调用 `AddressEngine.forward_seq()`
  - `reference_precompute`：保留旧的 `doc_local + sam precompute` 回归路径
  - `reference_backend`：继续走现有整段 reference 地址逻辑
  - `online_exact`：按左上下文顺序扫描整段，生成 `[B, T]` 地址结果
  - `online_sam`：使用真正的在线 suffix automaton state 生成整段/逐步地址
  - `online_sam + fast`：训练/sequence 路径使用整段 `sam_rosa_predict` 快路径
  - `online_sam + compiled_cpu`：训练/sequence 路径使用编译型 CPU C++ 扩展
  - `online_sam + stateful`：训练/sequence 路径也严格复用逐 token stateful SAM
- 当前用途：
  - 作为在线训练主线的第一版统一入口
  - 当前训练默认已切到 `online_seq + online_exact`
  - `doc_local` / `global_train` 在线模式默认提供 `full doc prefix` 左侧 memory，用于和旧 reference 路径对齐
  - 对 `online_exact` / `online_sam`，当前训练阶段默认回到逐 batch 在线构建地址
  - 如需显式启用“训练地址缓存”，可加：`--enable_rosa_train_address_cache`
  - 如需启用“训练状态快照 + 短 replay”，可加：
    - `--enable_rosa_train_state_snapshot`
    - `--rosa_train_state_snapshot_interval 256`
  - 当前训练默认会开启“地址异步预取”：主干训练当前 batch 时，CPU 后台准备下一 batch 的在线地址
  - 如需关闭，可加：`--disable_rosa_train_address_async`
  - 如需调整后台预取队列深度，可加：`--rosa_train_address_async_prefetch_batches N`
- 当前训练默认推荐：`--rosa_online_sam_impl fast`
- 如果本机已具备 C++ 编译环境，当前更高性能的候选实现是：`--rosa_online_sam_impl compiled_cpu`
  - 训练 / 评测输出里可直接看地址来源统计：
    - `rosa_address_source_precomputed`
    - `rosa_address_source_online_seq`
    - `rosa_address_source_reference_seq`
    - `rosa_address_source_online_step`
  - 当前 `OnlineRosaState` 已切到真正的在线 SAM 状态；旧 list-state 以 `ExactMatchRosaState` 保留作 fallback/reference

## Recipe / Preset

- 配置模块：`rosa_recipes.py`
- 开关：`--rosa_recipe custom|online_v1|online_v2|online_v2_sparse|online_v2_sparse_sharded`
- 当前推荐主线：`--rosa_recipe online_v1`
- `online_v1` 会固定：
  - `--rosa_train_mode online_seq`
  - `--rosa_memory_mode doc_local`
  - `--rosa_backend sam`
  - `--rosa_seq_address_mode online_sam`
  - `--rosa_value_mode shared`
  - `--rosa_context_gate`
  - `--rosa_inject_layers 1`
  - `--rosa_inject_layer_ids 0`
  - `--rosa_min_match_len 1`
  - `--rosa_scale 0.15`
- `online_v2` 会固定：
  - `--rosa_train_mode online_seq`
  - `--rosa_memory_mode doc_local`
  - `--rosa_backend sam`
  - `--rosa_seq_address_mode online_sam`
  - `--rosa_value_mode per_layer`
  - `--rosa_context_gate`
  - `--rosa_inject_layers 1`
  - `--rosa_inject_layer_ids 0`
  - `--rosa_min_match_len 1`
  - `--rosa_scale 0.15`
- `online_v2_sparse` 会在 `online_v2` 基础上额外固定：
  - `--rosa_sparse_value_training`
- `online_v2_sparse_sharded` 会在 `online_v2_sparse` 基础上额外固定：
  - `--rosa_value_shards 4`
- `online_v1` 当前默认仍走在线主线语义，不会自动启用训练地址缓存
- `online_v2` 当前更适合作为“进阶实验 recipe”，用于验证 per-layer ValueStore 是否值得正式进入主线
- `online_v2_sparse` / `online_v2_sparse_sharded` 当前更适合作为规模化实验入口：
  - 前者验证稀疏活跃行训练
  - 后者验证单机本地分片表存储
- 典型 smoke：

```powershell
conda run -n model python train_qwen_llama_vs_rosa_v2.py `
  --train_data_path data/minipile/train-00000-of-00012-6fbcb5acda05b3c0.jsonl `
  --val_data_path data/minipile/validation-00000-of-00001-a2192e61a091cecb.jsonl `
  --test_data_path data/minipile/test-00000-of-00001-010a6231c4b54d31.jsonl `
  --data_format jsonl `
  --json_text_keys text `
  --max_train_docs 24 `
  --max_val_docs 8 `
  --max_test_docs 8 `
  --tokenizer_name_or_path D:/codes/Qwen3.5-0.8B `
  --arch_style qwen `
  --seed 42 `
  --seq_len 64 `
  --stride 64 `
  --batch_size 2 `
  --epochs 1 `
  --lr 3e-4 `
  --weight_decay 0.01 `
  --grad_clip 1.0 `
  --dim 64 `
  --n_layers 2 `
  --n_heads 4 `
  --n_kv_heads 4 `
  --intermediate_size 128 `
  --dropout 0.0 `
  --rosa_recipe online_v1 `
  --out_dir outputs/p1_online_v1_smoke_qwen
```

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
- 统一扫描模块：`rosa_layer_sweep.py`
- CLI 入口：`scan_rosa_injection_layers.py`
- 统一实验模式：`--experiment_mode profile|train|both`
- 典型用途：
  - 单层扫描：`--scan_mode single`
  - 双层扫描：`--scan_mode pair`
- 输出文件：
  - 合并报告：`<out_dir>/layer_scan_report.json`
  - 单层位 profile 报告：`<out_dir>/layers_*/profile/profile_report.json`
  - 单层位训练报告：`<out_dir>/layers_*/train/train_summary.json`
- 当前推荐做法：
  - 只看推理侧层位：`--experiment_mode profile`
  - 同时对齐训练/推理层位：`--experiment_mode both --rosa_recipe online_v1`

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

## Training Timing

- 开关：`--train_timing`
- 作用：输出训练/评估阶段的同步 timing 统计，帮助判断 `rosa_fused` 到底慢在：
  - `data_wait`
  - `forward`
  - `backward`
  - `optim_step`
  - `model_rosa_address`
  - `model_rosa_payload`
  - `model_trunk`
  - `model_head`
- 说明：
  - 在 CUDA 上会显式 `synchronize`，统计更准，但会带来额外开销
  - 适合定位瓶颈，不建议长期作为默认训练配置
- 当前一次 `64/16/16` 小实验观测：
  - baseline `step ~37.9ms`
  - rosa_fused `step ~124.1ms`
  - 其中 `rosa_addr ~85.7ms`
  - 说明当前体感变慢主要来自在线地址生成，而不是主干 Transformer 或 value payload
- 当前一次 `8/4/4` 小实验对比（`online_seq + online_sam`）：
  - 关闭训练地址缓存：`rosa_addr ~85.1ms`，`step ~133.5ms`
  - 开启训练地址缓存：`rosa_addr ~0.2ms`，`step ~56.9ms`
  - 说明当前最有效的训练加速手段，是把在线 sequence 地址从“每个 chunk 前向重放”改成“数据集构建期缓存”
- 当前一次 `8/4/4` 小实验对比（训练地址异步预取，缓存关闭）：
  - 同步地址：`rosa_addr ~44.7ms`，`step ~103.3ms`
  - 异步地址：`rosa_addr ~0.5ms`，`step ~69.1ms`
  - 说明当前更适合作为主线的加速方案，是“next-batch 地址异步预取”，而不是默认整数据集缓存
- 当前一次异步预取深度小实验（`16/4/4 docs`, `online_v1 + fast + async`）：
  - depth 1：`step ~103.47ms`，`async_wait ~0.09ms`
  - depth 2：`step ~103.62ms`，`async_wait ~0.03ms`
  - depth 4：`step ~104.27ms`，`async_wait ~0.03ms`
  - 说明更深的预取队列已经可用，但在这组小配置里默认 `depth=1` 仍然最好；`depth>1` 更偏向在更长前缀和更重地址支路下提供稳定性
- 当前一次编译型 CPU microbenchmark（`B=4, T=128, M=256`）：
  - `stateful ~3.984ms`
  - `fast ~2.723ms`
  - `compiled_cpu ~0.944ms`
- 当前一次同配置训练 smoke（`16/4/4 docs`, `online_v1 + async depth=1`）：
  - `fast`：`step ~36.20ms`，`async_prep ~24.51ms`
  - `compiled_cpu`：`step ~35.22ms`，`async_prep ~10.55ms`
  - 说明编译型 CPU 版已经显著减少后台地址准备时间，并带来小幅端到端训练收益
- 当前一次纯地址 microbenchmark（`B=4, T=128, M=256`）：
  - `online_sam stateful ~4.734ms`
  - `online_sam fast ~1.836ms`
  - 说明 `fast` 已经把 sequence 地址层成本压到 stateful 的约 `39%`
- 当前一次小型同步训练 smoke（关闭 async）：
  - `stateful`: `rosa_addr ~40.41ms`，`step ~51.38ms`
  - `fast`: `rosa_addr ~39.56ms`，`step ~50.60ms`
  - 说明在真实训练闭环里，`fast` 已经是更好的默认 sequence 实现，但端到端收益会被主干/反传与 CUDA 同步噪声部分稀释
- 当前一次 MiniPile 小实验（`32/8/8 docs`, `fast + async`）：
  - `online_v1`: `test loss 16.5346`，`token_acc 0.04419`，`step ~35.50ms`
  - `online_v2`: `test loss 16.5328`，`token_acc 0.04602`，`step ~74.60ms`
  - 说明 per-layer ValueStore 有轻微效果增益，但当前训练成本上升明显
- 当前一次 MiniPile 同步训练小实验（`32/8/8 docs`, `fast + sync`）：
  - full prefix：`rosa_addr ~27.00ms`，`step ~59.36ms`
  - `snapshot interval 256`：`rosa_addr ~21.08ms`，`step ~55.65ms`
  - test 指标保持一致：`loss 16.5346`，`token_acc 0.04419`
  - 说明 `snapshot + replay v1` 已经能在保留 full-history 语义的前提下，继续压低同步地址成本
- 当前一次 `snapshot interval sweep` 小实验（`16/4/4 docs`, `online_v1 + fast`）：
  - 脚本：`sweep_rosa_snapshot_intervals.py`
  - 汇总：`outputs/snapshot_interval_sweep_smoke/snapshot_sweep_summary.json`
  - sync 最优：`snapshot 64`
    - `step ~48.56ms`
    - `rosa_addr ~15.60ms`
  - async 最优：`no snapshot`
    - `step ~36.51ms`
    - `rosa_addr ~0.22ms`
  - 当前建议：
    - 主线默认仍优先保留 `async overlap`
    - `snapshot` 更适合作为 sync / 无 async / 长前缀恢复场景下的额外优化

## VS Code Launch

- `ROSA v2 - Online Baseline Profile (Smoke)`：快速验证脚本链路。
- `ROSA v2 - Online Baseline Profile (Qwen ckpt compare)`：用现有 Qwen checkpoint 直接比较 reference vs online。
- `ROSA v2 - P1 Value+Gate Smoke (Qwen)`：直接训练一版 `per_layer + context_gate` 小实验。
- `ROSA v2 - Online V1 Smoke (Qwen)`：在线主线 V1 配方的小样本训练入口。
- `ROSA v2 - Compare Small Online V1 (64 docs)`：用 `64/16/16` docs 对比 baseline vs ROSA 的小实验入口。
- `ROSA v2 - Online Baseline Profile (P1 Value+Gate Smoke)`：快速看 P1 组合路径是否跑通。
- `ROSA v2 - Online Baseline Profile (P1 Prefetch Smoke)`：快速看 prefetch/staging 统计是否正常。
- `ROSA v2 - Online Baseline Profile (P1 Hot Cache Smoke)`：快速看热点缓存的命中率与 tail latency。
- `ROSA v2 - Injection Layer Scan (Smoke)`：快速扫描不同注入层位。
- `ROSA v2 - Injection Layer Sweep (Train+Profile Smoke)`：同一层位组合同时跑训练小样本与 profile。

## 当前开发约定

- 新任务统一在 `codex/*` 分支上进行。
- 每完成一个可独立验收的任务，更新 `progress.md`，然后提交一版 commit。
- 规划与拆解写入 `planning.md`，执行过程中持续修订。
- 每一步完成后都要做自我验证：跑测试、看日志、核对输出，再决定是否进入下一步。
