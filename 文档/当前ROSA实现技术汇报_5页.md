# 当前 ROSA 实现技术汇报

## 第 1 页：标题页

- 标题：当前 ROSA 实现技术方案
- 副标题：在线地址主线、SAM 状态、注入与运行时工程
- 适用对象：内部技术汇报 / 实现评审
- 仓库主入口：`train_qwen_llama_vs_rosa_v2.py`

## 第 2 页：系统目标与总体架构

- 目标：在同一套 Qwen/LLaMA 风格 decoder-only 主干上，对比 baseline 与 `rosa_fused`
- 公平性约束：主干结构一致、参数量对齐、初始化一致、训练脚本统一
- 当前训练主线：`rosa_train_mode = online_seq`
- 回归/对照路径：`rosa_train_mode = reference_precompute`
- 核心模块分层：
- 训练入口：`train_qwen_llama_vs_rosa_v2.py`
- 地址引擎：`rosa_addressing.py`
- 运行时 payload / cache：`rosa_runtime.py`
- 在线 session 生命周期：`rosa_session.py`
- profiling 与实验脚本：`profile_rosa_online_baseline.py`、`scan_rosa_injection_layers.py`

## 第 3 页：地址生成主线（AddressEngine + Online SAM）

- 统一入口：`RosaAddressEngine.forward_seq()` 与 `forward_step()`
- 当前支持三类地址模式：
- `reference_backend`：旧 reference 路径，便于回归
- `online_exact`：按左上下文顺序扫描，逐位置生成地址
- `online_sam`：使用真正的 suffix automaton state 在线生成地址
- 训练与推理统一为“先算地址，再做 value lookup”
- `OnlineRosaState` 负责在线状态维护；旧 list-state 保留为 fallback/reference
- `doc_local` 在线模式默认提供 `full doc prefix` 左侧 memory，与 reference 对齐
- 工程收益：训练 prefill、推理解码、reference 回归共享同一地址抽象

## 第 4 页：注入路径与运行时结构

- 模型入口：`RosaFusedLM`
- 地址到注入的主路径：
- `compute_rosa_address_batch()`
- `build_rosa_injection_payload()`
- `forward(..., rosa_payload=...)`
- value backend：
- `shared`：复用共享 embedding / value table
- `per_layer`：每个注入层独立 value store，初值可从共享 embedding 拷贝
- gate 机制：
- `match_len` 作为先验项
- 可选 `context-aware gate`，由当前 hidden state 与 memory value 共同决定注入强度
- 层位控制：
- `rosa_inject_layer_ids`
- `scan_rosa_injection_layers.py` / `rosa_layer_sweep.py` 支持单层与层对扫描

## 第 5 页：工程验证、性能现状与下一步

- 正确性验证：
- 在线训练路径与 `reference_precompute` 已做一致性回归
- 当前记录：`train path address agreement = 1.0`，`logit diff = 0.0`
- 测试现状：仓库当前已有 56 个测试，覆盖地址、runtime、profile、主训练路径
- 性能现状：
- baseline 平均 step 约 `37.9 ms`
- `rosa_fused` 平均 step 约 `124.1 ms`
- 主要慢点在地址支路：`rosa_addr ~85.7 ms`
- 已有工程优化：
- session 统一 `prefill + decode`
- staging / prefetch / hot cache 已接入主线
- 下一步重点：
- 地址支路异步化与 overlap
- `DocMemory` / 外部文档记忆
- 更正式的 value store 与 host memory / mmap 路线
