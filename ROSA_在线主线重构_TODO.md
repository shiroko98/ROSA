# ROSA 在线主线重构路线图（TODO）

将“旧的 doc-local/global precompute 训练路径”降级为参考/回归路径，把“teacher forcing 并行主干 + 在线 ROSA side-branch”确立为新的主线方案。

## 说明

- 这份路线图不否定旧实现的价值。`doc_local + sam precompute` 与 `global_train` 仍保留，用于：
  - 回归验证
  - 训练/推理一致性对照
  - 性能基线参考
- 新主线目标是统一训练与推理的 ROSA 组件：
  - 训练期：并行主干 + 在线地址生成 side-branch
  - 推理期：prefill + decode 共用同一套 `RosaState / AddressEngine / ValueStore / Gate`
- 新主线优先保证：
  - 语义正确
  - 训练/推理一致
  - 组件生命周期稳定
  - 可逐步扩展到更大的 ValueStore / compression / 稀疏读写

## 当前推进状态

- 已完成：训练期地址支路异步化 / overlap（默认主线已启用 next-batch 地址异步预取）
- 已完成第一版：`online_sam` sequence 快路径，默认通过 `--rosa_online_sam_impl fast` 走整段 `sam_rosa_predict`
- 仍保留：`--rosa_online_sam_impl stateful` 作为逐 token SAM 回归/对照实现
- 后续真正的高性能目标不再是 Python 级“快一点”，而是进一步下沉到 C++/CUDA/Triton 等更低开销实现

## 性能优化待办补充

- 训练地址异步预取 v2
  - 从当前 `1 worker + 1 batch ahead` 升级到可配置 queue depth
  - 比较 thread / process 两种实现
  - 支持按 batch token 数做更稳的预取调度
- 训练地址异步预取 v3
  - 把地址计算、batch 准备、host->device 拷贝拆成独立阶段
  - 尝试 pinned host buffer，减少主线程等待
  - 在 timing 中单独记录 queue wait / prepare wait / copy wait
- `online_sam` sequence C++ CPU 实现
  - 先给 `sam_rosa_predict` 做编译型 CPU 版本
  - 目标是在不依赖 GPU kernel 的情况下，先消掉 Python 循环和 dict 开销
- `online_sam` sequence CUDA/Triton 实现
  - 目标对象是训练期 `[B, T]` 的 sequence addressing
  - 不直接翻译 decode `update_one()`，而是做 batch-sequence fused kernel
  - 保持与 step/session stateful 路径逐位置一致
- 地址支路长度控制
  - 默认 `full doc prefix` 代价太高，需要支持更积极的 memory window
  - 实验固定 tail window / 动态 window / bookmark 触发
  - 把 coverage-loss-speed 三者一起记进报告
- 状态快照恢复
  - 对大数据集支持 chunk 起点 snapshot
  - 用“snapshot + 短 replay”替代“每次从长前缀重扫”
  - 减少大规模训练时的内存与预处理压力
- 训练样本调度优化
  - 增加按长度/地址成本的 bucketing
  - 减少 batch 内 address branch 的拖尾
  - 让 overlap 更容易吃满
- 更细粒度 profiling
  - 区分地址构建、地址打包、payload lookup、host/device transfer
  - 在 train/profile/sweep 三条路径输出统一 timing 口径

## 在线主线路线图

| 优先级 | 任务 | 目标输出 | 完成标准 | 主要风险 | 工程上的细分实现 |
| --- | --- | --- | --- | --- | --- |
| P0 | 明确旧路径退位为 reference/fallback | 新旧路径职责划分文档与配置开关 | 训练主入口默认不再依赖 `rosa_precomputed_ids`；旧路径仍能单独跑回归 | 主线切换过早会影响现有实验连续性 | 增加 `train_mode=reference_precompute / online_seq`；文档中明确 local/global 的定位 |
| P0 | 定义训练主形态：并行主干 + 在线地址 side-branch | `AddressEngine.forward_seq(input_ids, memory)` 接口 | 在 `[B, T]` teacher forcing 下返回整段 `addr_id/raw_match_len/fired_match_len/valid_mask` | 若实现退化成 Python 逐 token loop，训练吞吐会明显下降 | 新增 sequence-level 地址引擎接口；先允许 CPU 侧顺序扫描版本 |
| P0 | 让训练集不再以 `rosa_precomputed_ids` 为主输入 | 更干净的 `DocChunkDataset` 输出 | 数据集主要只产出 `input_ids / labels / optional_memory / meta` | 切换过程中可能影响现有 collate 与训练循环 | 保留可选 precomputed 字段，但在线主线默认不依赖它们 |
| P0 | 在训练前向中接入 `forward_seq()` | 在线训练版 `RosaFusedLM.forward()` | 不传 `rosa_precomputed_ids` 时，也能稳定训练与反传 | 地址路径与注入路径的张量 shape 容易对不齐 | `compute_rosa_address_batch()` 支持 sequence-online 模式；整段构建 `RosaAddressBatch` |
| P0 | 建立“在线训练 vs reference 预计算”一致性回归 | 对齐测试与 smoke 报告 | 在同一输入上，新训练路径与旧 reference 地址结果逐位置对齐 | online/address 代码一旦分叉，后面很难维护 | 新增逐位置地址一致性测试；在 profile 中增加训练形态对照 |
| P1 | 训练 V1：shared value + 1 个早层 + context gate | 第一版真正统一训练/推理组件的在线训练主线 | 小样本训练可跑通，loss 正常下降，指标不劣于旧主线太多 | 早期训练可能因 gate 或注入位置不稳而震荡 | value 先复用 `embed_tokens`；注入层先限定为 1 个早层；保留 `match_len prior` |
| P1 | 统一推理主形态：prefill + decode 共用 RosaState | 一个有稳定生命周期的 `RosaState` | prefill 后 decode 能无缝续接，地址结果与训练期定义一致 | request 生命周期、混批与 cache 生命周期容易出错 | 区分 `RosaState.prefill_seq()` 与 `update_one()`；统一 payload / prefetch / cache 生命周期 |
| P1 | 用真正的在线状态结构替换参考级 list-state | 高性能 `OnlineRosaState` | 不再依赖 `exact_match_step_address(list)` 做主线 | 状态机构造与 fallback/clone 逻辑复杂 | 将当前 reference state 保留为基线；新增 suffix automaton state 实现并做逐 token 对齐 |
| P1 | 让 prefetch / staging / hot cache 服务于新主线 | 在线推理流水线优化版 | decode 中能观测到稳定的 prefetch hit / cache hit，并且不破坏语义 | 当前 value backend 过轻，优化收益不容易显现 | 保留 stats 优先；先做正确性与稳定性，再评估真实收益 |
| P1 | 训练/推理统一注入层搜索协议 | 一套可复用的层位实验配置 | 训练和 profile 都支持 `inject_layer_ids`，结论可复现 | 训练期最好层位与推理期最好层位未必一致 | 扩展扫描脚本，增加训练小样本 sweep 与 profile 对齐报告 |
| P2 | 接入外部文档 memory（ROSA-DocMemory） | 类 RAG 的 ROSA 文档参考路径 | 推理时可将检索到的文档作为 `optional_memory` 注入 `RosaState`；prefill/decode 可稳定使用 | 文档排序、截断和 memory 污染会影响命中质量 | 先支持 `retrieved_docs -> token stream memory`；再考虑 bookmark / anchor / compression |
| P2 | 训练 V2：per-layer ValueStore 正式接入在线训练主线 | 在线训练版大 ValueStore | `per_layer` 成为在线训练默认实验对象之一 | 参数量和显存/主存成本上升 | 将 `per_layer` 从“功能可用”推进到“主线训练可复现” |
| P2 | tokenizer compression / canonicalization | 压缩 token 流版 AddressEngine | 能在压缩流上生成地址，并与原 token 流做对照实验 | 压缩可能伤害语义边界 | 先做轻量 canonicalization，再做压缩流实验 |
| P2 | token value -> memory value 升级 | 更正式的 Memory Value 路径 | value 不再只是 token embedding，而是可学习 memory payload | value 设计过早复杂化会拖慢主线收敛 | 先从轻量 memory value 开始，再考虑分块/低秩/量化 |
| P2 | 稀疏活跃项训练与分片 ValueStore | 大表训练基础设施 | 前向/反向只 gather 活跃项，支持更大 memory 表 | 分片/通信复杂度高 | 参考 Engram 的活跃项 gather 思路，先做单机稀疏版，再考虑多卡 |
| P2 | 在线 SAM sequence 路径下沉到高性能实现 | 比 `SuffixAutomatonRosaState.update_one()` 更快的训练期 sequence 地址引擎 | 训练期 `online_seq + online_sam` 不再主要耗时在 Python 逐 token 状态推进；与现有 step/session 语义保持一致 | 若训练 sequence 路径与推理 step 路径语义漂移，会破坏一致性 | 保留 step/session 的真实在线 SAM；单独为 `forward_seq()` 实现 array-backed / fused SAM sequence 版本，并做逐位置一致性回归 |
| P2 | 训练期地址支路异步化 / overlap | 不改变 `online_seq` 语义的训练加速方案 | 训练 step 中地址生成不再完全阻塞主干；能比较同步 / worker 前移 / next-batch overlap 三种模式 | 若重新退化成离线持久 precompute，会削弱在线主线的一致性 | 保持 `AddressEngine.forward_seq()` 为主线定义；优先尝试 CPU worker 临时预取本 step address，再尝试与 GPU 主干重叠计算 next batch address |
| P2 | 训练期 memory 范围控制与 bookmark/window 实验 | 不依赖 full doc prefix 的轻量在线训练配置 | 在较短 memory window 下维持大部分收益，同时显著降低地址构建规模 | window 过短会伤害匹配覆盖率 | 先支持固定 tail window，再实验 bookmark / anchor / canonicalization，比较 coverage / speed / loss |
| P2 | 文档级状态快照与 chunk 起点增量恢复 | 比“整文档地址全缓存”更省内存的训练执行模式 | 大数据集下无需为每个 sample 复制 `rosa_precomputed_*`，可通过起点快照 + 短 replay 恢复在线状态 | snapshot/restore 的正确性和 clone 成本需要严格验证 | 先做 chunk 起点 `RosaStateSnapshot` 缓存，再尝试“每 K token 一个 snapshot + 局部 replay”的折中方案 |
| P2 | ROSA × Engram 融合路线 | 文档 memory 与参数化 memory 共存的 hybrid 方案 | 同时支持 `external doc memory` 与 `learned memory table` 两条 value 分支，并可由 gate 融合 | 两类 memory 的优先级与冲突处理复杂 | 先实现 `doc memory branch + learned branch` 的双分支 payload；再研究共享 gate / branch-specific gate |
| P3 | 服务化 request 生命周期与混批 | 面向 serving 的 ROSA 运行时 | `RosaState`、prefetch、cache、bookmark 在并发请求下生命周期稳定 | 与现有推理框架集成难度高 | 先设计 request API、状态快照、回收与 fallback 策略 |

## 里程碑建议

### 里程碑 A：训练主线切换

1. `AddressEngine.forward_seq()` 跑通
2. `DocChunkDataset` 不再默认依赖 `rosa_precomputed_ids`
3. 在线训练前向可以稳定下降 loss
4. 与 reference 预计算路径做逐位置一致性对照

### 里程碑 B：统一训练/推理组件

1. 训练期和推理期都共用地址引擎、值路径、gate、注入层协议
2. prefill / decode 使用同一 `RosaState`
3. 旧 precompute 路径退为 regression / fallback

### 里程碑 C：更大 Memory 系统

1. `per-layer ValueStore` 成为主线实验对象
2. `ROSA-DocMemory` 接入外部检索文档作为 side memory
3. compression / memory value / sparse gather 逐步接入
4. prefetch / cache 在重 value backend 下体现真实收益
5. 训练阶段探索地址支路 worker 前移 / overlap，减少 `online_sam` 串行地址生成对 step time 的阻塞
6. 训练 sequence 路径切到高性能 SAM 实现，并逐步摆脱 `full doc prefix`
7. 用状态快照 / chunk 起点恢复替代“整文档地址全缓存”

### 里程碑 D：Hybrid Memory

1. 外部文档 memory 与 learned memory table 能共存
2. gate 可以协调外部 memory 与参数化 memory
3. 形成 ROSA × Engram 的混合 memory 架构

## 当前结论

- 旧 `doc_local/global` 方案：保留，但作为 reference/fallback，不再作为未来训练主线。
- 新主线：`teacher forcing 并行主干 + 在线地址 side-branch + 少层注入`。
- 中期扩展：在这条主线上接入 `ROSA-DocMemory`，使外部检索文档可以像 RAG 一样成为 side memory。
- 进一步扩展：参考 Engram，将 `external doc memory` 与 `learned memory table` 组合成 hybrid conditional memory。
- 训练优化方向：在不回退到旧离线持久表的前提下，探索训练期地址支路异步化 / overlap，把 `online_seq` 的定义和执行优化分离。
- 当前代码基础已经具备较大一部分骨架：
  - 地址抽象
  - value store
  - context gate
  - 显式注入层
  - prefetch/staging
  - hot cache
  - 在线 state 接口
- 下一步最关键的不是继续堆新 feature，而是把训练主路径从 `precompute` 切到 `forward_seq()`。
