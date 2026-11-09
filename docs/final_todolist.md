# ROSA Final Todolist

将当前仓库里的 TODO 合并为一份统一清单，同时保留“已完成追溯”和“未完成主表”两部分，后续默认以这份文件作为总入口。

## 合并范围

- 根目录：
  - `ROSA_在线主线重构_TODO.md`
  - `ROSA_规模化数据与模型_TODO.md`
- `文档/` 目录：
  - `ROSA_统一路线图_TODO合并版.docx`
  - `ROSA_PLE_Engram_分析与改良方案_v2_工程版.docx`

## 合并原则

- 已完成事项也保留，但单独放进“已完成追溯表”，用于回看演进路径。
- 对语义重复的事项做合并，但会把“已经做到哪一步”写清楚，避免看起来像完全没做。
- 以当前代码、`progress.md`、测试与最近提交为准修正状态，不机械照抄旧 TODO。
- 优先保留工程上真正还会影响下一步开发排期的任务，不把纯历史里程碑再抄一遍。

## 状态说明

- `已完成`：已经实现、验证并纳入当前主线、对照路径或基础设施。
- `进行中`：已经有第一版实现，或者已经有部分基础设施，但离可长期作为主线使用还差关键收尾。
- `未开始`：目前仓库里还没有真正进入工程实现，最多只有文档设计或零散想法。

## 当前已具备的基础

- 在线主线骨架已具备：
  - `OnlineRosaState`
  - `RosaAddressEngine`
  - `RosaBatchSession`
  - `online_seq`
  - `online_sam`
- 地址实现已具备三条路径：
  - `stateful`
  - `fast`
  - `compiled_cpu`
- value/gate/runtime 已有基础版本：
  - `shared/per_layer ValueStore`
  - `context-aware gate`
  - `prefetch/staging/hot cache`
  - 层位 sweep
- 训练与数据底座已有第一版：
  - `memmap` 预分词数据集
  - sparse value training
  - 本地行分片 `ValueStore`
  - snapshot + 短 replay
  - 地址异步预取
- 规模化训练已有第一版：
  - activation checkpointing
  - grad accumulation
  - DDP/FSDP v1
  - 8 卡服务器脚本
  - `wandb` 监控
  - 容量匹配 baseline 对照入口

## 当前统计

- `已完成`：18 项
- `进行中`：6 项
- `未开始`：10 项

## 优先级建议

1. 先把训练主线的“吞吐瓶颈”和“大表扩展能力”补齐。
2. 再把外部 memory、memory value 和 memory window 这三类真正影响能力上限的项接进来。
3. 服务化 runtime、hybrid memory、latent code、spec decode 放到主线稳定后再集中推进。

## 已完成追溯表

| 状态 | 优先级 | 任务 | 目标输出 | 完成标准 | 主要风险 | 工程上的细分实现 |
| --- | --- | --- | --- | --- | --- | --- |
| 已完成 | P0 | 把 ROSA 从离线/整段检索改成增量在线状态机 | 可在 `prefill + decode` 中复用的在线状态接口 | `OnlineRosaState` 已落地，支持 `reset/prefill/update_one/snapshot`，并与 reference 逐 token 对齐 | 若只保留原始 list-state，会限制 decode 场景演进 | 已完成 `OnlineRosaState`、`RosaStateSnapshot`、逐 token 对齐测试和 session 复用基础 |
| 已完成 | P0 | 抽象地址生成接口，解耦“匹配”和“取值” | 统一的 `AddressMeta` / `RosaAddressEngine` | 地址生成不再直接绑定 token embedding；训练和推理都可消费统一 `addr/raw_match/fired_match/valid_mask` | 地址和值耦合过深会卡死后续扩展 | 已完成 `AddressMeta`、`RosaAddressBatch`、`RosaAddressEngine`、`rosa_addressing.py` 拆分 |
| 已完成 | P0 | 搭建最小在线注入闭环 | 一版能工作的在线 ROSA 注入路径 | decode 场景下已能按步生成地址并做单早层残差注入；功能和 logits 对齐可回归 | 一开始如果引入过重 value backend，排障会很困难 | 已完成 `forward_online()`、最小 online injection、共享 embedding value 路径 |
| 已完成 | P0 | 建立性能与正确性基线 | profiling 脚本、回归测试和对照报告 | 已能输出 prefill/decode 地址一致性、logit diff、coverage 和基线对比 | 没有统一口径时，后续优化容易变成口头感觉 | 已完成 `profile_rosa_online_baseline.py`、基线报告、相关单测 |
| 已完成 | P1 | 引入 `per-layer ValueStore` v1 | 每层独立的 value table lookup | `shared/per_layer` 两种模式都已可用，`per_layer` 可用于训练、profile、recipe | 参数量增长明显，需要后续规模化支持 | 已完成 `RosaValueStore` 和 `online_v2` recipe 基础 |
| 已完成 | P1 | 引入 context-aware gate v1 | `Gate(h_t, value_t, match_len)` 与指标统计 | 当前已能输出 `avg_gate/gate_coverage/gate_hit`，并参与前向融合 | 门控过强或过弱都会伤害稳定性 | 已完成 context gate 前向与统计落地 |
| 已完成 | P1 | 把在线调度改成“地址先算、值后取” | 清晰的地址/值/runtime 分层 | 主干层不再直接耦合检索逻辑；地址批、payload、层消费路径已分层 | 地址生成和层执行混在一起会拖慢后续优化 | 已完成 `compute_rosa_address_batch()`、`build_rosa_injection_payload()`、`forward_hidden(..., rosa_payload=...)` |
| 已完成 | P1 | 异步预取、staging buffer 与热点缓存 v1 | `Prefetcher + StagingBuffer + HotCache` 基础能力 | decode 和 profile 路径已经可以观测 hit/miss/wait/cache 统计，不破坏语义正确性 | value backend 偏轻时，收益不一定立刻明显 | 已完成 `RosaPrefetcher`、`RosaStagingBuffer`、`RosaHotAddressCache` 和 profile 指标 |
| 已完成 | P1 | 训练/推理统一层位 sweep 协议 | 可复用的注入层实验入口 | train/profile/both 三种扫描模式都已统一输出报告 | 如果没有统一协议，层位结论很难复现 | 已完成 `rosa_layer_sweep.py`、`scan_rosa_injection_layers.py` 薄封装和报告聚合 |
| 已完成 | P0 | 训练主线切到 `online_seq` 并建立一致性回归 | 不再默认依赖 `rosa_precomputed_ids` 的训练主入口 | `forward_seq()` 已进入训练主线，online vs reference 已有逐位置一致性 smoke | 新旧训练路径分叉会让维护成本快速上升 | 已完成 `--rosa_train_mode online_seq|reference_precompute`、训练路径一致性报告 |
| 已完成 | P1 | 统一推理主形态：prefill + decode 共用 session 生命周期 | 稳定的 `RosaBatchSession` | prefill/decode 现已共用 session API，支持 snapshot/reset/prefetch stats | 生命周期管理不统一时，serving 很难推进 | 已完成 `rosa_session.py`、`init_online_session()`、`prefill_seq()`、`decode_step()` |
| 已完成 | P2 | `online_sam` sequence 快路径 v1 | 比纯 `stateful` 更快的训练期 sequence addressing | 已完成 `fast` 路径，并有一致性回归；sequence 层已有明显加速 | 若 sequence 实现和 decode 语义漂移，后果很重 | 已完成 `--rosa_online_sam_impl fast|stateful` 和 fast/stateful 对照 |
| 已完成 | P2 | `online_sam` 编译型 CPU 实现 v1 | 编译后的 sequence 地址实现 | 已完成 `compiled_cpu` 扩展，并接入训练/profile 路径 | 平台兼容和部署链路需要额外照顾 | 已完成 `cpp_extensions/rosa_sam_cpu_extension.cpp`、构建脚本与单测 |
| 已完成 | P2 | 训练期 `snapshot + 短 replay` v1 | 文档级状态快照与 chunk 起点恢复 | 已支持稀疏 snapshot、起点 replay 和 interval sweep | 如果没有 snapshot，长前缀训练成本过高 | 已完成 `rosa_training_snapshot.py`、恢复入口和 sweep 脚本 |
| 已完成 | P0 | 预分词 + `memmap`/二进制数据集管线 v1 | 不依赖 Python 大 list 的数据底座 | 已支持 manifest 构建、按需切片 dataset、直接从 `--pretokenized_manifest` 训练 | 大数据下若仍依赖 Python list，训练会很吃内存 | 已完成 `build_rosa_memmap_dataset.py`、`rosa_memmap_builder.py`、`rosa_memmap_dataset.py` |
| 已完成 | P1 | 稀疏活跃项 `ValueStore` 训练 v1 | 活跃地址驱动的 sparse value 优化路径 | 已支持 `Embedding(sparse=True)`、`AdamW + SparseAdam` 双优化器和活跃地址统计 | 与分布式/分片结合后复杂度会继续上升 | 已完成 `rosa_value_store.py`、`rosa_optim.py`、`online_v2_sparse` |
| 已完成 | P1 | 本地行分片 `ValueStore` v1 | 单机按词表行切分的 value table | 已支持本地 shard、active shard 统计和 `online_v2_sparse_sharded` 配方 | 这还不是跨卡 all-to-all 大表，只是单机第一步 | 已完成 `--rosa_value_shards`、分片 lookup 和相关测试 |
| 已完成 | P1 | 大模型训练基础设施 v1 | 大模型训练可恢复、可监控、可跑服务器脚本 | activation checkpointing、grad accumulation、save/resume、`wandb`、8 卡脚本已落地 | 如果基础设施不先打底，后面的规模化任务很难推进 | 已完成 `rosa_checkpointing.py`、`rosa_distributed.py`、`rosa_wandb.py`、服务器脚本与环境文件 |
| 已完成 | P2 | 容量匹配 baseline 对照入口 | 与 ROSA 表达能力更接近的 baseline 比较方式 | baseline 已能按 ROSA 增量参数量自动估算 adapter 宽度，并进入服务器脚本 | 如果 baseline 太弱，后续对照意义不大 | 已完成 `--baseline_capacity_match_rosa`、`CapacityMatchedBaseLM` 和对应 launch 脚本 |

## 未完成主表

| 状态 | 优先级 | 任务 | 目标输出 | 完成标准 | 主要风险 | 工程上的细分实现 |
| --- | --- | --- | --- | --- | --- | --- |
| 进行中 | P2 | 在线训练 V2 主线化：`per-layer ValueStore` 成为稳定实验主线 | 一套可重复、可对照、可在本地和服务器统一复现的 `online_v2` 主线配置 | `online_v2` 不再只是“能开开关跑通”，而是能稳定复现实验；train/profile/sweep/8GPU 脚本都能统一使用；与 capacity-matched baseline 能做公平对照 | 参数量、显存和主存开销明显上升；如果只做小样本 smoke，结论容易失真 | 补齐 `online_v2` 与 `online_v2_sparse(_sharded)` 的实验矩阵；固定推荐默认层位；把 equal-param baseline 纳入对照；把关键结果沉淀到统一 summary/report |
| 进行中 | P2 | 稀疏 / 分片 `ValueStore` 走向跨卡 / host-memory 版本 | 不再被“词表大小 × 注入层数”线性卡死的 memory table 基础设施 | 不只支持本地 active-row/single-node shard，而是支持跨卡或 host-memory 大表；前向/反向只搬运活跃项；训练能稳定跑多卡 | 路由、通信、优化器状态和容错都更复杂；不小心会把简洁主线拖成“大表工程” | 在现有 sparse/local shard 基础上补：全局 shard 元数据、active row 路由、all-to-all 或 gather/scatter、host DRAM / mmap 后端、fallback 本地路径、活跃分片与通信统计 |
| 进行中 | P2 | 训练期地址支路异步化 v2/v3 | 更稳定的 `online_seq` 地址 overlap 训练路径 | 不再只有“能异步”这一层，而是能比较 thread/process、支持按 batch token 数调度、并把 queue wait / prepare wait / copy wait 分开记时；在更长上下文和更重配置下仍稳定 | 容易把逻辑重新做回“离线预计算”；异步链路一复杂就容易出现状态错位或回归难排查 | 比较 thread/process worker；支持 token-aware prefetch 调度；把地址构建、batch 准备、拷贝等待拆阶段统计；保留同步模式作回归基线；继续验证 next-batch overlap 在大配置下的稳定性 |
| 进行中 | P2 | `online_sam` sequence 高性能后端继续下沉到 CUDA/Triton | 训练期 `[B, T]` sequence addressing 的 fused 高性能实现 | 训练期地址生成不再主要受 CPU 和 Python 限制；CUDA/Triton 路径与 `stateful` decode 逐位置一致；性能明显优于 `fast` 和 `compiled_cpu` | 训练 sequence 路径和推理 step 路径一旦语义飘了，后面很难维护； fused kernel 调试成本高 | 保留 `stateful` 作为语义 reference；针对 `[B, T]` sequence 实现 fused kernel；补 helper/address/logit 三级一致性回归；把 profile/train/sweep 都接到新后端 |
| 进行中 | P2 | 状态快照轻量化 / 磁盘化 / 与 `memmap` 联动 | 真正适合大数据集的 snapshot 恢复链路 | 不只支持内存内 sparse snapshot + replay，还能支持更轻量序列化、磁盘索引、与 `memmap` 数据集协作；长文档训练时恢复成本明显下降 | snapshot 一旦做复杂，最容易出现在边界位置恢复错误；过度序列化也可能抵消收益 | 设计可落盘的 snapshot 索引格式；减少 snapshot clone/restore 成本；支持更细粒度 interval；把 `memmap` 数据集、异步地址预取和 snapshot 恢复接到同一路径；补大文档回归测试 |
| 进行中 | P2 | 分布式训练 v2：ZeRO / 更高效 FSDP / sparse value 兼容 | 面向更大模型和更大表的多卡训练主线 | 不只停在 DDP/FSDP v1；需要支持 ZeRO 或更高效 state-dict；`sparse ValueStore` 与分布式训练能兼容；8 卡训练更稳、更省内存 | 分布式 checkpoint、sparse optimizer、分片参数和恢复逻辑组合起来容易失控 | 评估 ZeRO 路线；优化 FSDP checkpoint/state-dict；补齐 sparse value + distributed 的兼容方案；把 8GPU 脚本和 resume 流程统一；补更正式的分布式 smoke 和故障恢复验证 |
| 未开始 | P2 | 训练数据预取与 Host->Device 拷贝流水线 | 数据读取、地址准备、H2D 拷贝的分阶段流水线 | GPU 训练不再被 dataloader、地址准备或拷贝阶段串行拖慢； timing 能清楚区分数据等待和搬运等待 | 如果把数据、地址和拷贝交织得太深，会让问题定位更困难 | 拆分数据读取、分词/切片、地址准备、H2D 拷贝阶段；尝试 pinned host buffer；和 `train_timing` 统一口径；在服务器脚本上给出中档和重负载默认值 |
| 未开始 | P2 | `ROSA-DocMemory`：接入外部文档 memory | 类 RAG 的 ROSA side memory 路径 | 检索到的外部文档可以被转成 `optional_memory` 或独立 memory branch，prefill/decode 都能稳定使用；不污染主线在线状态定义 | 文档排序、截断、拼接和污染控制都容易影响命中质量；外部 memory 不稳定时可能伤害生成 | 先支持 `retrieved_docs -> token stream memory`；再做文档排序、截断和去噪；区分本地历史 memory 与外部文档 memory；补外部 memory 开/关对照实验 |
| 未开始 | P2 | 训练期 memory 范围控制：tail window / dynamic window / bookmark | 不依赖 `full doc prefix` 的更轻量在线训练配置 | 在较短 memory window 下维持大部分收益，同时显著降低地址构建和恢复成本；coverage / loss / speed 三项都有明确对照 | window 过短会直接伤害长匹配；动态 window 规则一复杂就难复现 | 先做固定 tail window；再尝试 dynamic window、bookmark 触发、anchor 恢复；把 coverage-loss-speed 一起写入报告；和 snapshot 路径一起评估 |
| 未开始 | P2 | tokenizer compression / canonicalization | 规范化后的离散 token 流地址引擎 | ROSA 可以在压缩/规范化 token 流上运行，并与原始 token 流做命中率、误命中、效果对照 | 压缩策略不当会破坏语义边界；数据预处理和线上推理不一致会带来新问题 | 先做轻量 canonicalization：大小写、空白、等价形式归一；建立原 token / 规范化 token 双路径对照；再尝试更激进的压缩流实验 |
| 未开始 | P2 | value 从 token value 升级为 memory value | 更正式、更有载荷的 memory 表达 | 注入值不再只是“历史 next token 的 embedding”，而是适合层内融合的 memory payload；能做同参数量对照 | 若训练信号不稳，可能变成“gate 在学、value 没学起来”；过早做复杂表征会拖慢主线收敛 | 支持 token-id value、learned memory table、hybrid value 三条路线；补同参数量基线；逐步尝试低秩、分块、量化等更重表达 |
| 未开始 | P2 | 轻量 memory fusion 后处理：`DWConv` / tiny MLP / residual adapter | 更稳的 memory 注入分支 | memory value 在注入前经过轻量加工后，收益更稳定，但端到端延迟仍可控 | 分支稍微做重一点，就可能把“低计算记忆”的优势吃掉 | 先做短核 depthwise causal conv；再试 tiny MLP / adapter；统一挂到 injection hook 之后；和纯 residual add 做同参数量、同时延对照 |
| 未开始 | P3 | 服务化 runtime：request 生命周期、混批、缓存回收 | 可对接真实 serving 场景的 Online ROSA Runtime | `prefill/decode` 双态、state 管理、cache 生命周期、混批和回收策略都稳定；能在真实服务负载中跑通 | 研究代码和服务代码的执行语义最容易在混批与回收处不一致 | 抽象 state manager；拆出 request API；设计 snapshot/clone/release；补并发一致性测试；逐步替换为 fused/custom op 或对接 vLLM / 自研推理栈 |
| 未开始 | P3 | `ROSA × Engram` hybrid memory | 外部文档 memory 与 learned memory table 共存的双分支方案 | 同时支持 `external doc memory` 和 `learned memory table`，并能由 gate 协调两类 memory 的使用 | 两类 memory 可能互相打架，优先级和冲突处理不好会直接拉低稳定性 | 先实现双分支 payload；再研究 shared gate / branch-specific gate；补 memory source 统计和冲突分析；和单分支版本做清晰 ablation |
| 未开始 | P3 | Latent-code ROSA | 从原始 token 迁移到内部离散码流的 ROSA | ROSA 不再永远绑定 tokenizer id，而能在模型内部离散 code 上做匹配，并证明收益高于原始 token 版 | 离散码学习、采样和 STE 都不稳定；研究成本高且不适合直接压到主线 | 先做小 codebook；只在少数层试验；从“code 生成稳定性”开始，再对比 token ROSA 与 latent ROSA 的命中率和效果 |
| 未开始 | P3 | Bookmark token / conversation anchors | 面向长对话的显式历史锚点机制 | 长对话和检索式聊天场景下，历史定位质量明显提升，并能与 ROSA 一起工作 | 锚点设计不好会污染生成分布；容易变成只对某些模板有效的特化技巧 | 先定义 bookmark token 或 anchor 语义；限制触发模式；先在对话/多轮历史数据做专项实验；再决定是否并入主线 |
| 未开始 | P3 | Speculative decode / retrieval-assisted decode | 结合 ROSA 的实测解码加速路线 | 不只是提高建模能力，而是在高重复模式任务上获得真实速度收益 | 命中率不够高时会只增加复杂度；服务化实现门槛也高 | 先离线统计可推测命中率；再做 lightweight candidate path；最后再和 serving runtime 结合，比较 wall-clock 真收益 |

## 建议执行顺序

### 第一阶段：先把训练主线做扎实

1. 在线训练 V2 主线化
2. 训练期地址支路异步化 v2/v3
3. `online_sam` CUDA/Triton 路径
4. 分布式训练 v2
5. 跨卡 / host-memory `ValueStore`

### 第二阶段：再补能力上限

1. `ROSA-DocMemory`
2. memory window / bookmark 控制
3. tokenizer compression / canonicalization
4. memory value 升级
5. 轻量 fusion 后处理

### 第三阶段：最后做服务化和研究项

1. 服务化 runtime
2. `ROSA × Engram` hybrid memory
3. Latent-code ROSA
4. conversation anchors
5. speculative decode

## 备注

- 后续如果 `progress.md`、实验报告和这份文件状态不一致，以“代码 + 最近实验结果”为准回写这份文件。
- 若新增专项路线，优先在这份文件里补一条，而不是再单开一份并行 TODO，避免状态再次分叉。
