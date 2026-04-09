# ROSA 规模化数据与模型路线图（TODO）

面向“大数据集 + 大模型 + 更重 memory backend”的专项路线图。目标不是替代在线主线，而是在保持 `online_seq` 主线定义不变的前提下，把数据、训练、参数与系统吞吐扩展到更大规模。

## 状态说明

- `已完成`：已经实现、验证并可在当前主线中复用
- `进行中`：已有第一版实现，但还未达到规模化可长期使用的形态
- `未开始`：尚未进入工程实现

## 当前优先级

1. 稀疏 / 分片 `ValueStore`
2. 大模型训练基础设施
3. 地址引擎 GPU/Triton/CUDA 路径
4. 分布式训练

## 主表

| 状态 | 优先级 | 任务 | 目标输出 | 完成标准 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 已完成 | P0 | 预分词 + `memmap`/二进制数据集管线 | 可复用的数据构建脚本、manifest、训练加载器 | 不再需要把全部 token/sample 常驻 Python list；`doc_local + online_seq` 可直接从预分词二进制数据训练 | 已新增构建脚本、manifest、训练入口 |
| 已完成 | P1 | 文档级索引与按需切片 | 文档偏移、长度、chunk 索引按需读取 | dataset `__getitem__` 动态切片，不复制大块 token | `MemmapDocChunkDataset` 已落地 |
| 进行中 | P1 | 稀疏 / 分片 `ValueStore` | 活跃项 gather + 可扩展表存储 | `per_layer` 不再线性吃完整词表乘层数的参数量 | 已完成 sparse active-row training v1 + 本地行分片 v1，跨卡分片仍未开始 |
| 已完成 | P1 | 大模型训练基础设施 | activation checkpointing / 更稳 checkpoint / 梯度累积策略 | 大模型训练不中断、可恢复、显存可控 | v1 已落地：逐层 activation checkpointing、epoch checkpoint/save-resume、grad accumulation、单模型 stage 运行 |
| 进行中 | P2 | 分布式训练 | FSDP/ZeRO 训练路径 | 单机多卡和更大模型训练可用 | DDP/FSDP v1 已落地，ZeRO 尚未开始 |
| 未开始 | P2 | 地址引擎 GPU/Triton/CUDA 实现 | `[B, T]` 训练 sequence addressing fused kernel | 训练期地址生成不再主要受 CPU 约束 | 优先 sequence 路径，不先改 decode step |
| 未开始 | P2 | 训练数据预取与拷贝流水线 | 数据读取、地址准备、Host->Device 拷贝分阶段 | 数据管线不再拖慢 GPU 利用率 | 与训练 timing 打通 |
| 未开始 | P2 | 状态快照磁盘化 / 轻量化 | 可落盘的 snapshot 索引与恢复 | 长文档下恢复更快、占用更低 | 仅在需要 full-history 时启用 |
| 未开始 | P3 | `ROSA-DocMemory` 规模化接入 | 外部文档 memory 构建与运行时注入 | 不仅能跑通，还能在大数据下稳定维护 memory source | 在数据底座稳定后推进 |
| 未开始 | P3 | `ROSA × Engram` hybrid memory | learned memory + external memory 双分支 | 参数化 memory 与文档 memory 共存并可控 | 依赖前面的表存储与稀疏化 |

## 本轮任务拆分

### 任务 A：预分词构建

- 输入：
  - 原始 `text/json/jsonl`
  - tokenizer
  - split 配置
- 输出：
  - `tokens.bin`
  - `offsets.npy`
  - `lengths.npy`
  - `manifest.json`
- 验证：
  - 文档数、token 总数、split 元数据可复现
  - 与现有 `tokenize_docs()` 结果逐文档一致
  - [x] 已完成第一版：`build_rosa_memmap_dataset.py` 可直接构建 `dataset_manifest.json`

### 任务 B：`memmap` 训练数据集

- 目标：
  - 不再把所有 sample 预展平成 Python list
  - 文档级存储，sample 级按需切片
- 验证：
  - `input_ids / labels / rosa_memory_ids` 与现有 `DocChunkDataset` 对齐
  - `doc_local + online_seq` 小样本训练可直接运行
  - [x] 已完成第一版：`MemmapDocChunkDataset` 按文档偏移动态切片，不再预展平 sample dict

### 任务 C：训练入口兼容

- 新管线必须与现有 raw 数据路径并存
- 允许：
  - 老路径继续跑回归
  - 新路径通过 manifest 显式启用
- 验证：
  - CLI 可切换
  - 报告中能明确打印当前数据来源
  - [x] 已完成第一版：`train_qwen_llama_vs_rosa_v2.py --pretokenized_manifest ...` 可直接训练

## 当前实现边界

- 当前 `memmap` 路径已支持：
  - `doc_local + online_seq`
  - `global_train + online_seq`
  - 原始数据离线预分词写入 `tokens.bin + offsets.npy + lengths.npy + dataset_manifest.json`
  - explicit split 下的流式构建
  - 可选多进程分词
- 当前 `memmap` 路径暂未接入：
  - `reference_precompute`
  - 训练地址缓存
  - 训练状态快照
- 当前优先保留“训练期下一批地址异步预取”作为规模化主线加速方式
- 当前 `ValueStore` 规模化路径已支持：
  - `per_layer + sparse value training`
  - 稀疏梯度 `Embedding(sparse=True)`
  - `AdamW + SparseAdam` 双优化器训练
  - 当前 batch 活跃地址统计：`rosa_active_address_count / fraction`
  - 本地按词表行分片的 `per_layer ValueStore`
  - 当前 batch 活跃分片统计：`rosa_active_value_shards`
- 当前 `ValueStore` 规模化路径尚未支持：
  - 跨卡 / 跨分片存储
  - 活跃项 all-to-all
  - 磁盘 / host memory 大表
  - 按分片独立放置到不同 device / host 的真正分布式表
- 当前大模型训练基础设施已支持：
  - `--activation_checkpointing`
  - `--grad_accum_steps`
  - `--save_every_epochs`
  - `--resume_from`
  - `--run_models baseline|rosa_fused|both`
- 当前分布式训练已支持：
  - `--distributed_strategy ddp|fsdp`
  - `DistributedSampler`
  - 分布式训练/评估指标归约
  - FSDP/DDP 下的 checkpoint/save-resume v1
  - 8 卡服务器启动脚本：
    - `scripts/launch_8gpu_ddp_online_v1.sh`
    - `scripts/launch_8gpu_ddp_online_v2_sparse_sharded.sh`
    - `scripts/launch_8gpu_fsdp_online_v1.sh`
- 当前分布式训练已知边界：
  - `FSDP` 首版暂不支持 `--rosa_sparse_value_training`
  - 当前“本地行分片 ValueStore”不会自动变成跨卡 all-to-all 表
  - `FSDP` 首版 checkpoint 仍是 full-state 保存，适合先跑通 8 卡，不是最终高效形态

## 当前结论

- 规模化阶段最先要解决的不是更多建模细节，而是数据与系统底座。
- 当前的 `DocChunkDataset` 更适合中小规模实验，不适合大数据长期训练。
- “预分词 + `memmap`/二进制数据集管线”第一版已经完成，下一优先级转向：
  - 分布式 / host-memory `ValueStore`
  - 地址引擎 GPU/Triton/CUDA 实现
  - 更正式的 ZeRO / host-memory 分布式 memory 表
