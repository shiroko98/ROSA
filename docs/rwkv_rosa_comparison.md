# RWKV-7 + 原版 / 新版 ROSA 对照训练

> **精度更新：**正式 2.9B 对照默认使用 FP32。BF16 会在部分分段预测位置改变 top-1，不只是逐位不同；已修复 WKV autocast 边界，但不能保证整体 BF16 等价。具体证据见 [BF16 专项复查](rwkv_bf16_audit.md)。

## 1. 范围与来源

本分支 `codex/rwkv-rosa-compare` 新增 `train_rwkv_rosa_compare.py`。
不是将 Transformer 改名为 RWKV：实际主干来自
`TimeMobius/RWKV-LM-V7` 的 `new_sft` 分支，固定提交
`1d7e87bedffdac9e117bc3e8942dd79b0ee73fd1`。

复用上游用于 checkpoint 一致性测试的**可微、支持循环状态的 PyTorch RWKV-7
参考实现**，以及上游的 SFT 模板、最后一次 assistant 回复的 mask 构建、分块 head loss。
源码、Apache-2.0 许可证和改动清单在 `third_party/rwkv_lm_v7/`。
没有复用上游 Lightning/DeepSpeed Trainer，也没有导入其启动时会编译的 fused CUDA ops。
当前 WKV 后端是 `upstream_torch_reference`，因此**不能将耗时作为上游高性能内核的基准**。

原有 `train_qwen_llama_vs_rosa_v2.py` 和纯 ROSA 入口继续保留，不改变其训练语义。

## 2. 实验组定义

| 名称 | RWKV 主干 | ROSA 匹配 | 记忆取值 / 融合 |
| --- | --- | --- | --- |
| `baseline` | 原始 RWKV-7 | 无 | 无 |
| `original` | 同一 RWKV-7 | 原始精确 SAM，默认最小匹配长度 1 | 共享 `backbone.emb`，固定 scale 的残差注入，无上下文门控、无匹配长度加权 |
| `improved` | 同一 RWKV-7 | 与 original 完全相同的 SAM | 每个注入层独立 ValueStore，隐藏状态 / value / 匹配长度门控，可选稀疏更新、本地行分片 |
| `capacity`（可选） | 同一 RWKV-7 | 无 | 在相同位置加入可训练 residual MLP adapter，近似匹配 improved 增量参数量 |

**“原版”指原始非参数匹配规则 + 本项目明确规定的最小神经融合。**
原始 ROSA 本身没有需要梯度训练的参数，这里不声称残差公式是官方 RWKV+ROSA 模型。

注入发生在指定 RWKV block 的输入处。`--inject-layers 0,1` 才是两个独立值表；
默认 `0` 只注入一层。`capacity` 的参数预算可能有少量取整误差，会记录在结果中，
没有通过“挂不参与前向的参数”凑数。

### 系统优化与架构消融分开

SAM 算法、匹配阈值、注入层位和 scale 在 original / improved 间保持相同。
默认两组都允许地址异步准备，以免把 Python 地址开销误判为记忆架构差异。
若研究系统优化，应另开实验控制：

- `--no-address-async` 与 `--address-async`；
- `--sam-impl stateful` / `fast` / `compiled_cpu`。

新入口复用了在线地址引擎和双状态管理；**当前公平训练协议不会让 ROSA 私自读取输入窗口之外的完整文档前缀**。
原仓库的文档级 snapshot 训练优化不是本入口的独立训练开关；当前通过循环状态中的 ROSA snapshot 恢复回归验证状态接口。
不要把旧 Transformer snapshot 的加速数据写成这次 RWKV 的加速结果。

## 3. 公平性协议

- 各组从同一份严格加载的 `.pth` 初始化，或同一个 seeded toy state_dict 初始化。
- 权重加载不允许 silent missing keys。自动推断层数、维度、head size、低秩维度。
- 数据在**文档级**拆分，不先切 chunk 再随机拆分；显式 split 拒绝跨集合相同文档。
- 每个窗口由“至多 `history_tokens` 个左侧 token + 至多 `chunk_len` 个当前 token”组成。
- **RWKV baseline、original、improved 可见的 token 完全相同**，不引入训练集 global memory。
- 左侧历史只用于预热，loss mask 为 0；监督 mask 按 next-token 目标正确右移。
- 同一文档的监督 target 不因左侧重叠而重复计入训练 / 评测。
- 每个 epoch 的样本顺序由 `seed + epoch` 确定，各组一致；ROSA 地址只依赖当前位置及更早 token。
- 梯度累积按本次 update 的有效监督 token 总数加权，而非不等长 microbatch loss 的简单平均。
- 记录实际参数量与可训练参数量；默认对比不是同参数量，需用 `capacity` 补充控制。
- 正式实验应完整验证 / 测试，至少多个随机种子；冒烟配置不能证明泛化收益。

## 4. 环境

新建的独立环境名：`rosa-rwkv`，不修改原有 base / RWKV 环境。

```bash
bash scripts/setup_rwkv_rosa_env.sh
conda activate rosa-rwkv
```

本机安装：Python 3.11、PyTorch 2.7.1+cu126、NumPy 1.26.4、Jinja2 3.1.6。
可移植环境描述：`env/rwkv_rosa_environment.yml`；CUDA wheel 明确安装在 setup 脚本中。
CUDA 训练默认关闭 TF32 和 BF16 低精度部分归约；另在 WKV 递归内部明确关闭 autocast，保证该部分的实际矩阵运算为 FP32。二者不是同一个设置。
默认后端不需要 nvcc、DeepSpeed 或 Lightning。可选 C++ SAM 的编译依赖保持原仓库设置。

## 5. 离线小模型验证

```bash
bash scripts/run_rwkv_rosa_compare.sh configs/rwkv_rosa/toy.json
```

byte tokenizer、toy 初始化只用于离线测试，不代表真实 RWKV 预训练模型。
输出目录默认不覆盖；重复执行请换 `--out-dir`，有 checkpoint 时也可用 `--resume`。

## 6. 用户指定的 2.9B 模型

```text
/hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth
```

已实际读取：32 层、hidden size 2560、40 heads、head size 64、词表大小 65536，
总参数 `2,948,065,280`。低秩维度 decay/aaa/mv/gate = 96/96/64/320。

### 三组 GPU 冒烟

推荐 FP32 正确性模式：

```bash
bash scripts/run_rwkv_rosa_compare.sh configs/rwkv_rosa/rwkv7_2_9b_fp32_smoke.json
```

原 `rwkv7_2_9b_smoke.json` 保留为 BF16 性能/通路模式，不承诺分段预测等价。

配置使用 6 条公开合成 SFT 记录，4/1/1 文档拆分，每组 1 个 optimizer step。
`eval_batches=1` 表示仅作通路检查，配置默认不保存大 checkpoint。

单卡 48GB 环境下，采用 `train_scope=last_block`：三组共同训练最后一个 RWKV block、
输出 LayerNorm 和 head；其余主干冻结，improved 额外训练 ValueStore 和 gate。
这**不是 2.9B 全参数微调，也不是 LoRA**。FP32 模式使用 FP32 参数和计算；源 `.pth` 中原本已经保存为 BF16 的权重升为 FP32 并不能恢复丢失的权重位数，但避免了后续 BF16 运算舍入。
可选 BF16 模式冻结参数 BF16 存储、可训练参数使用 FP32 主副本并启用 autocast；WKV 递归明确在 FP32 精度边界内。

当前冒烟使用仓库已有 RWKV 词表，其中含 65530/65531/65532 的聊天特殊标记。
会检查 ID 范围、特殊标记和 shape，但**仅凭 `.pth` 不能证明词表语义与原始预训练完全一致**。
正式 SFT 需确认 tokenizer 和 chat template；冒烟 loss 很高不能用于宣称聊天能力已适配。

`--verify-online` 使用最多 64 个输入 token 额外做两类检查（不能代替扩大数值审计）：

- 临时将模型提升为 FP32，检查整段 vs prefill+decode 的状态语义一致性；
- 检查 ROSA scale=0 时与未改动上游主干的 logits 完全相同，并单独记录 BF16 分段数值误差。

FP32 检查完成后恢复参数存储 dtype；它会额外占用显存，不属于训练 step 峰值。
BF16 的矩阵形状变化不仅会带来不同舍入结果，在扩大的 2.9B 审计中确实观察到部分 top-1 翻转；不把 FP32 状态检查通过当成 BF16 生成等价。

### 真实数据上的训练

正式配置故意不提供训练数据默认值，必须显式指定，避免不知情地用 toy 语料训练：

```bash
bash scripts/run_rwkv_rosa_compare.sh configs/rwkv_rosa/rwkv7_2_9b_sft.json \
  --data /path/to/train.jsonl \
  --val-data /path/to/val.jsonl \
  --test-data /path/to/test.jsonl \
  --out-dir outputs/rwkv7_g1k_real_sft_seed42
```

支持 `.json` 记录列表、`.jsonl`、每行一个文档的 `.txt`。
SFT 记录使用上游 `messages` / `tools` 格式（不是 Hugging Face tokenizer）：

```json
{"messages":[{"role":"user","content":"记录：编号 A17，数值 603。请复述数值。"},{"role":"assistant","content":"603"}]}
```

`{"text":"..."}` 或 `.txt` 走全 token causal LM 监督。
不指定 val/test 时按文档自动拆分；正式实验优先显式拆分。

若需容量控制：增加 `--variants baseline,original,improved,capacity`。
若需同一设置的多个种子：分别指定 `--seed 42` / `43` / `44` 和不同输出目录。
`--train-scope head` 只训练共同输出头；`last_block --train-last-blocks N` 训练最后 N 个 block；
`full` 全量训练需要自行确认显存，不保证这张 48GB 卡承载 2.9B 全参数 AdamW。

## 7. 输出与恢复

- `manifest.json`：上游 commit、基础权重 SHA256、模型维度、代码 / 模板 / 词表 hash、
  split 文档 hash、超参数和相同历史范围协议。
- `comparison.json`：每组初始验证、最终验证、test loss/PPL/token accuracy、参数量、训练记录。
- `<variant>/train.jsonl`、`summary.json`：逐步指标和组内结果。
- `<variant>/last.pt`：可训练参数增量、优化器状态、epoch / batch 游标和 RNG。

checkpoint 是 **trainable delta**，不重复存储冻结的 2.9B 主干。
必须保留原始基础 `.pth`；不能当作独立完整 RWKV 权重交给原生推理器。
恢复时重建相同基础权重，校验数据、代码、tokenizer、训练范围及 optimizer 配置。

```bash
# 原来的其余参数不变；允许增大 epochs / max-steps
bash scripts/run_rwkv_rosa_compare.sh configs/rwkv_rosa/rwkv7_2_9b_sft.json \
  --data /path/to/train.jsonl --val-data /path/to/val.jsonl --test-data /path/to/test.jsonl \
  --out-dir outputs/rwkv7_g1k_real_sft_seed42 --resume --epochs 2
```

训练日志中的 step 秒数包含本次数据 / 地址等待与参数更新，不包含模型构建、验证、保存时间。
梯度裁剪计算包含密集参数和稀疏活跃行的统一全局范数，并在更新前拒绝非有限梯度。
不同 variant 的初次分配和 optimizer 状态初始化会影响第一步耗时，冒烟数字不能当吞吐 benchmark。

## 8. 当前边界

- 单进程 CPU / 单 GPU；不支持此入口的 torchrun/DDP/FSDP。原 Transformer 的 8 卡脚本不能直接用于 RWKV。
- PyTorch WKV 参考递归侧重正确性，不是高吞吐 fused CUDA 训练器。
- 数据集目前为内存构建的 SFT 窗口，不是原 Transformer memmap 数据接口。
- 值表是本地行分片；全部 shard 仍在模型所在设备，SparseAdam 状态也不等于全稀疏存储。
- 在线状态同时保存 RWKV recurrence 和 ROSA SAM；不支持在同一 stateful batch 中混入 padding。
- 上游 SFT mask 主要监督最后一次 assistant 回复；不宣称所有轮 assistant 都参与 loss。
- 尚无真实业务数据、多随机种子、容量匹配大实验，不能据冒烟结果评价新版优于原版。

初次历史记录见 `docs/rwkv_rosa_validation.md`；当前精度修复、扩大审计和训练重跑见 `docs/rwkv_bf16_audit.md`。
