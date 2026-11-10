# ROSA-Tuning 论文与当前 RWKV-7 ROSA 的对照

对照对象是当前已训练和评测的 step1800 配置：RWKV-7 G1k 2.9B 冻结本体，单个 post_norm ROSA，output_norm=rms，logit_gap 动态增益，dynamic_min_match=6，dynamic_max=0.3，margin=0.5。仓库还支持其他注入配置，本报告不把它们当成当前检查点已经具备的能力。

论文为 [ROSA-Tuning: Enhancing Long-Context Modeling via Suffix Matching，arXiv:2602.02499v2](https://arxiv.org/html/2602.02499v2)。公开代码检查到提交 `ad5de0fb7d03d79c2506346af671c7296a1b44fc`，重点是 [`2025.12.29 qkv_update.py`](https://github.com/zyaaa-ux/ROSA-Tuning/blob/ad5de0fb7d03d79c2506346af671c7296a1b44fc/2025.12.29%20qkv_update.py)，增量生成另看 [`2025.11.10 generate.py`](https://github.com/zyaaa-ux/ROSA-Tuning/blob/ad5de0fb7d03d79c2506346af671c7296a1b44fc/2025.11.10%20generate.py)。公开文件是示例脚本，本次只作静态代码核对，没有执行外部脚本或复现论文训练。

## 最主要的区别

当前实现从原始 token 历史中寻找最长重复后缀，取匹配的后继 token ID，再读取按 token ID 索引的可训练值表。论文先把每层隐藏状态通过可训练 Q/K/V 投影变成二值符号流，在多路符号流中匹配 Q 与历史 K，得到历史位置，再读取该位置的 V 符号。后者能够学习检索所用的特征；前者的原始 token 匹配规则固定。

二者都采用确定性的后缀匹配，并基于历史后继信息提供额外残差，但检索对象、可训练部分和残差参与后续计算的位置不同。

|维度|当前 step1800|论文／对应多路公开代码|
|---|---|---|
|本体|RWKV-7 2.9B，递归状态|主实验 Qwen3-Base-1.7B，2048-token 滑窗 attention|
|寻址输入|原始 tokenizer token ID|每个注入层的隐藏状态，经 Wq/Wk 投影、按正负二值化|
|Q/K 关系|同一原始 token 流，寻找重复后缀|不同的可训练 Q/K 符号流，Q 查询 K 历史|
|路数|每条序列一套原始 token SAM，当前只有一个注入点|每注入层 R=d/M 路独立匹配；主实验 M=4，每路16种符号|
|返回／读取|后继 token ID，查固定于模型参数中的 E[token_id]|历史后继位置 tau，读取该时刻的上下文相关 V 二值符号|
|值的表示|词表大小×隐藏宽度的可训练 FP32 表|历史 V 比特，按每维 e0/e1 映射，再由 Wout 投影|
|注入位置|最终 ln_out 之后、LM head 之前；本步没有后续 RWKV block|多个中间层：post-attn 加性融合，或 pre-attn 混合；之后仍有 MLP／后续层|
|检索可学习性|不能学习 SAM 匹配键，学习值表、门控和投影|Q/K/V 投影可训练，检索路由有反事实／代理梯度|
|强度控制|hidden/value 门控、匹配长度项、min_match=6、logit_gap、cap=0.3|post-attn 直接加残差，pre-attn 用逐通道 alpha；没有当前这套 LM-head logit_gap 控制器|
|初始化|值表由本体 embedding 初始化；投影为单位阵；不保证初始残差严格为零|e0=e1=0；post-attn 注入初始为零；Wout 初始化为单位阵|
|训练|冻结本体，只更新 ROSA 的 SFT|主实验4B token adapter warm-up，26B token 全参数续训，再7B token SFT|
|执行|当前增量推理使用 CPU C++ SAM/LCT；分支 CPU/GPU 可选|CPU Numba 多路 SAM＋RLE，GPU Q/K/V 和融合，pinned buffers／copy stream／线程池进行重叠|

## 1. 当前 key_projs 不等于论文的检索 K

`rwkv_rosa/native/adapter.py:100` 先按已经确定的地址取 `value`，再算 `key_projs(value)` 与当前 hidden 的相似度，决定 gate。它影响取回内容的注入强度，不影响 SAM 找到哪个后缀、选择哪个候选。

当前检索地址由 `cpp_extensions/rosa_sam_cpu_extension.cpp:242` 附近决定，返回 `seq[endpos+1]` 这个 token ID。`gpu_state.py` 的 IncrementalAddress 把它包装成 addr_ids。原始 ID 的匹配路径没有可训练查询投影，也没有梯度穿过离散寻址。

论文代码的 `rosa_q_proj` 和 `rosa_k_proj` 则直接生成参与匹配的符号，改变这些投影有机会改变检索目的地。

## 2. 读取历史位置的值，与读取 token 值表不同

当前即便匹配来自不同历史位置，只要后继 token ID 一样，值表读出的向量就一样。上下文对输出的影响还来自本体 hidden 和门控，但值表本身不保留该 token 在那次历史出现时的上下文表示。

论文每路可读不同的历史位置，V 是该位置隐藏状态经过投影及二值化的结果。需要强调：它读取的是量化 V 特征，而非完整、无压缩的历史 hidden，也非直接从原文取答案字符串。二值化带来碰撞和信息压缩，不能自动等同于语义检索或零错误召回。

公开代码可核对：`2025.12.29 qkv_update.py:838` 生成 q/k/v，`:908` 附近按 dest_time 从 v_cat gather，之后用 e0/e1 和 rosa_out 构造注入量。

## 3. 注入到中间层和最终 head，表达能力与计算成本不同

当前 `third_party/rwkv_lm_v7_native/src/model.py:944` 先做 ln_out，再调用 inject_post_norm，随后进入输出头。ROSA 可以改变当前 token 的 logits；这一步的残差不进入后续 RWKV block，也不直接写进该步已经算出的本体递归状态。它会通过改变生成 token，间接影响后续状态。

论文 post-attn 在当前层 attention 之后、MLP 之前注入；公开多路脚本跳过第0层，其余 decoder layers 安装各自的模块。注入信息可被当前层 MLP 和后续层继续加工。论文 pre-attn 还让 attention 直接处理混合输入。这种深层融合的能力更强，同时需要更多参数、历史符号缓存、CPU路由和梯度工程。

## 4. 反向传播不同

当前值表、投影、RMSNorm 和 gate 走普通自动微分；地址来自整数匹配，不参与学习；logit_gap 控制器在 no_grad 下计算。训练不会让“原本不相同的两段 token”变成 SAM 的匹配。

论文公开代码 `MultiRouteBinaryROSAFunction` 定义自定义 backward。V 按检索目的地聚合梯度，Q 利用逐 bit 置0/置1后的反事实读出差异，K 使用 run 级代理与聚合。论文也明确忽略部分 run 边界高阶效应，因此应理解为为离散路由构造可训练信号，而不是对整个离散检索求得常规意义上的精确导数。

## 5. 参数量应按总注入层数计算

当前 Vocab=65,536，d=2,560，单注入点参数量：

```
Vocab*d + 2*d*d + 2*d + 2 = 180,884,482
```

论文对应代码每个注入层有 Wq/Wk/Wv/Wout 四个 d×d 无偏置矩阵，e0/e1 两个 d 向量，以及注册的 d 维 alpha：

```
每层 4*d*d + 3*d
```

按同样 d=2,560 推算，每层26,222,080个参数；若仿照公开代码在32层本体中跳过第0层、加到其余31层，总新增812,884,480个参数，约8.13亿。这是同宽度移植的静态估算，不是论文报告的 Qwen 模型实测参数量；post-attn 中 alpha 还不参与实际融合。它消除了大词表值表，但多层合计不一定比当前1.81亿更小。

M=4 是每个符号由4个 bits组成，不是“匹配至少4个token”。以d=2,560为例，R=640；论文的 M 与当前 dynamic_min_match=6 不是同一个超参数，不能直接替换。

## 6. 速度与内存

论文把 CPU 工作按 batch×route 并行，并采用相邻符号 RLE、uint8 传输、pinned memory、异步 stream 和线程池；post-attn 可与 GPU attention 重叠。当前仅一条原始 token 地址流，使用 C++/LCT维护最近出现位置，增量解码不重建全部历史；本文所对照的推理路径没有论文那种多层学习路由流水线。

双方都把随历史增长的检索状态保存在CPU，均不能把“单步开销可控”解释为无限历史的总内存为O(1)。论文多层多路的状态与缓存常数也需要实际测量。论文效率图比较单核单SAM与小型FlashAttention核；它不能替代完整模型在目标机器上的端到端测速。我们已有的H800／Xeon测速也不能直接作为论文模块速度。

## 7. 对当前检索结果的解释与边界

当前短口令测试在128K的自然问答为ROSA13/15、精确前缀补全15/15；256K为0/15和6/15。精确前缀直接提供可被原始 token 后缀识别的线索，自然问题却可能没有足够长的相同后缀；该结构差异与测试现象相容。

论文的学习特征路由、多层融合和全参数续训提供了另外的处理方式，但没有证据说明任意一项单独移植就能解决当前256K问题。论文的NIAH表格为32K设置，主干、训练数据、预算和提示都不同；论文100%分数不能与本仓库256K结果直接排优劣。

若研究这一方向，最应区分的改动是：(1)检索键从原始token变为可训练符号；(2)读取上下文相关历史V；(3)从最终head前单点注入变为少量中间层注入。每项都需要重新训练和独立消融，不应把现有step1800权重直接解释成论文模块。

## 公开代码的版本区别

`Native ROSA-Tuning.py` 是早期单分类符号流、Embedding和STE式写法，不代表论文最终的QKV二值多路方案。`2025.12.29 qkv_update.py` 才包含上面核对的四投影、位符号、自定义反事实梯度和流水线；`2025.11.10 generate.py` 另有持久化多路SAM和单token增量生成状态。论文训练规模4B+26B+7B与公开示例脚本里可见的阶段预算占位／默认值也不应混同。结论以论文v2及上述具体文件为准。
