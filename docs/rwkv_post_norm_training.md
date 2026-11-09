# RWKV-7 post_norm ROSA 实验（2026-11-09）

## 位置及公平比较

显式接口：

```bash
--rosa_position block --rosa_layers 31
--rosa_position post_norm
```

post_norm 不对应某个特殊 block 层号，元数据为 position=post_norm、layers=""。
所有 frozen RWKV blocks -> frozen ln_out -> improved ROSA residual -> frozen head。
沿用原值表、门控、匹配长度特征、投影、scale=.05、embedding 值表初始化和
单位投影初始化，**不增加输出概率 copy 分支，也不声称初始化就能正确解码**。
只改变注入位置，从头训练新的 ROSA，不恢复/复用 block 模式优化器和权重。
单位置 trainable 参数同为 180,884,482。本体 2,948,065,280 参数全部冻结。

CPU FP32 下显式分解 W_head(z+delta) 为 W_head(z)+W_head(delta)，数值舍入可使
逐位相等不成立；native BF16 下仍保持原精度策略，不以分解为逐位等价承诺。
不把“无反向经过 RWKV blocks”说成省掉前向：每一步仍计算完整本体前向。
对固定输入 token 的 recurrent cache 不受注入改变；自由生成 token 改变后
后续输入及其状态当然可以不同。

## 一致支持的路径

- Native upstream `new_sft` 的 `_forward_features` 在 ln_out 后调用独立
  `inject_post_norm`；block 模式这个调用是 no-op。
- CPUInference 同一位置，覆盖流式状态、生成、checkpoint 转换和离线评测。
- 每次 CPU NIAH 请求/结果都明确记 position。adapter tensor shape 相同不代表
  可跨位置 resume：训练 checkpoint 严格检查 position、layers 及数据策略。
- 旧 checkpoint 缺少 position 按历史 block 解释；不能因此恢复成 post_norm。
  缺少整个 native_rosa 元数据拒绝恢复，不盲猜位置。

## 当前试训

Launcher: `/mnt/data/Codes/ROSA/scripts/run_rwkv_3b_postnorm_training.sh`。
输出：`/hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx131072_20261109`。

Base: `/hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth`。
本体 CUDA BF16，ROSA CUDA FP32，SAM compiled CPU / suffix_link_lct_v1。
128K=131072 输入，batch1、accum1；不再测更长长度。
原 192512 数据集物理 split / 两个虚拟窗口覆盖全部 targets，split seed42、
同样 data order；419032 更新/单遍，具体政策见 rwkv_3b_lastlayer_training.md。
LR3e-5->3e-6、warmup200、最后10% cosine、betas .9/.99、eps1e-8、dense decay.01、
clip1、fused CE chunk1024、scale.05 均与 block-31 预设一致。

监控调度的明确差别：本次不在 step0 等待漫长 CPU NIAH，先进行实际更新和
梯度/冻结审计。常规每200步 SFT eval（4个 held-out windows），同步 CPU 8K
NIAH（3 depths ×2 difficulties ×2 prompt modes；4 threads）。CPU monitor 超时
上限4小时，失败会明确报错而非伪造/跳过结果。此监控不是128K质量评测。
旧 block-31 run 在初始 CPU NIAH 期间停止，0个 optimizer updates，未形成
可比较的训练后质量结果；不能据此声称 post_norm 比 block-31 更好。

首个完成更新保存 resumable adapter+optimizer+loop checkpoint，之后约每小时
保存到 checkpoints/，native-last.ckpt 指向最近完整文件。复现/恢复仍需原base。
W&B online，独立run，保留 train.log/native_steps.jsonl/native_audit.json 和
每次 needle 原始回答/诊断；源码和测试在生产输出 source_snapshot 中留档。

## 直达输出诊断

每个 query 的首个目标 token：post_norm_target_logit_before/delta/after、
post_norm_target_rank_before/after；有有效候选时记录候选 logit delta 和排名。
同时记录实际最终 logits 与线性分解的最大绝对误差。
聚合目标 logit 增量，以及正确候选/错误候选子集各自的目标 top1 前后比例。
这些是 query 首 token 的诊断，不把它当成完整口令生成准确率。

## 验证

新增 tests/test_rwkv_post_norm.py：
- 无魔法层号/层位置校验、与单 block 同规模同初始化；
- CPU logits 分解、state 完全不变、流式/全量一致、生成可用；
- 从真实 native 源码提取 forward，检查所有 block/norm 均不建反向图，
  无 checkpoint 调用，只有 ROSA 获得梯度；
- position 元数据转换/导出/重载及跨位置 resume 拒绝；
- NIAH 输出 logit、候选正确/错误分组诊断。

实际 GPU 128K 更新、冻结 hash、保存文件和速度以 native_audit.json/
native_steps.jsonl 为准，不把 toy 单测当真实3B实测。

## 已完成的实测（17:36，Asia/Shanghai）

- 观察到至少9次真实128K optimizer更新，无OOM/非有限数值；前9步中排除首步
  的普通更新耗时中位数8.96s。反向约0.21s，SAM完整寻址约0.25–0.56s。
- 已分配显存峰值22.65GiB；这是当前样本观察值，不是无限长稳定性保证。
- 冻结本体初始/首更新哈希相同，body无梯度，ROSA值表确实变化，优化器状态CUDA。
- step1 compact checkpoint 2,170,658,487 bytes，明确保存position=post_norm及
  layers=""；包含dense+sparse两个优化器和loop状态，引用base哈希及1062个参数名。
- 首次网络盘保存151.67s，随后恢复正常更新。9秒/步不包括保存与CPU NIAH暂停。
- W&B在线run ID e145574e。每200步首轮SFT+NIAH尚未完成，不能声称质量提升。
- 最终CPU全套测试：213项，OK，3项CUDA专用跳过；额外post_norm测试6项均通过。
  原生vendor43文件哈希核对通过，UPSTREAM_PATCH.diff在原始文件上apply --check通过。

## 后续：正式任务切换到192K

用户明确要求以完整192512输入继续正式训练，不再恢复旧128K任务。
旧任务停留step30，checkpoint和失败恢复日志均保留。新任务从相同初始化独立
启动，不沿用旧上下文的循环状态。最新入口与配置见
`/mnt/data/Codes/ROSA/docs/rwkv_3b_postnorm_192k_training.md`。
