# 3B + post_norm ROSA：192512-token 正式训练（2026-11-09）

用户在192K容量测试通过后明确要求：不要恢复128K，改用192K正式训练。
原128K保留到step30及原日志、checkpoint；不把旧数据循环/优化器状态强行
解释成192K。新任务独立目录、独立W&B，从同一原base和ROSA初始化开始。

## 配置

Launcher: `/mnt/data/Codes/ROSA/scripts/run_rwkv_3b_postnorm_192k_training.sh`
输出: `/hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109`
Base: `/hikscale/models/RWKV/rwkv7-g1k-2.9b-20260930-ctx25600.pth`
数据: `/hikscale/datasets/SFT_RWKV7_13B_ctx192512/results/SFT_RWKV7_13B`

- 完整物理行192513 stored tokens -> **192512模型输入**；batch1/accum1。
- virtual-window开关保留但每行只有1个窗口，没有128K切片，没有重打包/截断。
- seed42，211633物理行中2117 eval、209516 train；每遍209516更新。
- 冻结本体2,948,065,280参数、CUDA BF16；只更新ROSA180,884,482参数、CUDA FP32。
- position=post_norm，layers为空：全部blocks -> ln_out -> ROSA residual -> head。
- compiled CPU SAM / suffix_link_lct_v1；不跳过padding或近似匹配。
- LR3e-5 ->3e-6，warmup200，最后10% cosine；betas .9/.99，eps1e-8，dense decay.01，clip1。
- Fused masked CE chunk1024。不开启外层AMP，FP32 ROSA关闭TF32。
- 从头初始化ROSA，新的优化器/循环/W&B；没有从128K checkpoint续训。

## 监控和保存

每200更新SFT eval（4个held-out rows），同时同步CPU FP32 8K NIAH：
3 depths ×2 distractor types ×2 prompt modes，max24生成token，4 CPU threads。
初始step0长CPU NIAH关闭，先进行训练；CPU NIAH超时上限4小时，失败不静默跳过。
这不是192K NIAH质量验证。post_norm首目标logit增量/排名和候选正确/错误分组
会在NIAH结果和W&B中记录。

首更新保存完整可恢复的adapter+两优化器+Lightning loop checkpoint，之后约每小时
在optimizer边界原子保存，保留历史，native-last.ckpt指向最新完整文件。
checkpoint外部引用原base路径+严格hash+参数名顺序，不是独立完整本体文件。
保存到用户指定/hikscale/models/ROSA；网络盘2.17GB首次保存可能花数分钟，
不把落盘等待当作训练/SAM耗时。

已修正compact模式原生epoch-end重复保存本体和fit-end同一步重复保存。
另修正PL1.9加载顺序：必须在LightningModule.on_load_checkpoint中合并并校验
外部本体，Callback.on_load_checkpoint发生在strict模型加载之后，单独使用太晚。
真实Lightning CPU单测已验证mid-epoch边界compact保存→恢复，参数及两个优化器
状态与不中断训练逐位一致；没有借此声称真实3B CUDA恢复已完成。

## 容量依据

192512测试已完成5次真实native GPU更新：无OOM/非有限值，本体完整hash不变，
只有ROSA更新。峰值allocated29.721GiB、reserved38.541GiB，预热后中位13.041s/步，
反向约0.31s、SAM约0.56–0.61s。不是长时间稳定性或192K质量提升证明。

测试报告：`/mnt/data/Codes/ROSA/outputs/native_rwkv_3b_postnorm_192k_20261109/report.json`
容量测试W&B run: 85291077。正式训练使用另一个run ID，记录在输出目录。
