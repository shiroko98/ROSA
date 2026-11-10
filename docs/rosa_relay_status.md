# ROSA-Relay：实现、最终配置与评测归档

截至 2026-11-10，分支为 `codex/rwkv-rosa-compare`。训练已按要求在保存
step 1800 后停止。本文记录最终状态；早期短跑及其他训练目录见
[DP 训练记录](rwkv_dp_192k_training.md)，其配置和结果不能混作最终模型。

## 模型与训练

- 名称：**ROSA-Relay / 接力 ROSA**，模型名 `ROSA-Relay-RWKV7-2.9B`。
- 冻结本体：RWKV-7，2,948,065,280 参数；ROSA 180,884,482 参数。
- 本体权重：`/mnt/data/Models/RWKV-7/rwkv7-g1k-2.9b-20260930-ctx25600.pth`。
- 最终训练目录：`/hikscale/models/ROSA/rwkv7_3b_rosa_dp8_plain_ctx192512_m6_cap03_20261109`。
- 最终检查点：该目录下
  `checkpoints/native-user-stop-step000001800-20261110T004518267791Z.ckpt`。
- 八卡复制式 DP，不使用 ZeRO；每卡 microbatch=2，全局 batch=16，上下文 192512。
- 本体 BF16，ROSA FP32；在 `ln_out` 后、LM head 前做一次 `post_norm` 注入，输出 RMS 归一化。
- `scale_mode=logit_gap`、`dynamic_min_match=6`、`dynamic_max=0.3`、`dynamic_margin=0.5`。
  元数据中的固定 `scale=0.05` 不是动态模式下实际使用的常数增益。
- 每 7200 秒在优化器边界保存，另有初始及结束保存；独立 CPU 检索评测不阻塞训练。

SAM 根据原始 token 后缀匹配返回历史后继 token ID，再查独立可训练 ValueStore；
门控使用当前 hidden、查表值和匹配长度。`min_match=6` 指历史后缀匹配 token 数，
不是答案最短长度，也不把检索固定为六元组。离散 SAM/LCT 寻址在 CPU 上运行。
可学习值表与门控受 PLE / Engram 启发，尚未实现完整双分支 Engram 混合模型。

## 数据与代码入口

原始数据 `/mnt/data/Datas/SFT_RWKV7_13B` 已重处理到
`/mnt/data/Datasets/SFT_RWKV7_13B_rwkv_g1_ctx192512`。
格式为普通文本 `System:` / `User: …\n\nAssistant:`，不是特殊 token ChatML；
EOS / padding ID 为 0，只监督最后一轮 assistant，保留已有 thinking / 工具内容。
每行存 192513 token，移位后形成 192512 个输入 / 目标位置。

|入口|用途|
|---|---|
|`build_rwkv_plain_sft.py`、`rwkv_rosa/plain_sft.py`|并行重处理、普通词表编码、监督 mask 与 binidx 打包|
|`cpp_extensions/rwkv_plain_tokenizer.cpp`|编译版贪心 tokenizer 与字节监督区间|
|`rwkv_rosa/native/data_parallel.py`|DDP 与稀疏触达行梯度同步|
|`scripts/run_rwkv_3b_dp_192k_training.sh`|原生八卡训练，支持环境变量覆盖配置|
|`scripts/start_rwkv_3b_dp_cpu_eval.py`|启动新训练并派生独立 CPU 评测进程|
|`rwkv_rosa/native/scale_study.py`、`scripts/run_rwkv_scale_study.sh`|真实数据的 scale / min_match / cap 短跑实验|
|`rwkv_rosa/native/detailed_gpu_needle.py`、`sampling_gpu_needle.py`|多长度、多位置检索及采样敏感性实验|
|`rwkv_rosa/native/long_answer_cases.py`、`long_gpu_needle.py`|长问题及长答案检索，记录自由生成阶段 ROSA 激活|
|`rwkv_rosa/native/short_256k_needle.py`|保持旧短问答协议的 256K 外推|
|`rwkv_rosa/native/inference_speed.py`|128K 的 GPU / 混合 / CPU 速度测试|
|`rwkv_rosa/native/cpu_wkv_inference.py`、`cpp_extensions/rwkv_wkv_cpu_inference.cpp`|Linux 上推理专用的编译 FP32 CPU WKV|
|`scripts/summarize_*needle*.py`、`finalize_long_needle.py`|原始输出评分、配对统计及协议校验|
|`scripts/report_inference_speed.py`、`report_short_256k_needle.py`|已完成实验的汇总、图表和可选 W&B 发布|
|`scripts/plot_*needle.py`、`plot_inference_speed.py`、`publish_long_needle.py`|图表与长答案结果上传|

GPU worker 通过 `scripts/run_detailed_gpu_needle.sh` 设置原生 CUDA 环境，
以 `ROSA_NEEDLE_MODULE=rwkv_rosa.native.long_gpu_needle` 等选择模块。
检索 worker 接收 `--output DIR --slot N --workers COUNT`，读取 DIR 中已有的
`manifest.json`，按 slot 分配完整案例；协议 manifest 和原始输出保存在下述实验目录。
速度模块接收 `--output DIR --placement gpu|hybrid|cpu`，三种放置依次运行以避免争用。
这些模块是实验工具，需要已有的协议 manifest、训练元数据、外部本体和 adapter 快照。
数据、权重、日志和生成结果不纳入 Git。

报告脚本绘图依赖 Matplotlib；默认使用当前 Python，可通过 `ROSA_PLOT_PYTHON`
指定单独的绘图环境。短答案与速度报告只有显式加 `--publish` 才上传 W&B；
`publish_long_needle.py` 本身是上传入口。W&B 凭据通过现有认证配置读取。

## 最终检索结果

冻结同一本体，对相同输入成对比较 base / step1800；普通 RWKV 提示，greedy
（temperature=0、top_p=1、top_k=0），无重复惩罚。自然问答与前缀补全分列，
补全反映复制能力，不能作为自然问答准确率。完整输入包含表中文档长度加问题后缀。

|文档长度|短答案 QA：base / ROSA|短答案补全：base / ROSA|长答案 QA：base / ROSA|长答案补全：base / ROSA|
|---|---|---|---|---|
|8K|1/15 · 10/15|3/15 · 13/15|0/36 · 0/36|0/36 · 13/36|
|16K|0/15 · 7/15|2/15 · 11/15|未测|未测|
|32K|0/15 · 6/15|0/15 · 15/15|0/36 · 6/36|0/36 · 19/36|
|64K|0/15 · 1/15|0/15 · 15/15|未测|未测|
|128K|0/15 · 13/15|0/15 · 15/15|0/36 · 5/36|0/36 · 23/36|
|256K|0/15 · 0/15|0/15 · 6/15|0/36 · 0/36|0/36 · 0/36|

短答案为六位数字（3 token），每长度五个位置、每位置三例；QA 按 512-token
输出预算评分，补全按 128。主指标为最终答复中的第一个独立六位数正确，
未闭合 thinking 判失败。8K–128K QA 总计 1/75 → 37/75，提升 **48 个百分点**，
按重复长度的种子家族聚类、配对 bootstrap 95% 区间为 **[37.33, 58.67]** 个百分点。
256K 额外放大输出上限到 QA 4096 / 补全 1024，命中数没有增加。

长答案每长度三位置、两次重复、两种答案类型、三个目标长度，共 36 例。
随机数字答案为 16 / 64 / 128 token，合成中文答案实际为 16–137 token；
QA 输出上限 4096，补全 1024。指标要求最终答复包含完整连续目标字符串，
不计部分复制；整段输出严格等于目标的 EM 另列，全部为 0。
8K–128K 长答案 QA 为 0/108 → 11/108，提升 **10.19 个百分点**，
配对聚类 bootstrap 95% 区间 **[4.63, 16.67]**。

256K 超过 192512 的训练上下文，属于外推；小样本零命中不能证明真实准确率为零。
同一种子家族跨长度复用，统计时不能当成完全独立样本。

## 128K 推理速度

单张 H800 80GB；CPU 为 Xeon Platinum 8468 的一个 48 物理核 NUMA 节点。
batch=1，同一文档，三轮各 128 个计时生成步；每种放置只测一次完整文档预填充。

|放置|文档预填充(s)|自然问答 TTFT 估算(s)|自然问答 token/s|补全 token/s|
|---|---|---|---|---|
|本体 GPU + ROSA GPU|7.2961|7.3359|67.7202|67.5489|
|本体 GPU + ROSA CPU|7.2963|7.3365|68.7847|68.8531|
|本体 CPU + ROSA CPU|165.9951|166.1994|12.9730|12.8355|

GPU 本体 BF16，CPU 本体 FP32，不能视为同精度的纯硬件比较。
所有放置均使用 CPU SAM；混合放置的动态增益仍使用 GPU 冻结 head。
TTFT 为预填充、问题后缀和首 token 预测的分阶段求和估算，排除加载、JIT、
tokenizer、状态克隆及线程切换。GPU / 混合生成 token 相同；约 1–2% 的吞吐差异
不足以认定混合部署更快。CPU BF16 / AMX、量化等进一步优化尚未实测。

## 数据归档

公共前缀为 `/hikscale/models/ROSA/`，每个目录含协议、原始输出及结果报告；
最终汇总目录还包含 CSV、图表、校验信息、源码快照和 `data_bundle.zip`。

|目录|内容|W&B run|
|---|---|---|
|`needle_sampling_step1800_20261110`|8K–128K 短答案与采样对比|`nmj76qga`|
|`short_needle_step1800_ctx256k_20261110`|短答案 256K 及所有长度汇总|`mtec1nsb`|
|`long_needle_comparison_step1800_20261110`|8K / 32K / 128K / 256K 长答案汇总|`llfjkvsf`|
|`inference_speed_step1800_ctx128k_20261110`|三种设备放置的速度对比|`an5z2y6v`|

W&B 地址模板：`http://dian.sea-group.org:9098/mobius/ROSA-RWKV7/runs/<run_id>`。
推理快照位于 `detailed_needle_gpu_step1800_20261110/adapter-step1800.pth`，
旁边 JSON 保存融合配置和权重哈希。Git 不备份这些外部文件。
