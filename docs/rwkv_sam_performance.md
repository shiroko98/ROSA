# SAM 长序列寻址优化（2026-11-09）

## 问题不是 BF16 精度，也不只是在 Python 上运行

原实现每插入一个 token，沿 suffix link 逐级更新最近出现位置 `endpos`。
长段重复 token（真实训练数据中的 padding 为 65532）形成长链，累积工作量
退化为平方级。之前的 128K 第 2 个训练 batch，完整更新约 69.94 秒；其中
Python SAM 寻址约 60 秒。简单搬到 C++ 仍约 5.75 秒，没有解决算法退化。

另外，`pack_address_meta_batch` 每个 token 对五个 Tensor 分别做标量写入。
128K 单样本约 65 万次 Tensor 索引赋值，在核心算法优化后仍花约 2 秒。

## 修改

1. C++ SAM 的 suffix-link 树使用 rooted link-cut tree，惰性执行整条
   root-to-last 路径的最近出现位置赋值，按需点查询；SAM clone 创建时保留
   原状态最近位置并重接父节点。树操作摊还对数级，避免重复 token 的逐级更新。
2. 五个寻址字段批量生成 Tensor，不再逐 token 写入。
3. 原 Python `fast` 后端保留作为语义参照；生产使用 `compiled_cpu`，
   审计中记录 `suffix_link_lct_v1`。没有跳过 padding、缩短匹配长度或近似搜索。
4. 完整寻址时间记录为 `rosa_address_seconds`，与模型 forward/backward、
   CPU needle eval、保存 checkpoint 的时间区分。
5. 编译缓存检查源文件时间，防止新进程继续使用旧二进制；显式 force rebuild
   也传递到 setuptools 编译命令。已加载扩展的旧进程需要重启。

## 实测（CPU 完整寻址：含元信息和五个结果 Tensor 打包）

固定 4 CPU threads。新实现取三次重复的中位数；旧 Python 为一次完整运行。
这些是 SAM-only 测量，不是 GPU 总训练吞吐测试。

| 输入 | Python 参照（批量打包已启用） | 新 C++ + 批量打包 | 加速 |
|---|---:|---:|---:|
| 真实 131,072-token 慢 batch | 58.889 秒 | 0.275 秒 | 214× |
| 16,384-token 全重复 padding | 3.139 秒 | 0.0227 秒 | 138× |
| 131,072-token 随机序列 | 0.446 秒 | 0.284 秒 | 1.57× |

真实 batch 的 token/mask 联合 SHA256 与之前慢训练 update 2 完全一致：
`9cf633ba91a5342f3744673567d21c31776114583c18054f24673bbe6df568af`。
这不是换简单样本后的比较。实际训练的单步速度要看生产日志，不直接把寻址
214× 加速说成整步训练也快 214×。

仅 C++ 核心（全重复 token，不含元信息打包）：
8K 0.0018s、16K 0.0040s、32K 0.0083s、64K 0.0178s、128K 0.0379s。
闭式正确性检查：首 token 无地址，之后始终返回同一 token，匹配长度逐个递增。

## 正确性和限制

- 三类实测输入均逐元素检查五字段完全一致：addr_ids、raw_match_lens、
  fired_match_lens、valid_mask、special_mask。
- 回归还覆盖随机/clone 密集序列、不同 min_match_len、特殊 token 开关、
  memory padding 过滤、因果前缀不变性、空 batch 和不等长行的打包默认值。
- 全套 CPU 可运行测试：207 项，成功，3 项 CUDA 专用测试跳过。
- 这是已测样本和回归覆盖的精确一致，不声称对所有输入的形式化证明。
  没有改变 BF16 模型计算策略，也没有承诺 CPU FP32 与 CUDA BF16 逐位一致。

报告/可复现脚本：
`/mnt/data/Codes/ROSA/outputs/native_rwkv_3b_lastlayer_20261109/sam_optimization.json`
和同目录 `sam_scaling.py`；测试：`tests/test_rosa_cpp_extension.py`。
生产配置：`/mnt/data/Codes/ROSA/docs/rwkv_3b_lastlayer_training.md`。
