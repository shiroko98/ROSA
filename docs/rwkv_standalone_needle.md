# 训练与大海捞针独立运行（2026-11-09）

## 新流程

- 192K post_norm训练默认 `--rosa_needle_every_eval 0`。仍每200步运行原SFT验证，
  但训练代码不再启动/等待大海捞针进程。
- 独立worker顺序读取 `native_checkpoints.jsonl` 中已原子发布的完整checkpoint，
  相同步数（initial/hourly/user-stop）只评估一次。训练无需等待评测或上传。
- 每个任务固定自己的checkpoint，直接mmap读取ROSA权重，不转换/重复保存3B本体。
  严格核对position、层号、scale、step、原base哈希和参数顺序。
- worker失败、超时、W&B上传失败只记录自己的status/日志，不影响训练。
- 单worker文件锁防止同一队列重复消费；FIFO，不静默丢弃积压的checkpoint。
- 默认CPU，不和训练争抢GPU显存；GPU方案仍需独立的实测与资源预算，不能把
  训练allocator cache误当成另一个进程可直接使用的显存。

## 启动

```bash
bash /mnt/data/Codes/ROSA/scripts/run_rwkv_standalone_needle.sh \
  --train-dir /hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109 \
  --watch --min-step 201 --wandb
```

也可独立测试指定checkpoint：

```bash
bash /mnt/data/Codes/ROSA/scripts/run_rwkv_standalone_needle.sh \
  --checkpoint /absolute/path/to/native-hourly-stepXXXXXXXXX.ckpt \
  --train-dir /hikscale/models/ROSA/rwkv7_3b_rosa_postnorm_ctx192512_20261109 \
  --wandb
```

默认8K文档，depths .1/.5/.9，普通/近似档案干扰，QA/exact_prefix，greedy24token。
这是CPU FP32 reference监测，不是192K质量验证或native BF16逐位一致保证。

每个step单独目录：request.json、progress.json、worker.log、report.json、status.json。
评估完成后W&B使用独立run，group=`needle-of-训练runID`，明确记录source checkpoint
和source_training_step，避免延迟完成的旧step写进较新的训练history。
progress.json逐条记录部分结果，base完整评测后立即缓存，不必等trained全部完成。
完整report才可标记complete；解析worker.log的部分结果不能当成完整准确率。

## 当前200步的安全切换

旧主进程已经进入同步subprocess.run，新源码不能改变正在运行的Python调用。
直接杀CPU子进程会令旧训练异常退出，200步权重虽有永久快照，但没有同时期的
优化器状态。因此不强杀、不拿step1优化器冒充step200续训。

已设置STOP，在当前旧评测返回后，主进程于第200步优化器边界保存完整checkpoint，
退出；控制器核对step>=200、ctx192512、post_norm和optimizer/loop状态后，按相同
base、ROSA权重、优化器、数据循环恢复192K训练，并明确关闭训练内needle。
控制器随后等待新optimizer update以验证真正恢复；失败会写switch_failed而不
伪称训练继续。

控制状态：
`/mnt/data/Codes/ROSA/outputs/needle_standalone_switch_20261109/status.json`

独立watcher已启动，从step201起消费后续checkpoint，避免把刚跑完的同一200步测试
重复计算一遍。当前200步旧测试原始结果和W&B仍保留。

## 验证

- 全套CPU可运行测试220项OK，3个CUDA专用跳过（新standalone模块加入后的结果）。
- 覆盖checkpoint元数据、重复步数、部分JSON行、独立子进程环境、异常不传播训练、
  不重复运行已完成任务、原始adapter与compact checkpoint得到相同NIAH结果。
- 真正独立CLI小模型端到端smoke：1次真实优化器更新的compact checkpoint，
  CPU子进程读取并完成256-token、2depth/difficulty组合的8条回答，status=complete。
  这只是流程测试，不能当成3B训练效果。结果位于
  `/mnt/data/Codes/ROSA/outputs/needle_standalone_smoke_20261109/`。
