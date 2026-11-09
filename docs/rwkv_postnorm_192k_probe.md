# 3B post_norm：完整 192512-token 容量测试（2026-11-09）

测试目标：不是继续搜索更长上限，只验证用户这份192K数据的完整192512输入能否训练。

配置：原生 new_sft 训练、2.9B frozen BF16 CUDA，本次使用 post_norm ROSA FP32 CUDA，
compiled CPU SAM (suffix_link_lct_v1)，batch1，accum1，fused CE chunk1024。
使用原192513-token物理行和原监督mask，seed42物理train/eval split；window开关开启
但每条物理行只映射一个窗口，没有截断、重打包、截短BPTT或参考Python回退。
进行5次真实前向/反向/optimizer updates，核查非有限数值、body冻结哈希和ROSA更新。

容量测试不运行CPU NIAH、不保存试验checkpoint，避免把评测与网络盘落盘当成
训练步耗时。独立W&B online capacity run，日志与native_steps均保留。

原128K生产训练先在优化器边界保存并停止。测试结束后，无论192K成功/失败，
都恢复原128K checkpoint+两个优化器+loop状态，不擅自把128K dataloader/scheduler
元数据解释成192K继续训练。正式切换上下文需另行明确数据遍历/调度政策。

控制器、命令及结果：
`/mnt/data/Codes/ROSA/outputs/native_rwkv_3b_postnorm_192k_20261109/`

- probe_and_resume.py：等待旧进程完整退出→192K测试→恢复128K；
- command.json、probe.log、probe/native_audit.json、probe/native_steps.jsonl；
- report.json：实际结果和恢复PID/检查点，不把仅启动当成通过；
- 独立W&B run id写在report.json。

短试验可证明这些batch已跑通，不等于长时间稳定性或192K检索质量已验证。

## 暂停保存的额外 I/O 修复

准备测试时发现，compact ROSA callback 保存 user-stop checkpoint 后，上游epoch-end
仍会保存包含不可变本体的完整.pth，随后fit-end又保存同一步compact checkpoint。
这与约9秒的GPU训练步无关，却会额外写入约8.8GB并等待网络盘。
修复为：compact ROSA模式由专属callback负责保存，原生通用保存函数不再重复写本体；
同一optimizer step已成功保存时，fit-end不再重复保存。
非ROSA和显式legacy full-checkpoint模式保留原行为。

修复不改变forward、loss、梯度或数据顺序；当前已加载旧代码的停止进程仍需要完成
它正在进行的保存，恢复训练后使用修复。覆盖测试见
`tests/test_rwkv_native_checkpoint_policy.py`。全套215项测试OK，3个CUDA专用跳过。
恢复代码overlay（原生产source_snapshot加以下覆盖文件）记录于容量测试目录的
`resume_source_overlay/`，附文件哈希。

## 后续用户指令（本次以此为准）

192K测试通过后，用户明确要求不要恢复128K、直接192K正式训练。
自动恢复128K的尝试在任何新更新之前因compact checkpoint模型hook顺序退出，
没有新增训练步；旧任务保留step30。已改为LightningModule加载前hook，真实
Lightning CPU测试验证adapter、dense/sparse optimizer逐位连续恢复，未再重试128K。

正式192K任务详见 `/mnt/data/Codes/ROSA/docs/rwkv_3b_postnorm_192k_training.md`。
旧控制器已结束，不会再次自动恢复128K。
