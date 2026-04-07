# ROSA 本地操作备忘

## 环境

- Conda 环境：`model`
- 主训练脚本：`train_qwen_llama_vs_rosa_v2.py`
- 主测试入口：`tests/test_train_qwen_llama_vs_rosa_v2.py`

## 常用命令

```powershell
conda activate model
python -m unittest discover -s tests
```

```powershell
git status --short --branch
git log --oneline -5
```

## Online ROSA 最小闭环

- 入口能力：
  - `RosaFusedLM.init_online_state(batch_size)`
  - `OnlineRosaBatchState.prefill(token_ids, pad_id=...)`
  - `RosaFusedLM.forward_online(input_ids, rosa_online_state=...)`
- 当前版本仍使用共享 `embed_tokens` 作为 value，适合先验证在线路径与语义一致性。

## Profiling 基线

- profiling 脚本：`profile_rosa_online_baseline.py`
- 输出文件：`<out_dir>/profile_report.json`
- 对比模式：
  - `baseline`
  - `rosa_reference`
  - `rosa_online`
- 当前 decode 指标是无 KV cache 的单步 microbenchmark，适合比较 ROSA 分支路径开销与一致性，不等同于最终 serving 吞吐。

## VS Code Launch

- `ROSA v2 - Online Baseline Profile (Smoke)`：快速验证脚本链路。
- `ROSA v2 - Online Baseline Profile (Qwen ckpt compare)`：用现有 Qwen checkpoint 直接比较 reference vs online。

## 当前开发约定

- 新任务统一在 `codex/*` 分支上进行。
- 每完成一个可独立验收的任务，更新 `progress.md`，然后提交一版 commit。
- 规划与拆解写入 `planning.md`，执行过程中持续修订。
- 每一步完成后都要做自我验证：跑测试、看日志、核对输出，再决定是否进入下一步。
