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

## 当前开发约定

- 新任务统一在 `codex/*` 分支上进行。
- 每完成一个可独立验收的任务，更新 `progress.md`，然后提交一版 commit。
- 规划与拆解写入 `planning.md`，执行过程中持续修订。
- 每一步完成后都要做自我验证：跑测试、看日志、核对输出，再决定是否进入下一步。
