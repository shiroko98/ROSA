# ROSA 开发进度

## 当前状态

- 已有基础：`train_qwen_llama_vs_rosa_v2.py` 中已实现 ROSA 的 `sam` 检索版本，并有基础回归测试。
- 当前分支：`codex/online-rosa-p0-foundation`
- 当前阶段：P0 基础设施

## 本轮任务

- [x] 梳理改良文档与统一路线图
- [x] 建立本地规划/进度/操作文档
- [x] 实现 Online ROSA State
- [x] 抽象统一 AddressMeta 接口
- [x] 补逐 token 一致性测试
- [x] 完成自我验证并提交本轮 commit

## 本轮已完成

- 在 `train_qwen_llama_vs_rosa_v2.py` 中新增：
  - `AddressMeta`
  - `RosaStateSnapshot`
  - `OnlineRosaState`
  - `rosa_addressing_with_memory()`
- 现有 `naive` / `sam` 检索已复用统一地址抽象，保留旧的 tensor 返回接口。
- `naive` reference 与在线 state 路径已分离，后续可继续做独立一致性验证。
- 新增测试文件 `tests/test_rosa_online_state.py`。

## 自我验证记录

- `conda run -n model python -m unittest tests.test_rosa_online_state`
- `conda run -n model python -m unittest discover -s tests`
- 结果：共 16 个测试，全部通过。

## 备注

- 本轮优先保证“语义正确 + 接口稳定”，性能优化与服务化调度放到后续阶段。
