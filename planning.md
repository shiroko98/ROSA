# ROSA 改良规划

## 总目标

围绕 `ROSA_PLE_Engram_分析与改良方案_v2_工程版.docx` 与 `ROSA_统一路线图_TODO合并版.docx`，按 P0 -> P1 的顺序推进 ROSA 改良，优先完成在线化基础设施，再逐步接入 PLE 风格的 value store 与 Engram 风格的门控/调度。

## 当前迭代

- 分支：`codex/online-rosa-p0-foundation`
- 迭代主题：P0 基础设施
- 当前状态：已完成第一轮在线状态与地址接口落地，下一步可进入“最小在线注入闭环”
- 对应路线图任务：
  - 把 ROSA 从离线/整段检索改成增量在线状态机
  - 抽象地址生成接口，解耦“匹配”和“取值”

## 本轮步骤

1. 建立本地规划、进度和操作文档。
2. 为在线 ROSA 定义统一地址结构 `AddressMeta`。
3. 实现可增量调用的 `OnlineRosaState`，支持 `reset()` / `prefill()` / `update_one()` / `snapshot()`。
4. 让在线地址结果与现有 reference 语义逐 token 对齐。
5. 补测试并完成一次自我验证。
6. 更新 `progress.md` 后提交 commit。

## 本轮结果

- 已新增 `AddressMeta` / `RosaStateSnapshot` / `OnlineRosaState`。
- 已新增 `rosa_addressing_with_memory()`，把地址生成与 value 消费初步解耦。
- 已保留独立 `naive` reference 路径，并单独提供在线 state 路径用于逐 token 对齐验证。
- 已补 `tests/test_rosa_online_state.py`，覆盖逐 token 对齐、special 过滤、snapshot/reset。
- 已在 `model` 环境执行 `python -m unittest discover -s tests`，当前通过。

## 自我验证清单

- 单元测试覆盖 `prefill`、`update_one`、`reset`、special target 过滤、逐 token 对齐。
- 在线状态输出与当前 reference 检索结果保持一致。
- 现有 ROSA / 数据加载 / dataset 测试不回归。
- 提交前检查 `git diff`，确认文档、代码、测试都已同步。

## 下一阶段预留

- 最小在线注入闭环：单层注入，先继续复用共享 `embed_tokens`。
- 性能基线：补 decode/prefill profiling 与 match coverage 指标。
- P1：per-layer value store + context-aware gate。
