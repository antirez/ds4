# Progress Log: DS4 DSpark Investigation

## P0: Skill 构建 & Handoff Bundle 归档 — Done

**日期**: 2026-08-07

用户要求从 handoff bundle 构建 skill，用于 DSpark exactness investigation。我的工作流程是先读完全部 10 个 handoff 文件理解意图，再向用户确认 scope（限于 exactness 调查）、目标模型和存放位置（ds4 仓库内），然后构建 skill 文档并附录所有原始文件原文。

完成了以下内容：
- SKILL-ds4-dspark-exactness.md — 整合了 DSpark numerical drift bug 描述、四阶段工作流、pre-GGUF 7 项 checklist、historical PR #590/#659/#677 参考、benchmark 方法，以及全部 10 个 handoff 文件原文附录
- PLAN-ds4-investigation.md — 全局 plan，包含 P0-P4 四个阶段的目标、步骤、decision gates、success criteria
- MEMORY.md 索引行已添加

### 待确认事项
- DSpark support GGUF 在 .download/ 目录中只有 24MB，不确定是否下载完成。91GB 的 Layers37-42 变体是完整的，但不确定是否为目标模型
- ds4_gpu_tensor_read() 在 upstream main 的 ds4_gpu.h:55 有定义。P0 原文"可能需要适配层"记录错误——原因是本地 checkout SHA b030961 与 upstream 一致，不需要 adapter

### P0 需要 GPT 审阅的问题
- skill 内容是否覆盖了 handoff 的全部意图？stage table 行名设计是否正确？ds4_gpu_tensor_read 是否有替代方案？
