# LLM 前视觉 Token 压缩方案

本目录整理 PruMerge、VisionZip、FOLDER、SCOPE 四类算法迁移到 LLM 层前的技术方案。

主文档：

- `TECHNICAL_DESIGN.md`：中文技术设计，包含位置考察、每个算法的推荐落点、实现约束、接口建议和验证计划。
- `ALGORITHM_POSITION_REVIEW.md`：逐算法位置考察，明确 adapter_pre、vision_late、llm_early、mlp_post 的取舍。
- `ADAPTER_PRE_POSITION.md`：解释 `adapter_pre` 在本仓库 ColQwen2.5 / Qwen2.5-VL forward 中的精确位置、张量流和代码入口。

当前状态：

```text
文档方案：已整理
代码实现：未开始
推荐首个实现：VisionZip / adapter_pre，因为本仓库已有 llmpre/visionzip_mrl 可复用
```

本目录只做技术文档，不改变现有训练或评测代码。若后续进入实现，建议新建可运行目录 `llmpre/pre_llm_algorithms/`，把四个算法做成统一 `adapter_pre` 框架下的 strategy。
