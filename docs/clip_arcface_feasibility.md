# CLIP + ArcFace 证人描述检索流水线

- 现有 `pyWitnessAI` 已有稳定的人脸检测/特征提取/相似度计算骨架（`DeepFace` + `FaceNet`），可直接复用为 ArcFace 重排阶段的工程底座。
- 需要新增的关键能力是：
  1) 文本结构化抽取（LLM）
  2) CLIP 文本-图像向量化检索（FAISS）
  3) 组合打分与多样性约束
- 研究设计（TP/TA 混合用于 ROC/pAUC/CAC）在实现上没有阻塞，主要挑战在于**离线预编码、缓存策略、实验平台接口、可复现日志**。

- **Phase A（2–3 周）**：离线索引 + 单机批处理 API + 小样本验证（200 lineups）
- **Phase B（2–4 周）**：接 Qualtrics/Webhook 在线生成 + 审计日志 + 失败回退策略

---

## 2. 现有代码与需求匹配度

### 2.1 已有可复用能力

1. **人脸分析与坐标/置信输出**
   - `FrameAnalyzerDeepface`, `FrameAnalyzerMTCNN`, `FrameAnalyzerOpenCV` 已把检测、坐标、置信封装成统一 `analyze_frame` 输出，可复用到素材清洗和质量控制。

2. **人脸 embedding 与相似度框架**
   - `SimilarityAnalyzer` / `LineupLoader.compare_faces` 已有 embedding + 欧氏距离对比流程；将模型切换为 ArcFace 并改成 cosine 即可迁移到你的 Stage 3 需求。

3. **图像加载与预处理骨架**
   - `LineupLoader` 提供统一读取与 resize，可扩展为“数据库预处理 -> 向量缓存”流水线。

### 2.2 当前缺口

- **无文本管线**：目前没有描述解析与属性 schema 校验模块。
- **无 CLIP/FAISS 依赖**：`pyproject.toml` 仅含 `deepface` 等，未包含 `transformers`/`open_clip`/`faiss`。
- **无实验服务接口**：尚无对 Qualtrics 的 API/Webhook 收口。
- **无审计与可复现元数据**：实验必须记录 prompt/version/model/index hash/α 参数等。

---

## 3. 工程化拆解

## 3.5.1 Stage 1：描述解析（LLM）

### 可行性
技术风险主要在“输出稳定性”而非模型能力。

### 实现建议
- 定义 Pydantic schema（字段与你文案一致：sex/age/ethnicity/skin tone/hair/face shape/facial hair/features...）。
- 使用“结构化输出”模式（JSON schema / function calling）而不是自由文本。
- 对缺失字段填 `unspecified`。
- 生成 CLIP-friendly 句子模板，例如：
  - `A frontal mugshot-style photo of a male, age 25-35, light skin tone, short dark hair, clean-shaven, oval face, neutral expression.`

### 关键控制点
- **提示词版本化**（`prompt_v1.0`）
- **解析失败回退**（直接使用原始描述 + 基本清洗）
- **敏感属性最小化**（仅用于研究变量，不参与不必要推断）

---

## 3.5.2 Stage 2：CLIP 语义检索

### 可行性
5000–10000 图像规模下，FAISS（CPU）完全足够，延迟通常 <100ms 到数百毫秒级（取决于硬件与批量策略）。

### 实现建议
- 模型：`openai/clip-vit-large-patch14`（与你设计一致）或 `open_clip` 等价实现。
- 离线任务：
  1) 图像统一预处理（灰度策略要谨慎，见下文）
  2) 批量编码图像 embedding
  3) 建 `faiss.IndexFlatIP`（向量先 L2 normalize，IP 等价 cosine）
  4) 存 index + id 映射 + 数据版本
- 在线检索：文本编码 -> top50 返回候选 ID。

### 风险提示
CLIP预训练是 RGB 分布；**直接灰度可能削弱检索能力**。建议：
- 检索阶段保留 RGB；
- 展示给被试时可灰度（如果伦理/规程要求）；
- 在预实验中比较 RGB vs 灰度对描述匹配评分影响。

---

## 3.5.3 Stage 3：ArcFace 重排

### 可行性
高。你仓库已基于 DeepFace 做 embedding；DeepFace 支持 ArcFace，迁移成本低。

### 实现建议
- 对 suspect mugshot 与 top50 候选提 ArcFace embedding。
- 组合分数：`score = α * clip_sim + (1-α) * arcface_sim`。
- 你给的 `α=0.6` 可作为先验，但建议在验证集网格搜索（0.3,0.5,0.6,0.7）。
- 多样性约束（pairwise sim < 0.55）可做贪心选择：按 score 排序依次加入，不满足阈值则跳过。
---

## 4. 与实验设计对齐的系统架构（可直接开工）

建议新增如下模块：

- `src/pyWitnessAI/pipeline/description_parser.py`
- `src/pyWitnessAI/pipeline/clip_index.py`
- `src/pyWitnessAI/pipeline/arcface_reranker.py`
- `src/pyWitnessAI/pipeline/lineup_builder.py`
- `src/pyWitnessAI/pipeline/schemas.py`
- `src/pyWitnessAI/pipeline/audit_logger.py`

### 数据流（在线）
1. Qualtrics 提交描述（participant_id, condition, TP/TA flag）。
2. `description_parser` -> structured attrs + normalized sentence。
3. `clip_index.search(text, k=50)`。
4. `arcface_reranker.rank(candidates, suspect_img, alpha=0.6)`。
5. `lineup_builder.select_top5_with_diversity(threshold=0.55)`。
6. 写回 lineup image IDs 给前端展示。
7. 记录审计日志（模型版本、耗时、失败码）。

### TA 线up注意点
- TA 条件必须保证不含 target/suspect；
- 重排阶段若传 suspect embedding 可能在 TA 条件引入不一致逻辑。
- 研究上更稳妥做法：
  - TP：用 suspect 重排；
  - TA：用“描述一致性 + 多样性 + 与已知 target 距离下限约束（若有）”。

（你文案当前把 ArcFace 重排绑定 suspect，建议在预注册里明确 TA 的处理规则。）

---

## 5. 需要先澄清的方法学点

1. **TA 条件的 ArcFace 重排定义**
   - 若无真实 suspect，不应使用 target embedding；否则可能破坏盲测逻辑。

2. **记忆污染控制**
   - 增加标准指令：不得查阅外部资料、不得截图回看；
   - 禁止后退与重复播放；
   - 在 free-recall 前加入短提醒，避免“猜测型补全”。

---

## 6. 最小可行实现（MVP）里程碑

### Milestone 1（基础可运行）
- 完成 5000 脸图离线 CLIP 编码 + FAISS 索引。
- 完成描述->结构化->句子重建。
- 完成 top50 检索 + ArcFace 重排 + top5 输出。

### Milestone 2（实验可用）
- 增加 participant 级日志（jsonl）。
- 增加失败回退：
  - LLM失败 -> 原描述直送 CLIP
  - ArcFace失败 -> CLIP top5 + 多样性过滤
- 增加缓存（重复描述、重复 suspect）。

### Milestone 3（预注册验证）
- 200 lineups 盲评工具 + 打分汇总。
- 输出 Krippendorff’s alpha 与组间比较报告。

---

## 7. 依赖与工程建议

建议在依赖中增加（按你环境二选一）：

- 文本/视觉：`transformers`, `torch`, `Pillow`
- 向量检索：`faiss-cpu`
- 结构化校验：`pydantic`
- 服务层（如要在线）：`fastapi`, `uvicorn`

并补充：
- 固定随机种子与模型版本；
- index 构建脚本输出 hash；
- 单元测试覆盖：schema 校验、多样性过滤、TP/TA 规则。

---

## 8. 与现有仓库的整合优先级

1. 先保留现有 `VideoAI.py` 不大改，只“新增 pipeline 子包”。
2. 复用 `DeepFace.represent` 的调用风格，统一 embedding 抽象。
3. 把实验逻辑（TP/TA、条件分配）放在独立 orchestrator，避免和视觉算子耦合。

---

## 9. 一句话建议

真正决定成败的是**实验条件等价性定义（尤其 TA/Benchmark）+ 审计可复现工程**。先做 200-lineup 验证闭环，再接入大样本 Prolific 主实验，会最稳。
