# pyWitnessAI 模块职责地图

这份地图描述当前代码归属，不记录历史审查过程。核心包提供数据、解析、评分、
生成、阵容和评估能力；研究目录组合这些能力，并保存自己的协议、数据规则和入口。

## 目录边界

| 位置 | 应保存的内容 |
| --- | --- |
| `src/pyWitnessAI` | 可由不同研究调用的功能；接收数据、配置和可替换后端 |
| `src/pyWitnessAI/API` | 对外入口及外部服务适配；HTTP、CLI、OpenAI 调用和模型下载 |
| `studies` 中保留的脚本与笔记本 | 各自的实验设置、数据准备、分析和验证步骤；调用核心包 |
| `test` | 通用模块的回归测试 |
| `test/fixtures/georgia_state_video1` | 八张必要的小型测试 PNG；随仓库和源码分发包提供，不进入 wheel |
| `data` | 被 Git 忽略的本地数据与生成结果；图片、Excel、生成 CSV 不随 clone 或包发布 |
| `docs`、`tutorial` | 公共接口说明与使用示例 |

HTTP/CLI 入口调用通用功能；通用功能通过后端接口使用外部服务适配器，
本地推理与计算留在核心。研究入口组合这些能力，核心模块不导入研究目录。
固定人群、固定身份数、研究阶段名称和冻结 JSON 均由研究持有。
Docker 和部署基础设施属于模块外部。
研究目录保留生成脚本、配置和数据准备说明；真实研究数据由使用者在本地提供。
`data/generated_filler_benchmark` 等现有生成路径继续可用，测试夹具由 `test/fixtures` 管理。

## 描述、图片与检索

| 核心文件 | 职责 |
| --- | --- |
| [ImageCatalog.py](../src/pyWitnessAI/ImageCatalog.py) | 图片发现、路径与 ID 规范化、图片有效性检查、元数据连接；共享图片扩展名和输入适配函数 |
| [FaceAttributeSchema.py](../src/pyWitnessAI/FaceAttributeSchema.py) | 属性定义、字段解析和结构化描述；默认 schema 只解析属性，benchmark 的负样本需显式定义 |
| [_description_rules.py](../src/pyWitnessAI/_description_rules.py) | 共享属性词表、通用对照规则与英语语义作用域；描述解析器和提示词构建器均依赖它，避免相互导入 |
| [Description.py](../src/pyWitnessAI/Description.py) | Raw、lenient、strict、schema 解析器及 `ParsedDescription`；保留原文、模型文本和属性审计 |
| [EyewitnessContrastPrompts.py](../src/pyWitnessAI/EyewitnessContrastPrompts.py) | 把描述/属性转成正向与参照提示词计划；通用模板默认不限定性别、族群或数据集 |
| [Scoring.py](../src/pyWitnessAI/Scoring.py) | 模型无关的 cosine/contrast 评分策略；使用已解析属性，决定如何组合模型分数 |
| [FaceSearch.py](../src/pyWitnessAI/FaceSearch.py) | Selector 后端、评分校验、稳定排序、入选规则和真值统计；返回可显示/导出的 `SelectionResult` |
| [FillerSelector.py](../src/pyWitnessAI/FillerSelector.py) | 为描述检索提供默认选前 5 张的入口；复用 `FaceSearch`；也提供按人脸距离选 fillers 的函数 |
| [clip.py](../src/pyWitnessAI/clip.py) | 可复用的文本/图片编码与索引、嵌入缓存、模型 revision 和本地快照支持 |
| [ContrastiveClipScoring.py](../src/pyWitnessAI/ContrastiveClipScoring.py) | 对多份提示词计划批量复用同一个 CLIP 索引，计算多种评分方法并输出分量、排名与 provenance |
| [_image_display.py](../src/pyWitnessAI/_image_display.py) | 私有排名图片网格和图片加载；供检索及两个 benchmark 共用，按需加载绘图库 |

常用路径是 `FillerSelector → FaceSearch → 评分策略 → SelectorBackend`。
已有图片池可直接用 `FaceSearch`，已有嵌入或批量实验可直接用 `ClipIndex` 和
`ContrastiveClipScorer`。这些入口共享底层计算，调用者按需要选择层级。

排名检查图显示身份、属性标签和分数；正式阵容展示由 `LineupBuilder` 管理，
需要隐藏研究角色及真值，故两种展示保留各自的接口。

## 生成、标准化与 benchmark

| 核心文件 | 职责 |
| --- | --- |
| [FaceGenerator.py](../src/pyWitnessAI/FaceGenerator.py) | 从描述、参考图或两者生成肖像；组织提示词、生成批次和结果记录 |
| [face_generation_backend.py](../src/pyWitnessAI/face_generation_backend.py) | 通用生成请求、后端基类、校验和注册表；按所选提供商加载适配器 |
| [GeneratedFaceDataset.py](../src/pyWitnessAI/GeneratedFaceDataset.py) | 维护可复用的生成图片 manifest；导入图片、按属性查询和更新记录 |
| [FaceStandardizer.py](../src/pyWitnessAI/FaceStandardizer.py) | 对齐、缩放和质量记录；保留原图并输出可用于检索或阵容的 catalog |
| [CodeFormerEnhancer.py](../src/pyWitnessAI/CodeFormerEnhancer.py) | 可选的本地肖像增强；由标准化流程调用并保留增强记录 |
| [model_files.py](../src/pyWitnessAI/model_files.py) | 本地文件的流式哈希；供模型校验和研究复现共用 |
| [AttributeRetrievalBenchmark.py](../src/pyWitnessAI/AttributeRetrievalBenchmark.py) | 评估现有 manifest 或两个标签文件夹中的单个二元属性；配置正/负提示词与评分模式 |
| [FillerSelectionBenchmark.py](../src/pyWitnessAI/FillerSelectionBenchmark.py) | 从描述构造单属性或累积阶段，匹配/生成受控正负样本，并逐阶段评估 |

两个 benchmark 复用图片发现、selector 评分归一化、稳定排序和排名展示；各自保留
数据准备及指标语义。`AttributeRetrievalBenchmark` 提供 Precision/Recall@K、AP、
ROC-AUC、阈值统计；`FillerSelectionBenchmark` 提供每阶段的 top-N 命中和完整命中检查。
它们按带标签的图片统计，保留标签来源，不自动把生成提示词当成已验证的视觉真值。
研究的固定人群与候选池规则不进入这两个类。

## 阵容与会话

| 核心文件 | 职责 |
| --- | --- |
| [LineupBuilder.py](../src/pyWitnessAI/LineupBuilder.py) | `LineupBuilder`、`Lineup` 和成员记录；固定成员、角色、顺序、布局和展示导出 |
| [LineupSession.py](../src/pyWitnessAI/LineupSession.py) | `create_lineup` 组合已有功能；`LineupSession` 处理人工反馈、下一轮、持久化和重试 |
| [LineupAIDecision.py](../src/pyWitnessAI/LineupAIDecision.py) | 依据距离表及明确角色作出阵容决策；供视频阵容流程使用 |
| [LineupDecider.py](../src/pyWitnessAI/LineupDecider.py) | 对已有表格提供阈值决策及 verify 风格决策结果 |

生成器负责产生图片，builder 负责组装和呈现，会话负责推进交互。HTTP 适配调用
同一套 Python 功能。人工确认会话使用未知身份标签；它不赋予 TP/TA 真值。

## API：对外入口与外部调用

| 文件 | 必要职责 |
| --- | --- |
| [API/__init__.py](../src/pyWitnessAI/API/__init__.py) | 轻量命名空间；导入包时不加载 FastAPI 或 OpenAI SDK |
| [API/http.py](../src/pyWitnessAI/API/http.py) | `create_app`：HTTP 请求、输入路径、响应和会话锁；调用通用阵容/会话功能 |
| [API/cli.py](../src/pyWitnessAI/API/cli.py) | `parse`、`serve` 的参数解析和分发；控制台与 `python -m pyWitnessAI` 共用 |
| [API/openai.py](../src/pyWitnessAI/API/openai.py) | `OpenAIImageBackend`、OpenAI 模型常量、客户端调用、参考图上传与响应解码 |
| [API/models.py](../src/pyWitnessAI/API/models.py) | `download_model`：下载、缓存和校验模型文件；复用本地文件哈希 |

HTTP 嵌入使用 `from pyWitnessAI.API.http import create_app`，命令行使用
`pyWitnessAI serve`。配置生成后端也可直接 `from pyWitnessAI import OpenAIImageBackend`。
API 文件各自持有实际适配逻辑；业务类无需增加一组转发方法。
CLIP、CodeFormer 的本地推理和模型文件哈希仍属于核心。

研究专用服务由研究项目提供独立入口、请求协议和测试，调用公共 Python 接口。
候选池、分组、问卷与验证配置由各项研究维护。

## 视频与既有辅助接口

| 文件或目录 | 当前用途 |
| --- | --- |
| [Images.py](../src/pyWitnessAI/Images.py) | 既有图片路径集合与 DataFrame 接口 |
| [ImagesAI.py](../src/pyWitnessAI/ImagesAI.py) | 图片集之间的人脸识别距离计算；需要 vision 可选依赖 |
| [ImagesCategorizer.py](../src/pyWitnessAI/ImagesCategorizer.py) | 依据 CSV/Excel 标签整理照片 |
| [Video.py](../src/pyWitnessAI/Video.py) | 视频读取、帧分析与处理循环 |
| [VideoAI.py](../src/pyWitnessAI/VideoAI.py) | 人脸检测/识别分析器及既有 `LineupLoader` 接口 |
| [VideoProcessor.py](../src/pyWitnessAI/VideoProcessor.py) | 裁剪、像素处理、显示和写出等帧处理器 |
| [VideoLineupPipeline.py](../src/pyWitnessAI/VideoLineupPipeline.py) | 组合视频帧、人脸距离和阵容决策，导出逐帧结果及 pyWitness 格式 |
| [utils/Constants.py](../src/pyWitnessAI/utils/Constants.py)、[utils/plots.py](../src/pyWitnessAI/utils/plots.py)、[utils/DataFlattener.py](../src/pyWitnessAI/utils/DataFlattener.py) | 既有配色、结果图和嵌套数据展开工具；`utils/__init__.py` 为包入口 |
| [_legacy_filler_selector.py](../src/pyWitnessAI/_legacy_filler_selector.py) | 历史 `CLIPFillerSelector` 的裁脸和返回值契约；当前 `FillerSelector` 使用通用检索 |
| [Old](../src/pyWitnessAI/Old) | 既有历史实现；当前检索、benchmark 和阵容接口由各自的核心模块提供 |
| `Local_Models` | 检测器资源；按可选视觉功能使用 |

上述既有视频与兼容接口本轮保留，未据此宣称其内部已完成全面剪枝。

## 包入口与研究复现

`__init__.py` 显式登记公开接口并按需加载所属模块；`API/cli.py` 提供 `parse`、`serve`；
`__main__.py` 支持 `python -m pyWitnessAI`。`_version.py` 由构建工具生成。

各项研究自行保存数据集适配、候选池与真值规则、分组、提示词、评分和验证配置，
并记录数据、模型、配置与代码版本。已删除研究的复现需要使用包含其完整实现的
历史 Git 版本；当前通用模块不包含旧研究的冻结协议。

详细使用见 [检索指南](FillerSelection.md)、[属性 benchmark](AttributeRetrievalBenchmark.md)、
[阵容与会话](LineupSession.md)。
