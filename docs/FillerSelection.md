# 由简入繁的 filler 实验设计

需要生成图片时，使用 [FaceGenerator](FaceGenerator.md)：支持随机、描述、参考图和联合生成，并记录可复用的 manifest。旧 `FillerGenerator` 已删除，下面的示例使用当前接口。

`FillerSelector` 复用 `FaceSearch` 的图像池、模型后端、排序、展示和导出功能。
实验设计者分别选择文本解析、模型和评分策略；这三个选择互不绑定。
数据依次经过 `文本 → parser → ParsedDescription → 评分策略 → 模型后端 → 排名`。
parser 只执行一次；后续使用同一份文本、属性映射和审计标记。

## 1. 最简单的流程

以下例子使用 CLIP 评分并显示图片，先安装对应依赖：

```bash
python -m pip install "pyWitnessAI[clip,plot]"
```

基础安装 `python -m pip install pyWitnessAI` 只包含 NumPy、pandas 和 Pillow，
可用于文本解析及不依赖内置模型的自定义评分。可先检查 parser 输出，无需下载模型：

```bash
pyWitnessAI parse "Short hair; maybe blue eyes" --parser lenient
```

```python
from pyWitnessAI import FillerSelector

selector = FillerSelector("./face_pool")
result = selector.select("  Short hair,  MAYBE blue eyes. ")
result.show()
```

默认 `parser="raw"` 合并多余空格、将字母转为小写，然后把完整文本交给
CLIP。默认 `scoring="cosine"` 使用原始 cosine similarity，不进行属性拆分、
contrast 或池内标准化。返回评分最高的 5 张图片；池中不足 5 张时返回所有可用图片。
图像池应由实验设计者预先准备好；这里的 n 指图片数量，不自动按人物身份去重。

```python
result = selector.select("short hair", n=8)
faces = result.matches       # 入选图片的路径、元数据、分数和排名
all_faces = result.ranked    # 完整候选排序及 selected 标记
result.save_csv("./results/top_faces.csv", matches_only=True)
```

CLIP 后端使用 `clip` 可选依赖；`plot` 用于 `result.show()`。
源码开发可用 `python -m pip install -e ".[clip,plot]"` 安装。
默认模型沿用 `clip-ViT-L-14`。
首次使用模型时可能需要下载模型文件。可配置磁盘缓存，在多次实验间复用图像嵌入：

```python
selector = FillerSelector(
    "./face_pool",
    selector_kwargs={
        "model_name": "clip-ViT-B-32",
        "cache_dir": "./build/filler_cache",
    },
)
```

## 2. 独立选择 parser

| parser | 进入模型的文本 | 不确定属性 |
| --- | --- | --- |
| `"raw"`（默认） | 仅清理空格和大小写后的全文 | 原文照常传递，不进行属性解析 |
| `"lenient"` | 清理后的全文，同时提取属性记录 | 接受并记录 `uncertain=True` |
| `"strict"` | 排除带不确定标记的分句 | 属性仍保留，记录 `accepted=False` |

```python
selector = FillerSelector("./face_pool", parser="lenient")
result = selector.select("short hair; maybe blue eyes; a crescent scar on the cheek")
parsed = result.parsed_description

print(parsed.original_text)       # 未修改的输入，包含原始大小写和空格
print(parsed.normalized_text)     # 清理后的全文
print(parsed.model_text)          # 实际用于后续评分的描述
print(parsed.attributes)          # 属性、原描述分句、uncertain、accepted、polarity 等
```

宽松规则保留未注册的属性，不要求每个属性都能在词表中找到。
严格规则只影响送入模型的文本，不删除留档的描述或属性。
若严格规则排除了所有文本，会在模型调用前报错，避免返回没有描述依据的排序。

内置规则解析器针对英文，按分句确定不确定性范围。例如
`short hair; maybe blue eyes` 仅排除第二句；`maybe blue eyes and short hair`
将两个属性都视为不确定。这是可检查、可替换的规则，并非通用语言理解。
Raw parser 可传递任意语言；模型能否处理该语言由模型本身决定。

```python
from pyWitnessAI import RawDescriptionParser

selector = FillerSelector(
    "./face_pool",
    parser=RawDescriptionParser(lowercase=False),  # 保留大小写
)
```

## 3. 独立选择评分策略

```python
selector = FillerSelector(
    "./face_pool",
    parser="lenient",
    scoring="contrast",
)
result = selector.select("short hair; maybe blue eyes; a crescent scar on the cheek")
print(result.components)
plan = result.scoring_metadata["contrast_plan"]
```

Contrast 直接使用 parser 返回的属性，不重新解析 `model_text`。计算规则是：

- `accepted=False` 的属性保留在审计记录中，不参与评分。
- 有有效参照的属性：`positive_similarity - negative_similarity`。
- 没有有效参照的属性：保留 `positive_similarity`，记录为 `positive_only`。
- 已接受的不确定属性：保留正常权重和 `uncertain` 标记；不会因为 `maybe` 自动归零。
- 否定、相互冲突或未知描述：保留相关原文分句作正向输入，不凭空推断相反属性。
- 默认先在同一属性维度内加权平均，再对各维度等权平均。

`components` 包含每张图片的分量分数、是否有 contrast、属性维度、权重与不确定标记。
`parsed_description` 保留 parser 的完整决定，`contrast_plan` 保留具体评分提示词和证据。
通用 selector 使用不限定性别或族群的提示词模板。

`parser="raw", scoring="contrast"` 没有属性记录，因此把完整 `model_text`
作为一个 `positive_only` 分量评分，不会自动切换为属性解析。
只返回文本、未提供属性的自定义 parser 也采用这一行为。
需要按属性计算 contrast 时，显式选择 `parser="lenient"`、`"strict"`，
或返回属性记录的自定义 parser。

正向 cosine 和 contrast margin 的分数范围不同。若实验需要按池内分布调整尺度，
可显式选择标准化；也可以先按 feature family 汇总：

```python
from pyWitnessAI import ContrastScoring

selector = FillerSelector(
    "./face_pool",
    parser="lenient",
    scoring=ContrastScoring(
        standardise_components=True,
        aggregation="family_mean",   # 默认 dimension_mean
    ),
)
```

是否标准化是实验条件的一部分。常量分量在标准化后为零，因为它无法区分池内图片；
这与依据不确定词过滤属性是不同的规则。可以通过 `builder=` 提供自定义 contrast 词表和模板。
自定义 builder 的接口为 `build(parsed_description) -> ContrastPlan`；
`ContrastScoring` 传入完整 `ParsedDescription`，不再传入字符串。
若属性全被拒绝，或自定义 builder 未能为已有属性生成任何评分分量，会在模型调用前报错，
不会退回整段文本而恢复已排除的内容。
单独调用内置 `EyewitnessContrastPromptBuilder.build(text)` 仍可使用字符串：
它按指定 policy 解析一次；传入已解析对象时则直接使用其中的决定。

## 4. 更换模型，或增加评分策略

现有 `SelectorBackend`、`CallableSelectorBackend` 和 `register_selector_backend`
继续使用。自定义模型接收 `SelectorQuery`，返回每个候选的 `image_id` 和
有限数值 `selector_score`。它应遵守 query 的正向/contrast 模式。
更换模型不需要修改 parser；更换评分策略不需要修改图像池。

```python
# my_backend 是 SelectorBackend 实例，或 callable(manifest, query)。
selector = FillerSelector("./face_pool", selector=my_backend, parser="lenient")
```

增加实验自己的评分方式只需实现 `score(manifest, description, backend)`：

```python
from pyWitnessAI import ScoringResult, SelectorQuery

class MyScoring:
    def score(self, manifest, description, backend):
        query = SelectorQuery.from_text(description.model_text)
        scores = backend.score(manifest, query)
        # 在这里加入实验的计算规则，保留每个 image_id 的 selector_score。
        return ScoringResult(scores, metadata={"scoring_strategy": "my_scoring_v1"})

selector = FillerSelector("./face_pool", scoring=MyScoring())
```

自定义 parser 实现 `parse(text) -> ParsedDescription`，可使用另一套规则或语言模型，
无需继承内置类。例如，实验词表可将内置别名未覆盖的 `sky-coloured irises`
映射为蓝眼睛，contrast 会直接使用该映射：

```python
from pyWitnessAI import DescriptionAttribute, ParsedDescription

class StudyParser:
    def parse(self, text):
        normalized = " ".join(text.split()).lower()
        uncertain = normalized.startswith("maybe ")
        phrase = normalized.removeprefix("maybe ")
        attributes = ()
        if phrase == "sky-coloured irises":
            attributes = (DescriptionAttribute(
                text=normalized, family="eyes", dimension="hue", value="blue",
                uncertain=uncertain, accepted=True,
            ),)
        return ParsedDescription(text, normalized, normalized, attributes)

selector = FillerSelector("./face_pool", parser=StudyParser(), scoring="contrast")
result = selector.select("maybe sky-coloured irises")
```

自定义 parser 同时决定 `model_text` 和属性的 `accepted`、`uncertain`、`polarity`。
若排除属性，应同时从 `model_text` 中移除相应内容，使 cosine 和 Generator 也遵守这一决定。

传入显式 `SelectorQuery` 时，`FaceSearch`/`FillerSelector`
直接使用其中的提示词和评分模式，不再执行 parser 或自动评分策略。

## 5. 同一 parser 用于 Generator

```python
from pyWitnessAI import FaceGenerator

generator = FaceGenerator(
    description="short hair; maybe blue eyes; a scar on the cheek",
    n=5,
    parser="lenient",            # 生成器默认 schema；这里显式选择宽松文本策略
    output_dir="./generated",
)
print(generator.parsed_description)
print(generator.generation_prompts())
generator.generate(dry_run=True)  # 查看模型将收到的提示词，不调用生成服务
```

生成器使用选定 parser 的 `model_text`，并在外层添加统一的肖像风格与每张图片的身份变化指令。
需要实际调用已配置的生成后端时执行 `generator.generate()`。
`manifest.csv` 记录 parser、输入全文、可规范化的请求属性及实际提示词。
提示词预览和 `dry_run=True` 使用基础安装即可；默认 OpenAI 后端的实际生成需要
`python -m pip install "pyWitnessAI[generation]"`，并配置 `OPENAI_API_KEY`。

需要规范属性与固定肖像模板时，使用默认 `parser="schema"`，通过 `attribute_schema=`
提供属性定义。`FaceGenerator` 不接受 `schema=`；已有结构化描述可通过
`ParsedDescription.from_attributes(schema.attribute_values(), original_text=...)`
转换后传给 `description=`，不会再次解析；也可以使用文本或自定义 parser。
通用生成请求、后端基类与注册表从 `pyWitnessAI.face_generation_backend` 导入。
OpenAI 适配器位于 `pyWitnessAI.API.openai`，也可直接
`from pyWitnessAI import OpenAIImageBackend` 后配置并传给生成器。
图片展示可使用返回的 `GeneratedFace.path` 配合 Pillow 或 Notebook 的 `display`；
生成器不提供 `show()` 或 `summary()`。

## 6. 旧接口与复现

CLIP 编码器、索引和模型常量统一位于 `pyWitnessAI.clip`：

```python
from pyWitnessAI.clip import (
    ClipIndex,
    SentenceTransformerClipEncoder,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_MODEL_REVISION,
)
```

旧 `pyWitnessAI.cfd_clip_pilot` 整包及其研究流程已删除。
`MODEL_SNAPSHOT_HASH_VERSION` 和 `directory_content_sha256` 同样从 `pyWitnessAI.clip`
导入。请更新旧脚本和笔记本；记录源码指纹时统一使用 `src/pyWitnessAI/clip.py`。

Qualtrics 的 CFD 适配、固定候选池、冻结 profile、解析/评分契约、服务及验证
归 `studies/qualtrics_cfd_clip` 所有。研究入口直接保存研究实现并调用通用功能。
核心包不再提供 `pyWitnessAI.cfd`、`pyWitnessAI.experiment` 或 CFD 服务别名。
研究专用导入、依赖和启动方式见
[Qualtrics study 说明](../studies/qualtrics_cfd_clip/README.md)。

`FaceSearch` 新增了同样的 `parser`、`scoring` 参数，仍要求明确的
`top_k`、`min_score` 或 `policy`；`FillerSelector` 是默认选择前 5 张的便捷入口。

公开的 `EyewitnessContrastPromptBuilder` 默认使用通用人脸模板和宽松规则；
`policy="strict_v1"` 选择严格解析规则。要复现 Qualtrics 冻结实验，应使用研究
目录内的配置与 parser/scorer；仅切换解析 policy 不会恢复旧模板和完整评分契约。

历史 `pyWitnessAI.FillerSelector.CLIPFillerSelector` 导入路径仍可用，保留其最大人脸裁剪行为。

## 7. 保留的两个通用 benchmark

两个 benchmark 都保留在核心包，因为它们接受调用者提供的数据、属性和 selector，
不绑定 CFD 文件名、特定人群或固定研究阶段。它们直接复用 `ImageCatalog` 的图片
发现、`FaceSearch` 的评分归一化与排序，以及内部共享的排名图片展示。

| 接口 | 解决的问题 | 主要指标 |
| --- | --- | --- |
| `AttributeRetrievalBenchmark` | 对现成 manifest 或两个标签文件夹进行单个二元属性检索评估；支持显式正负提示词 | Precision/Recall@K、AP、ROC-AUC、阈值准确率与正样本排名 |
| `FillerSelectionBenchmark` | 从描述构建单属性或逐步累积的实验阶段；在 `GeneratedFaceDataset` 中匹配或生成每阶段的正负样本 | 每阶段前 N 项的正样本数、Precision/Recall@N、是否全部命中及排名诊断 |

`FillerSelectionBenchmark` 的 `n` 指每阶段各自需要的正样本和负样本数量，
默认也作为评估截断位置。负样本须通过 `contrast_overrides` 或显式 schema 定义；
选择负样本标签本身不会启用正负提示词差值评分，默认 CLIP 仍对阶段描述评分。
自定义评分可通过 `selector` 接入。

`AttributeRetrievalBenchmark` 默认令 K 等于正样本数，也可显式设置 `top_k`。
`score_mode="auto"` 有负提示词时采用正负相似度均值之差，否则只使用正向相似度。
两者按图片上的属性标签统计，不把同一身份的多张图片自动合并为一次命中。
具体示例见 [属性 benchmark 指南](AttributeRetrievalBenchmark.md)。
