# LineupBuilder

For generated TP/TA lineups and iterative witness feedback, see
[Lineup workflows and sessions](LineupSession.md). Neutral candidate sets use
`target="unknown"` with no designated suspect; every member has role `candidate`.
They support the same presentation and saving methods, but cannot be used for
TP/TA decisions or pyWitness TP/TA exports.

`LineupBuilder` 负责把可选的 suspect 和候选 fillers 构建成阵容；返回的 `Lineup`
保存本次成员、角色和排列，负责展示、统一风格及导出。生成和筛选先完成，再把结果交给
builder。构建阵容本身只需要基础安装，不启动生成服务或人脸识别模型。

```python
from pyWitnessAI import LineupBuilder

lineup = LineupBuilder(
    suspect="./123.jpg",
    filler_pool="./directory_filler",
).build()
lineup.show()
```

传入 suspect 时，默认构建 6 人、2 行、随机排列的 target-present 阵容。`rows=2` 表示两行，
6 人时自动得到三列；因此不使用含义相反的 `cols=2` 表示行数。

## 一次配置，一次构建

```python
lineup = LineupBuilder(
    suspect="./123.jpg",        # 也接受 PIL.Image.Image
    filler_pool="./fillers",    # 也接受路径列表、ImageCatalog、image_path 清单
    size=6,                    # 包括 suspect，即需要 5 个 fillers
    order="random",            # random / given
    rows=2,
    target="present",          # 省略时按 suspect 推断；也可显式指定 present / absent
    seed=42,
).build()
```

用构造参数表达完整输入，`build()` 返回完成后的阵容；无需在多次
`addSuspect()`、`addFillers()` 调用之间维护“尚未完整”的对象，也无需通过
`buildStrategy()` 选择另一套内部工作流。同一 builder 可以重复构建；固定 `seed`
用于复现随机选择和排列。已构建的阵容在重复 `show()` 时不会重新抽样或洗牌。

有 suspect 时需要 `size - 1` 个 fillers；`suspect=None` 时需要 `size` 个。
候选多于所需人数时，builder 均匀无放回抽取；`order` 只控制展示顺序：
`"random"` 随机排列所有成员，`"given"` 把 suspect（如有）放在第一位，再按输入顺序排列
选中的 fillers。按评分挑选好的结果应先截取到所需人数，再交给 builder，
这样随机展示不会改变选中的集合。候选不足时直接报错。

`target="present"` 将显式传入的 suspect 标为 `guilty_suspect`；
`target="absent"` 将它标为 `innocent_suspect`。若不指定任何嫌疑人，显式传入
`suspect=None`，阵容全部由 fillers 组成：

```python
lineup = LineupBuilder(
    suspect=None,
    filler_pool="./fillers",
    size=6,
    seed=42,
).build()
lineup.show()
```

此时需要 6 个 fillers，`target` 自动为 `"absent"`，`lineup.suspect_position` 为
`None`；有 suspect 时该属性返回从 1 开始的位置。省略 `target`（或传入 `None`）时，
有 suspect 则推断为 `"present"`，无 suspect 则推断为 `"absent"`。
`suspect=None, target="present"` 是矛盾输入，会直接报错。
有明确无辜嫌疑人的 target-absent 阵容仍通过 `suspect=...` 和 `target="absent"`
构建。角色由参数标注，不从文件名猜测，也不会把 filler 自动升为嫌疑人。

## image_pool 和 filler_pool 共用数据形式

这两个名字只描述当前工作流中的用途：`image_pool` 是候选图片，`filler_pool` 是
准备交给 builder 的候选 fillers。无需创建两个 pool 类，也无需为了衔接步骤复制目录。
需要复用元数据时使用已有 `ImageCatalog` 或包含 `image_path` 的 DataFrame；
简单任务直接传目录或路径列表。

`FillerSelector.select(...).matches` 和 `select_by_similarity(...)` 的 DataFrame
可直接作为 `filler_pool`。`FaceGenerator` 的结果通过 `generator.image_paths()`
或 `generator.to_manifest()` 进入下一步；后者同时保留生成条件等元数据。
选中行的 `image_id`、`face_distance`、请求描述等来源字段保存在
`member.metadata["source"]` 中，并转成可写入 JSON 的值。`image_manifest(pool)`
统一这些输入为清单，`image_paths(pool)` 复用同一规范化，只返回图片路径。

这条工作流中，每个核心文件对应一个明确职责：`LineupBuilder.py` 放构建器、
完成后的 `Lineup` 和小型 `LineupMember` 记录；`FillerSelector.py` 提供描述与人脸距离筛选；
`ImageCatalog.py` 统一图片清单、可用状态和来源数据；已有 `FaceGenerator.py`
负责生成请求与结果记录，`FaceStandardizer.py` 负责统一像素处理和质量记录。
组合流程只连接这些输入输出，不另建 pool 文件或策略层。

## 生成、筛选与组合

以下步骤共享同一个 `suspect`、描述和阵容大小。按实验需要选择其中一种准备 fillers 的方式，
最后统一调用 builder。

```python
suspect = "./123.jpg"
description = "short hair and blue eyes"
size = 6
```

### 1. 根据描述生成

```python
from pyWitnessAI import FaceGenerator

generator = FaceGenerator(description=description, n=size - 1, output_dir="./generated")
generator.generate()
fillers = generator.to_manifest()
```

实际生成需要配置对应后端；默认 OpenAI 后端需要 `pyWitnessAI[generation]` 和
`OPENAI_API_KEY`。生成器负责图片和生成记录，builder 负责阵容。

### 2. 根据参考图和相似程度生成

```python
generator = FaceGenerator(
    reference_image=suspect,
    description=description,    # 可省略，只按参考图生成
    similarity=75,
    n=size - 1,
    output_dir="./generated_reference",
)
generator.generate()
fillers = generator.to_manifest()
```

`similarity=75` 是生成提示中的相似程度要求，不是模型已经测量得到的相似度。
如果需要按实际人脸距离控制选择，生成较大的候选池后使用第 4 步的筛选函数。

### 3. 从图片池按描述筛选

```python
from pyWitnessAI import FillerSelector

fillers = FillerSelector("./image_pool", n=size - 1).select(description).matches
```

默认 CLIP 模型需要 `pyWitnessAI[clip]`。描述评分和人脸相似度是两个独立条件，
可以单独使用，也可以依次筛选。

### 4. 按与 suspect 的人脸距离筛选

```python
from pyWitnessAI import select_by_similarity

fillers = select_by_similarity(suspect, "./image_pool", n=size - 1, order="most")
# order="least" 选择最不相似的人脸。
print(fillers[["image_path", "face_distance"]])
```

默认识别工作流使用 `ImagesAI`，需要 `pyWitnessAI[vision]`。
`face_distance` 越小，模型判断的人脸越相似。这个 `order` 表示筛选方向；
builder 的 `order` 表示最终展示顺序，两者互不代替。

### 5. 组合筛选步骤

先按描述选 30 个候选，再从中选 5 个与 suspect 最相似的 fillers：

```python
candidates = FillerSelector("./image_pool", n=30).select(description).matches
fillers = select_by_similarity(suspect, candidates, n=size - 1, order="most")
```

先按描述生成 30 个候选，再从中选最不相似的 fillers：

```python
generator = FaceGenerator(description=description, n=30, output_dir="./generated_candidates")
generator.generate()
fillers = select_by_similarity(
    suspect, generator.to_manifest(), n=size - 1, order="least",
)
```

每一步都是普通输入和返回值，因此也可以先生成，再按描述筛选，最后按人脸距离筛选。
不需要为每种组合增加策略类或专用 pipeline 文件。准备好 fillers 后：

```python
lineup = LineupBuilder(suspect, fillers, size=size, seed=42).build()
```

## 统一风格并展示

```python
from pyWitnessAI import FaceStandardizer

lineup = lineup.standardize(FaceStandardizer())
lineup.show()
```

`standardize()` 将同一个 standardizer 用于所有成员，返回新的 `Lineup`，
保留成员顺序和角色，在 `member.metadata["standardization"]` 中保留质量记录。
任一成员处理失败时整次调用报错，不会静默删人。原阵容与源图片不变。此步骤需要
`pyWitnessAI[standardization]`；模型仅在实际处理时加载。
标准化处理的具体含义、质量标记与可选增强见 [FaceStandardizer](FaceStandardizer.md)。

```python
lineup.show(annotate_roles=True)  # 研究者检查角色
preview = lineup.render()       # 返回 PIL 图片，默认只显示位置编号
preview.save("./preview.png")
```

`show()` 默认隐藏角色并展示阵容；`show(display=False)` 可只取得渲染结果。
`len(lineup)` 是实际人数，`lineup.members` 按展示顺序保存成员，
`lineup.images` 和 `lineup.roles` 共用中性的成员 ID，方便视频分析使用。
`lineup.images` 每次返回独立的 sRGB RGB 图片副本，并把透明部分合成到白色背景；
渲染、保存和视频比较共用这套呈现像素。`member.image` 保留该成员本身的图片模式和像素，
供标准化使用，读取展示图片不会提前改变它。

## 保存与后续平台接入

```python
lineup.save("./lineup_001")      # 使用新的或空的目录
public_manifest = lineup.to_dict()
research_manifest = lineup.to_dict(include_roles=True)
```

保存的成员图片采用 `member_01.png` 等中性文件名，并输出 `lineup.png` 和
`lineup.json`。公开清单只包含中性 ID、相对图片文件名、位置和布局；角色、
target 条件、seed、源路径及研究元数据仅在 `to_dict(include_roles=True)` 中提供，
不会由 `save()` 自动写进公开目录。需要保存研究记录时，将该返回值另存到研究者自己的位置。
阵容成员与顺序可以直接交给未来的 Qualtrics 或 Gorilla 适配器，
当前接口不自动创建平台问卷或发送给被试。

`VideoLineupPipeline` 直接接收构建好的 `Lineup`；标准化后的阵容也使用同一接口。
完整示例见 [VideoLineupPipeline](VideoLineupPipeline.md)。

## 从旧 Lineup 迁移

旧 `pyWitnessAI.Lineup` 模块和目录加载构造函数已移除，不保留包装兼容层。
公开名称 `Lineup` 现在表示构建结果；常规使用只需导入 `LineupBuilder`。

| 旧用法 | 当前用法 |
| --- | --- |
| `Lineup(folder, guilty_suspect=...)` | `LineupBuilder(suspect=..., filler_pool=..., target="present").build()` |
| `Lineup(folder, innocent_suspect=...)` | `LineupBuilder(suspect=..., filler_pool=..., target="absent").build()` |
| `generate_lineup(lineup_size=6, ...)` | 在 builder 中指定 `size=6` 后调用 `build()` |
| `show_lineup(...)` | 已构建阵容的 `show(...)` |
| 从文件名推断角色 | 显式 `suspect` 和 `target` |

`VideoAI.LineupLoader` 属于历史视频复现接口，和本页的构建流程不同。
