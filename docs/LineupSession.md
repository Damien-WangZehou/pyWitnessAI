# Lineup workflow and witness sessions

同一入口负责组合现有模块，不按描述生成、参考图生成再拆一组 API。
`create_lineup()` 创建一个阵容；`LineupSession` 保存人工反馈并推进下一轮；
`API/http.py` 把它们接到 HTTP，复用同一套生成逻辑。

## 一个入口，独立控制成员和生成条件

```python
from pyWitnessAI import create_lineup

# TP：传入的 suspect 是实验中已知的目标；另外生成五张 fillers。
lineup = create_lineup(
    suspect="./suspect.png",
    description="a man with short hair and blue eyes",
    target="present",
    output_dir="./generated_tp",
)
lineup.show()
lineup.save("./presentation_tp")
```

| 参数 | 含义 |
| --- | --- |
| `suspect` | 必须进入阵容的指定成员；不自动作为生成参考图 |
| `description` | 生成新人的文字条件 |
| `reference_image` | 生成参考图；不会因此自动进入阵容 |
| `description` + `reference_image` | 同时约束新候选，复用 FaceGenerator 的组合模式 |
| `filler_pool` | 已选好的候选目录、路径列表、ImageCatalog 或 manifest；直接交给 LineupBuilder |
| `target` | `present` / `absent` 为已知实验条件；`unknown` 为身份尚未知的候选画像 |
| `size` / `rows` | 阵容人数和展示行数，默认 6 / 2 |
| `similarity` | 对参考图的提示词相似程度要求，0–100，默认 75 |
| `parser` | 默认 `raw`，保留自由描述；可选 schema / lenient / strict |
| `generation_options` | 可选字典：model、size（生成图片尺寸）、quality、output_format |

`similarity` 不是实测人脸相似度、数值图文融合权重或身份一致保证。
描述与参考冲突时，目前生成提示词让明确描述优先。
生成接口与 `FaceGenerator` 一样接受 `generator_backend`，便于替换提供商或离线测试。
真实 OpenAI 生成需要安装 `pyWitnessAI[generation]` 并设置 `OPENAI_API_KEY`。

```python
# 参考图生成 fillers；suspect 原图独立保留在 TP 阵容中。
lineup = create_lineup(
    suspect="./suspect.png",
    reference_image="./suspect.png",
    similarity=60,
    target="present",
)

# TA：目标图仅作生成参考，不作为成员放入阵容。
lineup = create_lineup(reference_image="./suspect.png", target="absent")

# TA 也可以包含一个明确指定的 innocent suspect。
lineup = create_lineup(
    suspect="./innocent.png", reference_image="./target.png", target="absent",
)

# 图文组合；参考图仍仅是生成条件。
lineup = create_lineup(
    description="short hair and brown eyes", reference_image="./reference.png",
    target="unknown",
)
```

未传 `target` 时，有指定 suspect 默认 `present`，否则默认 `absent`，沿用
LineupBuilder 的规则。只有描述、没有目标原图，不能凭描述声明 TP；显式 TP
但缺少 suspect 会在生成前报错。未知身份的画像应显式使用 `unknown`，
会话会自动使用它。

已有候选库时，选择与构建仍分别复用各自的接口：

```python
from pyWitnessAI import FillerSelector

matches = FillerSelector("./face_pool").select("short hair").matches
lineup = create_lineup(suspect="./suspect.png", filler_pool=matches)
```

`filler_pool` 不能同时传 `description` 或 `reference_image`，避免悄悄忽略
生成条件。对现有照片按描述选择用 FillerSelector，按人脸距离选择用
select_by_similarity，已经选好的结果直接传入即可。

## 多轮会话

```python
from pyWitnessAI import LineupSession

session = LineupSession(
    "a man with short hair and blue eyes",
    output_dir="./sessions/example_01",  # 新目录或空目录
    size=6,
    max_rounds=10,
)
first = session.start()
first.lineup.show()

# 用户认为目标不在阵容中，但第 3 张最像。
second = session.respond(choice=3, confirmed=False, round_number=first.number)
second.lineup.show()

# 用户确认第二轮中的第 2 张画像。
session.respond(choice=2, confirmed=True, round_number=second.number)
print(session.status)          # confirmed
print(session.confirmed_face)  # 最终选中展示图片的 Path
```

每轮始终是 `size` 张图片。第一轮按原描述生成；后续轮次把上一轮选中图作为
reference，按同一原描述生成 `size - 1` 张新图，再插入上一轮选中的展示原图，
重新随机排位。选中图使用已保存的展示像素，不要求模型重新画出同一个人。
可传 `reference_image` 作为第一轮生成参考，它会复制到会话目录中。

所有会话阵容都是 `target="unknown"`、`role="candidate"`。
确认只表示用户确认该画像，不修改真值标签。AI 的 TP/TA 决策、视频识别流程
和 pyWitness TP/TA CSV 导出会拒绝这些未知身份阵容。

| 状态 | 行为 |
| --- | --- |
| `new` | 已校验配置，尚未成功展示第一轮 |
| `active` | 当前轮等待反馈，或已收反馈但下一轮生成失败、等待重试 |
| `confirmed` | 用户确认了某张图片，不再生成 |
| `limit_reached` | 最后一轮仍未确认，达到 max_rounds，不再生成 |
| `stopped` | 调用了 `session.stop()`，未作确认 |

`choice` 从 1 开始，`round_number` 必填。相同轮次的相同反馈可重复提交；
不会再生成一个已经成功完成的下一轮。相同轮次的不同反馈会报错，避免旧页面
或双击把选择错误地应用到新阵容。重复 `start()` 返回当前轮。

用户没有可用选择或想退出时调用 `stop()`，不必强制选一张脸。

## 保存与重试

会话自动保存 `session.json` 私有审计和每轮的独立展示包。
审计包含原描述、配置、成员顺序、来源和反馈；展示包只有中性编号与图片。
`session.to_dict()` 返回独立的审计快照，不能直接发给参与者。

```python
# 在同一配置/提供商下继续已保存的会话。
session = LineupSession.load("./sessions/example_01")
current = session.current
print(session.history, session.status)
```

反馈先保存，下一轮只有全部图片成功生成并完成展示导出后才加入历史。
生成失败会抛出异常；用原来的 `round_number`、`choice`、`confirmed` 重试。
同一个活跃实例保留待生成的 FaceGenerator，复用已成功生成的批次图片。
跨进程恢复会保留完成轮次和反馈，但未完成的生成批次会重新开始，可能重复
部分生成费用。`load()` 时需重新提供自定义 `generator_backend`；提供商实例
和密钥不存入 JSON。

Python 实例本身为同步接口；同一会话的调用需要串行执行。HTTP 层已提供会话锁。
HTTP 生成操作使用共享的调用控制层，可通过 `create_app(generation_runtime=...)`
配置并发、请求频率和有限重试；私有调用统计保存在 `output_dir/private/operations/`。
任务存储与实验服务的复用方式见 [生成运行控制](GenerationRuntime.md)。
`seed` 记录展示随机性，但不保证提供商生成像素可复现，也不保证多轮会收敛。

## HTTP：实验平台和 UI 使用同一套端点

```bash
python -m pip install "pyWitnessAI[api,generation]"
pyWitnessAI serve --image-root ./input_images --output-dir ./lineup_service
```

默认监听 `127.0.0.1:8000`，可通过 HOST / PORT / LOG_LEVEL 环境变量配置。
浏览器跨域接入时重复传 `--allow-origin`，例如 `--allow-origin http://localhost:3000`。
默认不允许跨域；同源的交互文档位于 `/docs`。

Python 嵌入使用明确的 HTTP 适配入口：

```python
from pyWitnessAI.API.http import create_app

app = create_app(
    output_dir="./lineup_service",
    image_root="./input_images",
    allowed_origins=["http://localhost:3000"],
)
```

`generator_backend` 可接收已配置的生成后端。命令行和 Python 嵌入使用同一个
`create_app`，无需分别配置生成或会话逻辑。

| 方法与路径 | 用途 |
| --- | --- |
| `POST /v1/lineups` | 生成或组装单个 TP、TA 或未知身份阵容 |
| `GET /v1/lineups/{id}/images/{position}` | 获取导出的成员图片 |
| `POST /v1/sessions` | 创建并生成会话第一轮 |
| `GET /v1/sessions/{id}` | 获取当前状态和展示；进程重启后按磁盘记录恢复 |
| `POST /v1/sessions/{id}/feedback` | 提交人工反馈并确认或生成下一轮 |
| `POST /v1/sessions/{id}/stop` | 停止会话 |
| `GET /v1/sessions/{id}/rounds/{number}/images/{position}` | 获取某轮展示图片 |

创建单个 TP 阵容：

```json
{
  "suspect": "suspect.png",
  "description": "short hair and blue eyes",
  "target": "present",
  "size": 6
}
```

创建会话：

```json
{"description": "short hair and blue eyes", "size": 6, "max_rounds": 10}
```

提交反馈：

```json
{"round_number": 1, "choice": 3, "confirmed": false}
```

响应包含中性的成员编号、`image_url` 和会话状态，隐藏角色、target、参考图、
原描述与服务器文件路径。客户端不能提交生成提供商或输出目录；这些由服务端配置。
输入图片路径相对 `image_root`，绝对路径也必须位于该目录内。
单轮资源目录中的私有 `audit.json` 保存实际乱序后的角色、target、seed 和来源，
供研究分析还原 position 与 suspect/filler 的对应；HTTP 不提供该审计文件的访问路由。

参数错误返回 422，反馈冲突返回 409，生成等运行失败返回 503。
创建请求每次分配新资源，不具有幂等性，不应自动重试；反馈可用相同参数重试。
第一轮创建失败的残留文件可由本地操作者检查，不会返回一个可指认的未完成阵容。
这是受信本地、单进程适配器；不要让多个服务进程同时写同一 output_dir。
公开部署的认证、作业队列和跨进程锁不属于此模块。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `LineupSession.py` | 一个单轮组合函数、一个会话控制类和一个轮次记录；独立于 HTTP |
| `LineupBuilder.py` | 固定成员、角色、顺序、展示和导出；不调用生成模型 |
| `FaceGenerator.py` | 图文生成计划、提供商调用和生成记录 |
| `API/http.py` | 通用 HTTP 参数、路径边界、响应、会话锁；调用上述核心 |
| `API/cli.py` | 解析 `parse`、`serve` 命令及启动选项；调用 HTTP 适配器 |
| `API/__init__.py` | 轻量命名空间；导入时不加载 FastAPI 或生成 SDK |

`pyWitnessAI serve` 直接启动本指南的阵容/会话服务。
研究专用服务应在研究项目中维护自己的入口、问卷协议、分组和验证测试，
通过本指南的 Python 接口组合生成、阵容和会话功能。
