# 生成任务与调用控制

两套 HTTP 入口保留各自的业务契约，共享生成调用控制。实验条件留在
`studies/`，公共模块只依赖 Python 标准库，不在导入时加载 OpenAI SDK。

| 模块 | 职责 |
| --- | --- |
| `job_runtime.py` | SQLite 任务记录、幂等提交、队列容量、后台执行、重启恢复、排队及执行耗时 |
| `generation_runtime.py` | 图片/文本独立并发上限、请求间隔、有限重试、操作截止时间、客户端复用、私有用量记录 |
| `FaceGenerator.py` | 提示词规划、可选图片并行、按编号整理结果、单线程写 manifest、复用成功图片 |
| `API/openai.py`、`API/openai_prompt_processor.py` | 服务商请求转换；接入当前 runtime 并关闭 SDK 内部重试 |
| `API/http.py` | 同步阵容与会话接口；使用公共 runtime，继续使用原来的会话文件 |
| 实验 `lineup_service.py` | 固定实验条件与标准化流程；使用公共任务存储和 runtime |

## 通用 API

默认 HTTP 接口仍同步返回阵容，创建请求仍每次分配新资源。没有把
`POST /v1/lineups` 改为排队接口，也没有把会话 JSON 搬入 SQLite。

```python
from pyWitnessAI.API.http import create_app
from pyWitnessAI.generation_runtime import CallPolicy, GenerationRuntime

runtime = GenerationRuntime(
    image_policy=CallPolicy(
        max_concurrent=4,
        requests_per_minute=14,
        max_retries=2,
    ),
    text_policy=CallPolicy(max_concurrent=2, max_retries=2),
)
app = create_app(
    image_root="./inputs",
    output_dir="./service",
    generation_runtime=runtime,
)
```

注入的 runtime 由调用方拥有；服务彻底停止后调用 `runtime.close()`。
不传入时，API 创建并关闭自己的 runtime，默认每类最多两个同时调用、
无请求频率限制、不自动重试。生成调用统计保存在
`output_dir/private/operations/<operation-id>.json`，没有对应的公开下载路由。

OpenAI 适配器自动使用操作作用域中的 runtime。自定义后端继续遵守原来的
`generate(request)` 契约；需要限制其外部调用时，可显式使用
`current_runtime().call(kind, model, invoke)`。`invoke(timeout_seconds)` 必须将
给定超时传给实际网络客户端。任意自定义后端不会被自动重试。

## Python 生成

```python
from pyWitnessAI import FaceGenerator
from pyWitnessAI.generation_runtime import CallPolicy, GenerationRuntime

runtime = GenerationRuntime(image_policy=CallPolicy(
    max_concurrent=4, requests_per_minute=14, max_retries=2,
))
try:
    with runtime.scope("./private/runtime.json", timeout_seconds=300):
        faces = FaceGenerator(
            description="A man with short hair.",
            n=6,
            max_workers=2,
            conversation_model="gpt-6.1-sol",
        ).generate()
finally:
    runtime.close()
```

`FaceGenerator.max_workers` 默认是 `1`，保持原来的串行行为。设置为 `2`
后先完成原有提示词规划，再并行请求图片。结果及 manifest 按原编号写入，
避免多个线程同时修改一个 CSV。部分调用失败时，在途调用仍可能成功；
所有成功结果会被记录，同一个生成器再次执行时只补缺失的图片。
调用方必须确认自定义生成后端支持并发，才启用 `max_workers > 1`。

## 实验服务

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `MAX_CONCURRENT_JOBS` | `2` | 同时执行的整组阵容任务 |
| `MAX_PENDING_JOBS` | 跟随 `MAX_CONCURRENT_JOBS` | 排队和生成中的任务总数；默认避免长队 |
| `IMAGES_PER_JOB` | `2` | 每个阵容内最多并行的图片生成流程 |
| `MAX_CONCURRENT_IMAGES` | `4` | 全服务同时进行的图片 API 调用上限 |
| `IMAGE_REQUESTS_PER_MINUTE` | `14` | 图片请求平滑发送上限；一次调用请求一张图片 |
| `PROVIDER_MAX_RETRIES` | `2` | 明确临时限流或过载错误的额外尝试上限 |
| `GENERATION_TIMEOUT_SECONDS` | `300` | 从任务提交起算的生成预算，包含排队 |

14 次/分钟是为已知 20 IPM 额度保留余量的实验默认值，不会自动查询账户。
修改模型、账户额度或共享服务流量时需要重新配置。并发任务仍从两组起步，
不代表已证明四组或八组被试能在五分钟内完成。

mixing 条件仍按各自参考图分别规划；noMixing 仍一次规划整组提示词。
并发优化没有合并或修改这两种提示词流程，也没有改变模型、图片质量、
参考图抽样、展示顺序或标准化参数。

服务启动时才创建 JobManager、打开数据库并恢复任务；仅导入模块或
构造 app 不会接触实际任务数据库。重启服务后新代码与配置才生效。

## 任务持久化与恢复

`SQLiteJobStore` 与原有 `jobs.sqlite3` 的表和字段兼容，既有记录无需重建。
保留原来的物理字段名 `lineup_id`、`participant_hash`，公共 Python 接口使用
任意 JSON 输入与幂等键。每次事务都显式关闭连接。

相同键和相同参数取回已有结果，即使队列已满；相同键、不同参数返回冲突。
只有 `queued` 状态能被工作线程领取，只有 `generating` 状态能被正常完成。
恢复为中断错误后，旧线程的迟到结果不能覆盖该决定。

新完成任务增加 `started_at`、`finished_at` 和 `timing`，其中包含
`queue_seconds`、`run_seconds`、`total_seconds`。历史记录没有这些字段时
不会倒推或补造分阶段耗时。

整个部署仍要求一个进程拥有一个数据库。运行中的任务在重启后标记为
`generation_interrupted`，不自动重放付费请求；尚未开始的任务可以恢复。
不要让两个存活服务或多个 Uvicorn worker 共同恢复同一个数据库。

## 用量、重试与适用边界

实验任务的 `runtime/private/<lineup-id>/runtime.json` 保存：

- 每次图片或文本调用的模型、尝试次数、等待和网络调用耗时。
- API 实际返回的 `input_tokens`、`output_tokens`、`total_tokens`。
- 可用时记录请求 ID；失败记录异常类型和 HTTP 状态。

记录不复制提示词、参考图内容或原始错误正文。未返回 usage 的失败请求
无法据此计算完整计费量；多个任务并行时，逐次调用耗时之和也不是整组
墙钟耗时。浏览器下载和显示时间不在这些服务端统计中。

重试只处理明确的临时拒绝：`429/rate_limit_exceeded`、`429/slow_down`
和 `503/server_is_overloaded`。遵守 `Retry-After`；总重试等待最多 30 秒，
同时受整组截止时间约束。余额不足、未知 500、网络断连和超时不会被此层
自动重试。受控调用设置 SDK `max_retries=0`，避免两层重试相乘。客户端
调用超时取单次配置与整组剩余时间的较小值。

截止时间用于停止后续发送和约束 SDK 网络超时，无法撤销上游已接受的请求，
也无法硬中断图片标准化或任意自定义代码。因此它不是五分钟交付保证。

当前控制器限制请求频率和在途数量，**不估算或强制执行 TPM/TPD**。
根据实测 usage、账户剩余额度和其他共享流量安排每日人数；达到日额度
时应暂停招募。限流实例只在进程内共享；跨进程部署需要外部协调层。

重试依据：[OpenAI rate limits](https://developers.openai.com/api/docs/guides/rate-limits#retrying-with-exponential-backoff)。
