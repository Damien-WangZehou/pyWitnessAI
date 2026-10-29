# 四条件 AI 生图 lineup 问卷

导入文件：`AI_Generated_Lineup_4_Conditions.qsf`。原始 QSF 保留不变。

**QSF 已生成，但生成服务尚未部署，`LINEUP_API_BASE_URL` 当前为空。** 导入问卷后可检查全部流程；实际生成参与者的 lineup 前，需运行配套服务、提供 HTTPS 地址，并将该地址填入 Survey Flow。不能在 Qualtrics 中填写本机 `D:/...` 路径来代替服务地址。

## 实验与数据

在 verbal description 提交之后、五分钟 Tetris 之前，Qualtrics Randomizer 从下列四个 Embedded Data 元素中均衡随机呈现一个。均衡分配按进入随机器的次数计数，不保证四组最终有效完成样本数完全相同。

| condition | targetLineup | 六人 lineup |
| --- | --- | --- |
| mixingFaces | targetAbsent | 6 张参考图混合生成的 filler |
| mixingFaces | targetPresent | 固定嫌疑人 + 5 张参考图混合生成的 filler |
| noMixing | targetAbsent | 6 张仅描述生成的 filler |
| noMixing | targetPresent | 固定嫌疑人 + 5 张仅描述生成的 filler |

| 导出字段 | 取值 / 含义 |
| --- | --- |
| responseType | `suspectId`、`fillerId`、`rejectId` |
| targetLineup | `targetPresent`、`targetAbsent` |
| condition | `mixingFaces`、`noMixing` |
| confidence | 数值字符串 `0`–`100`，间隔 10；技术失败为空 |
| SelectedPosition / SelectedImage | 最终选择的位置与图片 ID；拒绝时位置为 `NotPresent` |
| suspectPosition | TP 的嫌疑人位置；TA 为空 |
| lineup1…lineup6 / …Url / …Role | 实际呈现顺序、图片地址和 `suspect` / `filler` 角色 |
| FirstResponseType / FirstSelectedPosition / FirstSelectedImage | 首次选择，用于对照修改后的最终选择 |
| LineupDecisionRTms / LineupDecisionRTs | 所有六张图片加载并同时呈现后，到第一次选择的时间 |
| LineupFinalDecisionRTms | 从六张图片呈现到提交 lineup 页的时间 |
| lineupLoadTimeMs | lineup 页打开到全部图片呈现的时间，包含等待生图 |
| FirstClickonLineupPage / LineupTiming | Qualtrics 原生页面计时，可能包含准备时间；分析识别 RT 时优先使用以上独立计时 |
| lineupID / generationMetadata | 服务任务 ID、模型、reference 文件名、prompt hash 和标准化参数 |
| technicalFailure / technicalFailureReason | 技术失败标志和原因；失败不是正式识别类别，`responseType` 留空 |

没有 designated 人物、没有 `designatedId`。TP 点击嫌疑人记录 `suspectId`；任一组点击 filler 记录 `fillerId`；任一组选择 Not Present 记录 `rejectId`。

保留原有信息页、同意题、视频、描述、Tetris、注意题、人口学题和完成码。Tetris 计时设为 300 秒。lineup 选择后在独立页面询问信心；技术失败分支在信心页之前退出，避免替参与者填入 0%。原有注意题错误即结束的逻辑保留。

## 生图方法

两组都直接使用参与者的描述，`RawDescriptionParser(lowercase=False)`、`conversation_model=None`、`prompt_processor=None`，不调用 `prompt_processing`。

- `mixingFaces`：从 `D:/PhD/Software/pyWitnessAI/data/ReferenceFaces_CFD_WhiteMale` 不放回随机取 5 或 6 张参考图，每张参考图分别生成一张 filler（`images.edit`）。
- `noMixing`：不传参考图，按描述生成 5 或 6 张 filler（`images.generate`）。
- 固定嫌疑人来自本目录 `perp_refer_to_AI_generated_image.png`，仅在 TP 中加入；不作为 filler 生图参考。
- 使用 notebook 的图像参数：1024×1024、high、PNG。呈现前全部成员（包括嫌疑人）经过相同 `FaceStandardizer` 中心裁切，统一为 RGB PNG、256×320；再随机打乱位置。

提示词遵循两个 notebook 的对应分支：mixing 的开头为 `Make a photo of a man based on this image, with the following features: ...`；noMixing 的开头为 `Make a photo of an under average-looking man, with facial features and hair style completely different from each other, but matching the following description: ...`。两者附加相同的肩部以上、灰色背景、正面注视相机要求。因此当前提示词除了参考图操纵外，也保留了 notebook 原有的措辞差异。

统一裁切不等于背景分割或感知外观匹配；正式收集前需要预试检查生成质量、嫌疑人是否突显、视频/Tetris 外链和四组数据导出。未调用付费 API 生成测试图片，也未在真实 Qualtrics 账号中导入或发布。

## 启动服务并配置问卷

在仓库根目录 `D:/PhD/Software/pyWitnessAI` 使用 Python 3.10 或 3.11：

```powershell
python -m pip install -e ".[generation]"
python -m pip install -r studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference/requirements.txt

$env:OPENAI_API_KEY = "<你的服务端 API key>"
$env:ALLOWED_ORIGINS = "https://bristolexppsych.eu.qualtrics.com"
# 如果需要，显式指定账号可用、同时支持 generate 和 edit 的图像模型：
$env:IMAGE_MODEL = "<账号可用的图像模型名>"

python -m uvicorn lineup_service:app --app-dir studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference --host 127.0.0.1 --port 8000 --workers 1
```

如果 PowerShell 当前已位于本研究子目录，启动命令改为：

```powershell
python -m uvicorn lineup_service:app --host 127.0.0.1 --port 8000 --workers 1
```

启动后访问 `http://127.0.0.1:8000/health`（或兼容入口 `/healthz`），应返回 `{"status":"ok","study_version":"ai_generated_lineup_2x2_v1"}`。根路径 `/` 返回服务入口信息，`/docs` 提供 API 文档。这些 GET 请求不会生成图片或调用付费 API。浏览器自动请求 `/favicon.ico` 返回 404 不影响服务。修改服务代码后需重启 uvicorn 才会生效。

API key 仅保存在服务端环境；不要填入 QSF 或前端 JavaScript。`IMAGE_MODEL` 未设置时沿用项目当前默认模型；本次没有验证该默认模型在你的账号中是否可用。运行上例前请将尖括号占位符替换成真实配置。

使用你管理的 HTTPS 反向代理/隧道，将服务的 `/health`、`/v1/lineups`、`/v1/lineups/{id}` 和 `/media/...` 原样转发。此实现是单个研究服务进程，必须使用 **一个 worker**，不要多进程或多实例共享同一个运行目录。公开入口应由研究者控制并限制用量；单纯的 CORS 设置不提供调用者身份验证。

1. 在 Qualtrics 创建新 survey，导入 `AI_Generated_Lineup_4_Conditions.qsf`。
2. 在 Survey Flow 最上方 Embedded Data 中，将 `LINEUP_API_BASE_URL` 设置为真实 HTTPS 服务地址，例如 `https://<你的生图服务域名>`；不带 `/v1/lineups`。其余分组字段由随机器设置。
3. 核对 `ALLOWED_ORIGINS` 与浏览器中问卷页面的真实 origin 一致。多个 origin 用逗号分隔。
4. 先在 Preview 检查四个分组及导出数据，再发布。要确定检查每组，可在临时测试副本中将 Randomizer 替换为对应组的 Embedded Data；正式版恢复均衡随机器。

也可以在导入前将地址写入文件：

```powershell
python studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference/build_qsf.py --api-base-url "https://<你的生图服务域名>"
```

Tetris 页提交一次任务来提前生成；lineup 页复用同一任务并轮询，不把一个长时间的图像生成请求挂在浏览器上。Retry 重用相同参与者、描述和条件，不自动重新生图。生成失败/中断后保留错误状态；需研究者检查，不能靠参与者重试反复产生费用。网络或图片加载失败可以 Retry 重新获取同一结果。

服务环境变量：`REFERENCE_DIR`、`SUSPECT_PATH` 可覆盖本机默认资源路径；`LINEUP_STORAGE_DIR` 设置持久存储目录；`MAX_CONCURRENT_JOBS` 默认 2，`MAX_PENDING_JOBS` 默认 40。上线时应按预计参与人数、模型限额和真实生成时长调整并预试。

`runtime/private/` 保存原始生成图、prompt、描述和研究记录；`runtime/jobs.sqlite3` 保存任务；仅 `runtime/media/` 对外提供标准化图片。运行目录已在本目录 `.gitignore` 排除。保留任务数据库与媒体，才能重现参与者实际看到的 lineup；迁移或重启服务时应保留该目录。

## 本地验证

```powershell
python -m pytest studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference/tests/test_qsf.py studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference/tests/test_lineup_service.py -q
node --test studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference/tests/test_frontend.cjs
```

测试使用假图像生成器，不发送真实生图请求。覆盖随机器/引用/字段、四组人数与角色、幂等重试和持久化、CORS/媒体边界，以及前端选择与计时逻辑。

Qualtrics 文档：[导入 QSF](https://www.qualtrics.com/support/survey-platform/survey-module/survey-tools/import-and-export-surveys/)、[Randomizer](https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/standard-elements/randomizer/)、[Embedded Data](https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/standard-elements/embedded-data/)、[JavaScript](https://www.qualtrics.com/support/survey-platform/survey-module/question-options/add-javascript/)。问卷依赖账号已启用自定义 JavaScript。
