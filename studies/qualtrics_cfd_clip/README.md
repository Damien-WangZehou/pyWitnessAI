# Qualtrics × CFD × CLIP（无分组、全 filler lineup）

本目录保存 Qualtrics CFD/CLIP 研究协议、问卷、服务和验证流程。通用描述解析、CLIP 编码、评分和文件校验由 `pyWitnessAI` 提供。研究代码与协议文件不随核心库发布，需要使用完整仓库；CFD 数据需另行在本地准备。

## 已实现的实验流程

1. Participant information（已按新流程更新）
2. Consent：保留原问卷的三项确认，并继续要求三项全部勾选
3. 观看原 mock-crime video
4. 新增强制回答的多行 `verbal description`（`QID59`）
5. 五分钟 Tetris（`QID24` 仍为 300 秒）
   - Tetris 开始时，浏览器把描述发给 CLIP 服务；这五分钟同时用于模型检索和图片预载
6. 原 lineup instruction
7. 一个固定条件的六人 lineup（`QID60`）
   - 固定 `targetLineup=targetAbsent`
   - 六张候选按参与者和描述生成确定性随机显示顺序，CLIP rank 不等于屏幕位置
   - 1–6 为正式 Qualtrics 单选项；第 7 项为 `Not Present`
   - 连续三次技术失败后才会显示第 8 项；participant 可结束研究、进入技术故障 debrief 并取得 completion code
8. 原 0%–100% confidence、lineup timing、attention check
9. 原 demographics、debrief、completion code 和 comments

可导入文件：`Can_AI_select_CFD_CLIP_fillers_no_groups.qsf`。

生成脚本：`build_qualtrics_qsf.py`。它删除原来的六条件随机化和不再使用的问题，生成正常流程所需问题及一个仅在连续技术失败后出现的退出/debrief 问题：

```powershell
python studies\qualtrics_cfd_clip\build_qualtrics_qsf.py `
  --source "studies\qualtrics_cfd_clip\Can_AI_select_CFD_CLIP_fillers_no_groups.qsf" `
  --output "build\qualtrics_cfd_clip\rebuilt.qsf"
```

Qualtrics 官方不建议手工改 QSF 内部 JSON；因此应把生成文件导入一个测试项目，在 Survey Builder / Survey Flow 中复核，再从 Qualtrics 重新导出保存。参见 [Import & Export Surveys](https://www.qualtrics.com/support/survey-platform/survey-module/survey-tools/import-and-export-surveys/)。

## CLIP 服务做什么

QSF 不能运行 Python/CLIP，也不包含 CFD 图片。`clip_lineup_service.py` 直接提供本研究的 FastAPI 路由、配置和并发控制：

- 默认加载固定 revision 的 `clip-ViT-L-14` 与独立的 93 人 White–Male–Neutral（WM-N）索引；不会复用旧的 B/32 向量；
- 服务端用 strict parser 解析描述，但 live profile 只批准离线 pilot 中方向一致、语义清楚的六个维度：`eyes.size`、`nose.length`、`nose.width`、`lips.fullness`、`chin.length`、`forehead.height`；
- 每个获批维度计算 `cos(image, positive) - mean(cos(image, concrete alternatives))`，在固定 93 人池内逐维 z-standardise，再对维度等权平均；
- `partial` plan 只使用已批准维度；零个批准维度时才明确使用 `raw_positive` fallback。模型、索引或打分错误不会静默 fallback；
- raw fallback 会用模型自身 tokenizer 将超过 77-token 窗口的描述切成安全 chunks，再按 token 数加权平均；
- 取 selector top 6。`selector_score` 在主路径是 z-score component 的均值，不是 CLIP cosine；
- profile 固定模型 revision、模型 snapshot 内容、index/candidate embedding、pool、实际展示图片的路径与 bytes、schema/parser/scorer 版本和 rule allowlist，任一不一致都会拒绝启动；每次发送图片前还会再次核对该文件的内容 hash；
- 使用绑定当前 profile 与 media digest 的 HMAC 生成可过期图片 URL，不暴露本地文件路径；部署切换 profile/media 后旧 URL 会失效；
- 同一 participant、描述与 scoring contract 会得到相同人选和显示顺序；
- 服务本身不把 raw description 写入文件或普通 access log body。

这里的 live profile 状态是 **pilot-supported; independent held-out validation pending**。它比把完整探索性 schema 接入线上更保守，但不能据此宣称已证明真实 eyewitness accuracy 提升。

返回示例：

```json
{
  "lineup_id": "c1d2...",
  "model_name": "clip-ViT-L-14",
  "model_revision": "1b4b2e899178706d1b9905460ac21de1e0ba86a5",
  "model_snapshot_sha256": "fc4c...",
  "model_snapshot_match_verified": true,
  "index_sha256": "0491...",
  "candidate_embedding_sha256": "0c0b...",
  "media_manifest_sha256": "dd63...",
  "pool_sha256": "...",
  "query_sha256": "...",
  "effective_query_sha256": "...",
  "query_strategy": "within_pool_z_equal_dimension_mean_v1",
  "model_context_tokens": 77,
  "prompt_version": "cfd_wm_neutral_v1",
  "pool_version": "CFD3_WM_N_v2_L14",
  "selection_algorithm": "contrast_profile_rank_distinct_then_seeded_shuffle_v4",
  "pool_size": 93,
  "study_version": "cfd_clip_contrastive_l14_v3",
  "scoring_contract_version": "cfd_clip_attribute_contrastive_v3",
  "selection_profile_id": "cfd_wm_n_pn_l14_pilot_supported_v2",
  "scoring_method": "attribute_contrastive",
  "effective_method": "attribute_contrastive",
  "score_space": "within_pool_zscore_equal_dimension_mean",
  "plan_status": "partial",
  "contrast_pair_count": 3,
  "contrast_dimension_count": 3,
  "accepted_rule_ids": ["strict_v1.eyes.size.large", "..."],
  "accepted_dimensions": ["eyes.size", "..."],
  "fallback_used": false,
  "items": [
    {
      "position": 1,
      "image_id": "CFD-WM-...-N",
      "target_id": "WM-...",
      "selector_rank": 3,
      "selector_score": 1.14,
      "image_url": "/v1/images/CFD-WM-...-N?expires=...&signature=..."
    }
  ]
}
```

API 暂时也返回 `clip_rank` / `clip_score` 兼容别名，但新分析应使用 `selector_rank` / `selector_score`。

## 代码位置与导入

核心包只保留通用能力。本目录直接调用核心函数，拥有本研究的全部固定规则：

| 文件 | 存在理由 |
| --- | --- |
| `cfd.py` | CFD 文件名、Excel 表头和元数据适配；图片扫描复用 `ImageCatalog` |
| `profile.py`、`contrast_profile_live_v1.json` | 单一冻结协议及其校验；固定模型、候选池、媒体和评分契约 |
| `description.py`、`scoring.py` | 对通用解析与评分引擎的小型适配，维持冻结研究版本 |
| `selector.py` | WM-N 候选池、获批规则、确定性排列、结果审计与图片签名 |
| `media.py` | CFD 展示图片路径与内容完整性检查 |
| `clip_lineup_service.py` | 一个服务实现与入口，包含环境配置、路由、缓存和并发控制 |
| `build_live_l14_index.py` | 直接构建并验证 93 人 live index |
| `validate_live_profile.py` | 本研究 controlled-query 验证和输出 |
| `bootstrap_model.py`、`model_download_manifest.json` | 固定模型文件清单、下载与快照验证；文件 hash 复用核心函数 |
| `build_qualtrics_qsf.py`、QSF | 问卷流程、JavaScript 与可导入的冻结问卷 |
| `requirements.txt`、`deployment.env.example` | 研究环境约束和服务参数示例 |
| `tests/` | 固定研究行为与问卷回归测试 |

```python
from studies.qualtrics_cfd_clip.cfd import build_cfd_manifest
from studies.qualtrics_cfd_clip.selector import CfdWhiteMaleLineupSelector
from studies.qualtrics_cfd_clip.profile import LiveSelectionProfile
```

通用算法来自 `pyWitnessAI.clip`、`EyewitnessContrastPrompts`、
`ContrastiveClipScoring`、`ImageCatalog` 和 `model_files`。
本目录不再提供核心包中的研究别名或重复转发入口。

## 构建与验证 ViT-L-14 live index

`data/` 是被 Git 忽略的本地数据目录，不随 clone、源码分发包或 wheel 提供。
使用者需自行准备 CFD 图片和 norming Excel。默认放置位置为：

| 输入 | 本地路径 |
| --- | --- |
| CFD 图片目录 | `data/CFD Version 3.0/Images` |
| Norming workbook | `data/CFD Version 3.0/CFD 3.0 Norming Data and Codebook.xlsx` |

也可通过 `--image-root` 和 `--metadata` 指定现有本地数据路径；启动服务时将
`CFD_IMAGE_ROOT` 指向同一图片目录。数据不由模型准备命令下载。
研究目录保留建索引脚本、冻结配置与本说明，索引和验证结果在本地生成。

不要覆盖旧 B/32 index。先取得固定模型 snapshot，再运行：

```powershell
python -m studies.qualtrics_cfd_clip.build_live_l14_index `
  --image-root "data/CFD Version 3.0/Images" `
  --metadata "data/CFD Version 3.0/CFD 3.0 Norming Data and Codebook.xlsx" `
  --local-model-path "$env:USERPROFILE\.cache\huggingface\hub\models--sentence-transformers--clip-ViT-L-14\snapshots\1b4b2e899178706d1b9905460ac21de1e0ba86a5" `
  --show-progress
```

省略 `--manifest` 时，从 CFD 图片和 norming workbook 直接生成 manifest；不再依赖 pilot 输出。模型名称与 revision 从 profile 读取，CLI 必须指定本地模型快照，全部 hash 通过校验后才保存。

当前生成的 live index 位于 `build/qualtrics_cfd_clip/clip_index_vit_l_14`，为 93×768。脚本会记录并打印 serialized index、实际 scoring array、pool identity、模型 snapshot 内容及展示媒体 manifest/content hashes；它们必须与 `contrast_profile_live_v1.json` 一致。模型目录名必须等于 frozen revision，且目录内容 hash 也必须相符。若重新编码、替换图片或模型文件后任一 hash 改变，必须把它当作新的 profile/version 并重做验证，不能临时绕过启动检查。

对 93 个 metadata-derived controlled descriptions 重跑 exploratory 检查。用 `--queries` 显式提供研究 CSV；需要 `query_id`、`target_id`、`description`、`gender`、`race`、`stage`，固定选取 `male` / `White` / `15_add_teeth` 行：

该 CSV 也是需自行准备的研究数据；clone 不包含历史 controlled descriptions。

```powershell
python -m studies.qualtrics_cfd_clip.validate_live_profile `
  --queries "path/to/controlled_queries.csv" `
  --local-model-path "$env:USERPROFILE\.cache\huggingface\hub\models--sentence-transformers--clip-ViT-L-14\snapshots\1b4b2e899178706d1b9905460ac21de1e0ba86a5"
```

本次 L/14 结果：raw-positive Recall@6 为 7/93（7.5%），live effective 为 13/93（14.0%）；12 gains、6 losses，exact McNemar `p=.238`。这是探索性 proxy 结果，不显著，也不是独立 eyewitness validation；正式收数前仍需人工 parser validation 和 held-out calibration。

## 本地启动服务

通过 Python 直接启动服务。图片和已冻结的 L/14 index 需单独提供；
模型准备命令负责下载或复用固定 revision 的快照，并校验完整内容。
部署编排配置由模块外的独立项目维护。

在项目根目录：

```powershell
python -m pip install -e ".[clip,excel,api]" -r studies/qualtrics_cfd_clip/requirements.txt
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

把最后一条命令生成的值设为 secret，然后启动：

`deployment.env.example` 列出了完整部署变量；真实 secret 应放入部署平台的 secret store，不要写回 QSF 或仓库。

```powershell
$env:LINEUP_SIGNING_SECRET = "粘贴随机 secret"
$env:QUALTRICS_ALLOWED_ORIGINS = "https://bristolexppsych.eu.qualtrics.com"
$env:CFD_CLIP_LOCAL_MODEL_PATH = "$PWD/models/clip-ViT-L-14/1b4b2e899178706d1b9905460ac21de1e0ba86a5"
python -m studies.qualtrics_cfd_clip.bootstrap_model
if ($LASTEXITCODE -ne 0) { throw "模型快照校验失败，停止启动服务。" }
uvicorn studies.qualtrics_cfd_clip.clip_lineup_service:app --host 127.0.0.1 --port 8000 --no-access-log
```

默认读取：

- index：`build/qualtrics_cfd_clip/clip_index_vit_l_14`
- CFD 图片：`data/CFD Version 3.0/Images`
- 模型：`clip-ViT-L-14`
- model revision：`1b4b2e899178706d1b9905460ac21de1e0ba86a5`
- frozen local snapshot：必须通过 `CFD_CLIP_LOCAL_MODEL_PATH` 指定；目录 basename 和完整内容 hash 都会校验
- live profile：`studies/qualtrics_cfd_clip/contrast_profile_live_v1.json`
- 图片 URL 有效期：7200 秒
- 同一时间最多执行 1 个模型检索；同步 CLIP 推理在线程池中运行，不阻塞 FastAPI event loop
- HTTP 等待窗口默认 25 秒（短于浏览器的 30 秒）；超时不会假装终止无法安全取消的模型线程，而是保留同一 in-flight task，重复请求复用它，其他新 query 暂时返回 503
- 最近 512 个确定性选择结果保存在进程内 LRU，便于超时后的幂等重试；服务重启后可按相同输入重新计算

可用这些环境变量覆盖：

- `CFD_REPOSITORY_ROOT`
- `CFD_CLIP_INDEX_DIR`
- `CFD_IMAGE_ROOT`
- `CFD_CLIP_MODEL`
- `CFD_CLIP_MODEL_REVISION`
- `CFD_CLIP_LOCAL_MODEL_PATH`
- `CFD_CLIP_SELECTION_PROFILE`
- `CFD_CLIP_DEVICE`
- `CFD_EXCLUDED_TARGET_IDS`（改变候选池后原 profile 会按设计拒绝启动；若视频人物属于 CFD，须先生成并验证一个排除该身份的新 versioned profile）
- `LINEUP_IMAGE_TTL_SECONDS`
- `CLIP_INFERENCE_TIMEOUT_SECONDS`
- `CLIP_MAX_CONCURRENT_INFERENCE`（GPU 通常保持 `1`）
- `CLIP_RESULT_CACHE_SIZE`
- `QUALTRICS_ALLOWED_ORIGINS`（逗号分隔）
- `QUALTRICS_ALLOWED_ORIGIN_REGEX`

测试健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

测试 lineup：

```powershell
$body = @{
  participant_id = "pilot-001"
  description = "a white man in his thirties with short dark hair"
  study_version = "cfd_clip_contrastive_l14_v3"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/lineups `
  -ContentType "application/json" `
  -Body $body
```

正式问卷必须使用大学控制的 **HTTPS** 地址；浏览器会阻止 HTTPS Qualtrics 页面调用 HTTP 服务。图片 URL 在 Tetris 前签发，所以 `LINEUP_IMAGE_TTL_SECONDS` 不可低于 900（默认 7200）。建议在反向代理上同时配置 TLS、请求大小上限、速率限制、监控和固定 host。CORS 只限制浏览器调用来源，不是身份认证机制；当前 participant-browser endpoint 没有 API 身份认证，必须用网关限流，不能暴露为不受控的公共推理服务。

Python 线程无法可靠地 hard-kill 正在运行的 GPU/CLIP 调用；若 25 秒窗口后 task 一直不结束，服务会保持 busy 而不启动重复推理。生产部署应使用进程 supervisor/watchdog，在持续 busy 或失去模型健康状态时重启独立 inference worker。

当前 QSF 是 participant 浏览器直接调用服务，因此服务和反向代理在网络层会看到 participant IP；Qualtrics 的 **Anonymize responses** 不会匿名化这条独立连接。若 consent 使用本草稿中的 anonymous 表述，部署时必须关闭或匿名化 proxy/application access log 中的 client IP，并验证日志结果。若机构策略不允许这样做，改用 Qualtrics Survey Flow Web Service 做 server-to-server 调用，或在 ethics/consent 中准确披露网络 metadata 与保留期限。

同理，现有附件从 YouTube 加载视频、从 GitHub Pages 加载 Tetris；这两个第三方也会看到连接 IP，且 YouTube 可能使用 cookies。新草稿已披露这一点，但正式匿名研究更稳妥的做法是经许可把视频和 Tetris 自托管到大学控制、无识别性 access log 的域名。必须让 ethics/privacy notice 与最终技术部署完全一致。

## Qualtrics 导入后必须配置

1. 在 Qualtrics 新建项目，选择从文件导入 `Can_AI_select_CFD_CLIP_fillers_no_groups.qsf`。
2. 打开 **Survey Flow** 最上方的 Embedded Data。
3. 把：

   ```text
   CLIP_API_URL = https://YOUR-CLIP-SERVICE.example.org/v1/lineups
   ```

   改为真实 HTTPS endpoint。这个 URL 不是 secret；不要把 signing secret 或服务器凭据写进 QSF/JavaScript。
4. 检查 `verbalDescription` 位于 QID59 后，值为：

   ```text
   ${q://QID59/ChoiceTextEntryValue}
   ```

   QID59 的 JavaScript 已禁止纯空白并把输入限制为 2000 字符；导入后还应在 Builder 中加同样的内建最大长度验证，并在 Preview 测试 1、2000、2001 字符与纯空白四个边界。

5. 确认 QID24 的 `MinSeconds`、`MaxSeconds` 和 `Seconds` 都是 `300`。
6. 确认 Flow 中没有 Randomizer，并且顺序是 Video → Verbal description → Tetris → Lineup instructions → Dynamic lineup。
7. 检查 Survey Options 中 **Anonymize responses** 已开启。原附件宣称 anonymous，但实际 `AnonymizeResponse=No`；新 QSF 已改为 `Yes`。参见 [Security Survey Options](https://www.qualtrics.com/support/survey-platform/survey-module/survey-options/survey-protection/)。
8. 用 Preview 和匿名发布链接各做一次完整测试；Preview 与正式链接的 Origin 可能不同，要相应配置服务 CORS。
9. 导出一条测试 response，确认下面字段均有值：

   - `verbalDescription`
   - `lineup1`–`lineup6`
   - `lineup1Rank`–`lineup6Rank`
   - `lineup1Score`–`lineup6Score`
   - `clipRequestID`, `clipModel`, `clipModelRevision`, `clipModelSnapshotHash`, `clipModelSnapshotMatchVerified`, `clipIndexHash`, `clipCandidateEmbeddingHash`, `clipMediaManifestHash`
   - `clipScoringContractVersion/Hash`, `clipSelectionProfileID/Version/Hash/Status`
   - `clipScoringMethod`, `clipEffectiveMethod`, `clipScoreSpace`, `clipScorerVersion`
   - `clipPlanStatus`, `clipContrastPlanHash`, `clipContrastPairCount`, `clipContrastDimensionCount`, `clipAcceptedRuleIDs`, `clipAcceptedDimensions`, `clipAbstentionCount`
   - `clipContrastSchemaName/Version/Hash`, `clipParserVersion`, `clipPromptTemplateVersion/Hash`
   - `clipFallbackUsed`, `clipFallbackReason`, `clipStandardiseComponents`
   - `clipQueryHash`, `clipQueryHashKind`, `clipEffectiveQueryHash`, `clipPromptSetHash`, `clipPromptCount`
   - `clipPoolHash`, `clipPoolVersion`, `clipPoolSize`, `clipSelectionAlgorithm`
   - `clipQueryStrategy`, `clipQueryChunkCount`, `clipQueryTokenCount`, `clipContextTokens`
   - `SelectedPosition`, `SelectedImage`, `selectedRetrievalRank`, `selectedRetrievalScore`, `Outcome`, `responseType`
   - `confidence`, `LineupDecisionRTms`, `lineupLoadTimeMs`
   - 技术失败时：`technicalFailure`, `technicalFailureReason`, `technicalFailureAttempts`
10. 检查浏览器开发者工具：POST `/v1/lineups`、六个 signed image GET 和 CORS 均应成功。

QID60 使用 Qualtrics 的正式单选答案列记录 1–6、Not Present，或仅在连续三次失败后出现的 Technical problem。技术选项会设置 `technicalFailure=true`，自动填入仅供通过页面验证的 0% confidence，并立即分流到独立的技术故障 debrief/completion page；这类记录必须排除在识别与 confidence 分析之外。JavaScript 负责载入动态图片、显示 confidence 和补充 Embedded Data。Qualtrics 将自定义 JavaScript 视为需要研究者自行维护的功能，正式收数前需跨 Chrome/Edge/Firefox 回归测试：[Add JavaScript](https://www.qualtrics.com/support/survey-platform/survey-module/question-options/add-javascript/)。Timing question 的官方说明见 [Timing Question](https://www.qualtrics.com/support/survey-platform/survey-module/editing-questions/question-types-guide/advanced/timing/)。

`QID44 First Click` 从整页出现开始，仍可能包含最后一段网络/渲染时间；主要决策 RT 请使用 `LineupDecisionRTms`，它从六张图片全部 ready 到第一次正式选择计算。正式 MC 答案同时通过 radio `change`、Qualtrics click hook 与 page-submit 对账，以覆盖键盘和辅助技术选择。

本 QSF 与附件一致，保留 `templated / *2014` survey-taking layout，并使用该 layout 中的 `getEmbeddedData` / `setEmbeddedData`。不要在没有回归测试的情况下切换到 **Simple layout / New Survey Taking Experience**；新体验使用 `getJSEmbeddedData` / `setJSEmbeddedData`，且 Survey Flow 字段需采用 `__js_` 前缀。若品牌已强制启用新体验，优先把 CLIP POST 改为官方的 [Survey Flow Web Service](https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/advanced-elements/web-service/)，或按当前 Qualtrics JavaScript API 迁移所有 embedded fields。

## 本实现保留的原问卷行为

- 视频仍是 `MVW5-QE_aGQ`，结束后自动翻页。
- Tetris 仍嵌入 `https://damien-wangzehou.github.io/Tetris/`，恰好 300 秒自动翻页。
- Confidence 仍为 0%、10%……100%，强制回答。
- Attention check 仍问视频里有几个人；正确答案为 `1`。答错者仍立即 End Survey，因此看不到 demographics、debrief 或 completion code。这是按用户“后续步骤相同”的要求保留的，正式招募前应确认补偿与伦理安排。
- Age 仍限制为 18–100 的整数；gender、ethnicity 和 comments 与源问卷相同。
- completion code 仍为 `BS-${RandomSurveyID}-FS`。
- completion code 的五位随机数只用于 participant-facing code；CLIP 幂等/排位使用独立的高熵 `lineupSeed`，避免把五位数误当唯一 participant ID。

## 必须由 PI / ethics 确认的事项

这份 participant information、consent 信息和 debrief 是与新流程一致的**研究草稿**，不是伦理批准文件。正式上线前必须由 PI/ethics 审批，尤其是：

- typed description 会传给研究托管的 CLIP 服务；
- raw description 是否保留、保存多久、是否进入 open data，以及如何人工去除误填的身份信息；
- CFD 图片能否以本服务的短时 signed URL 向参与者展示；不要把 CFD 图片复制到公开仓库；
- 本设计是固定 target-absent、六名全 filler lineup。若本意是 `5 fillers + perpetrator`，必须先改设计与结果编码；
- 所有参与者只有一个 CLIP 条件，所以可以估计 correct-rejection / filler-ID、confidence 和 RT，但**不能单凭本实验比较 CLIP 是否优于其他 filler 方法**；
- 文案说不能使用手机，但当前仍和附件一样没有 Device Type 分支。若手机排除是纳入标准，应在 Survey Flow 增加设备分支；
- free text 通常具有再识别风险，不能仅因为没有姓名就自动视为匿名。
- 当前 ViT-L-14 snapshot 的 context window 为 77 tokens；主 contrast prompts 都在窗口内，raw fallback 使用显式 chunk-weighted aggregation。该策略、approved rules 与 fallback 必须在 protocol 中固定，不能在收数中途更换。
- index 与服务已核对同一 Hugging Face revision；正式部署仍应锁定并记录 `sentence-transformers`、`transformers`、`torch` 与硬件环境。

## 自动检查

```powershell
python -m pip install pytest httpx
python -m pytest `
  studies/qualtrics_cfd_clip/tests -q
```

测试使用模拟编码器与临时模型文件，不下载 CLIP 权重。覆盖：White Male pool 先筛选后检索、top-6 与确定性随机排位、signed URL 过期、manifest 路径越界防护，以及 QSF 中无 Randomizer、流程顺序、同意验证、300 秒 Tetris、正式 lineup response、技术失败出口和匿名设置。
