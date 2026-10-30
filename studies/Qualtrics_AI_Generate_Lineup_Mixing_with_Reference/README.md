# AI-generated lineup survey

[Chinese documentation](README-CN.md)

This guide describes the four-condition Qualtrics study and its local lineup service.

## Survey and service configuration

- The survey has already been imported into Qualtrics and is maintained there. After formal testing, the researcher will export its QSF file back into this directory for archiving. Local survey generation is not part of this workflow.
- The service enables prompt processing with `conversation_model="gpt-6.1-sol"` and standardizes portraits with `eye_distance=0.23` and `fill_canvas=True`.
- Local tests use fake providers to verify the configured behavior. They do not verify availability of the conversation or image model through live API requests.

## Experimental design and data

After the verbal description is submitted and before the five-minute Tetris task, the Qualtrics Randomizer presents one of the following four Embedded Data elements using balanced allocation. Allocation is counted when participants enter the randomizer; it does not guarantee identical numbers of valid completed responses in all four groups.

| condition | targetLineup | Six-person lineup |
| --- | --- | --- |
| mixingFaces | targetAbsent | Six fillers generated using reference images |
| mixingFaces | targetPresent | The fixed suspect plus five fillers generated using reference images |
| noMixing | targetAbsent | Six fillers generated without reference images |
| noMixing | targetPresent | The fixed suspect plus five fillers generated without reference images |

Target-present and target-absent lineups are abbreviated as TP and TA below.

| Export field | Values / meaning |
| --- | --- |
| responseType | `suspectId`, `fillerId`, or `rejectId` |
| targetLineup | `targetPresent` or `targetAbsent` |
| condition | `mixingFaces` or `noMixing` |
| confidence | A numeric string from `0` to `100`, in increments of 10; blank for technical failures |
| SelectedPosition / SelectedImage | The final selected position and image ID; the position is `NotPresent` for a rejection |
| suspectPosition | The suspect's position in TP lineups; blank in TA lineups |
| lineup1…lineup6 / …Url / …Role | The actual presentation order, image URLs, and `suspect` / `filler` roles |
| FirstResponseType / FirstSelectedPosition / FirstSelectedImage | The first selection, for comparison with the final selection after any changes |
| LineupDecisionRTms / LineupDecisionRTs | Time from all six images being loaded and displayed simultaneously to the first selection |
| LineupFinalDecisionRTms | Time from the six images being displayed to submission of the lineup page |
| lineupLoadTimeMs | Time from opening the lineup page to displaying all images, including any wait for image generation |
| FirstClickonLineupPage / LineupTiming | Native Qualtrics page timing, which may include preparation time; use the separate timings above when analyzing identification response times |
| lineupID / generationMetadata | The service job ID, models, reference filenames, prompt hashes, and standardization parameters |
| technicalFailure / technicalFailureReason | A technical-failure flag and reason; failure is not an identification category, and `responseType` remains blank |

There is no designated lineup member and no `designatedId` response. Selecting the suspect in a TP lineup records `suspectId`; selecting any filler in any condition records `fillerId`; selecting Not Present in any condition records `rejectId`.

The survey retains the original information page, consent question, video, description task, Tetris, attention checks, demographic questions, and completion code. Tetris is timed for 300 seconds. Confidence is collected on a separate page after the lineup decision. The technical-failure branch exits before the confidence page, so participants are not assigned an artificial confidence rating of 0%. The original termination logic for incorrect attention-check answers is retained. Survey-flow validation is performed in Qualtrics as part of formal testing.

## Image generation

Both conditions pass the participant's description through `RawDescriptionParser(lowercase=False)`. The service sets `conversation_model="gpt-6.1-sol"`, which enables the default `OpenAIPromptProcessor`, even though `prompt_processor=None` is passed. The latter means that no custom processor is supplied; it does not disable the processor selected by `conversation_model`.

- `mixingFaces`: randomly sample five or six reference images without replacement from `data/ReferenceFaces_CFD_WhiteMale` under the repository root (on the original machine, `D:/PhD/Software/pyWitnessAI/data/ReferenceFaces_CFD_WhiteMale`). Generate one filler per reference image using `images.edit`; references are used individually, not combined into a single image request.
- `noMixing`: generate five or six fillers from the description without supplying reference images, using `images.generate`.
- The fixed suspect is `perp_refer_to_AI_generated_image.png` in this directory. It is added only to TP lineups and is never used as a filler-generation reference.
- Generation uses the notebook image settings: 1024×1024, high quality, and PNG. Before presentation, all members, including the suspect, are shuffled and processed with the same `FaceStandardizer(alignment="center", eye_distance=0.23, fill_canvas=True)` configuration to produce 256×320 RGB PNGs. The requested eye distance is 23% of the output width, or 58.88 pixels. Scaling must cover the canvas to avoid introducing padded borders when opaque source images are reduced, so the actual eye distance can exceed this target. Each image's actual eye distance and whether the coverage constraint was applied are recorded in `standardization_measurements` in the private `manifest.json`. The first standardization run requires downloading the YuNet face detector.

The notebook base prompts use different wording: the mixing prompt begins with `Make a photo of a man based on this image, with the following features: ...`; the no-mixing prompt begins with `Make a photo of an under average-looking man, with facial features and hair style completely different from each other, but matching the following description: ...`. Both add the same requirements for a portrait from the shoulders up, a gray background, and a frontal gaze toward the camera. With prompt processing enabled, these base prompts are input hints for the processor; the final prompts are rebuilt from the original description, a supplementary appearance profile, and the fixed portrait requirements.

The current processing differs between conditions:

- `noMixing` plans five or six supplementary profiles together in one batch. Each final prompt starts with `An unretouched candid photograph of an adult in an everyday street setting.`, followed by the original description, that filler's profile, and the fixed portrait requirements. The original `under average-looking` opening is not directly retained.
- `mixingFaces` creates a separate one-image generator for each reference, so each filler receives its own single-item profile-planning request. The fillers are not planned together as a whole lineup. With the current raw parser, each final prompt begins with `Make a photo of a person based on this image, with the following features:`, followed by the original description, its supplementary profile, reference-compatibility guidance, and the fixed portrait requirements.

The manipulation therefore includes differences in prompt wording and profile-planning scope as well as reference-image use. The no-mixing prompt currently combines a street-setting instruction with the fixed gray-background requirement; reconcile this wording before relying on uniform backgrounds. The service records final prompt text and hashes for inspection.

Standardized cropping does not provide background segmentation or guarantee perceptually matched appearance. Before formal data collection, pilot the generation quality, whether the suspect stands out, the external video and Tetris links, and exports from all four conditions. The local checks described here do not generate test images through a paid API or verify import/publication in a live Qualtrics account.

## Start the service and configure the survey

From the repository root, `D:/PhD/Software/pyWitnessAI`, use Python 3.10 or 3.11:

```powershell
python -m pip install -e ".[generation,standardization]"
python -m pip install -r studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference/requirements.txt

$env:OPENAI_API_KEY = "<real API key>"
$env:ALLOWED_ORIGINS = "https://bristolexppsych.eu.qualtrics.com"
# If needed, explicitly select an image model available to your account
# that supports both generate and edit:
$env:IMAGE_MODEL = "<image-generation-model>"

python -m uvicorn lineup_service:app --app-dir studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference --host 127.0.0.1 --port 8000 --workers 1
```

If PowerShell is already in this study directory, use this startup command instead:

```powershell
python -m uvicorn lineup_service:app --host 127.0.0.1 --port 8000 --workers 1
```

After startup, visit `http://127.0.0.1:8000/health` (or the compatibility route `/healthz`). The expected response is `{"status":"ok","study_version":"ai_generated_lineup_2x2_v1"}`. The root path `/` returns service entry points, and `/docs` provides API documentation. These GET requests do not generate images or call a paid API. A 404 for the browser's automatic `/favicon.ico` request does not affect the service. Restart uvicorn after changing service code.

Keep the API key only in the server environment; do not place it in the QSF file or frontend JavaScript. If `IMAGE_MODEL` is unset, the service uses the project's current default image model. Neither that default nor the hard-coded conversation model has been verified as available to your account. Replace the angle-bracket placeholders with real configuration before running the example.

Use an HTTPS reverse proxy or tunnel that you control to forward `/health`, `/v1/lineups`, `/v1/lineups/{id}`, and `/media/...` without rewriting those paths. This implementation is a single study-service process and must use **one worker**. Do not share the same runtime directory across multiple processes or service instances. The researcher should control the public endpoint and limit usage; CORS settings alone do not authenticate callers.

Configure the existing survey in Qualtrics:

1. Open the survey that has already been imported into Qualtrics.
2. In the Embedded Data element at the top of Survey Flow, set `LINEUP_API_BASE_URL` to the real HTTPS service address, such as `https://<your-lineup-service-domain>`, without `/v1/lineups`. The randomizer sets the remaining condition fields.
3. Check that `ALLOWED_ORIGINS` matches the actual origin of the survey page in the browser. Separate multiple origins with commas.
4. Check all four conditions and exported data in Preview before publishing. To test a specific condition deterministically, replace the Randomizer with that condition's Embedded Data element in a temporary test copy. Restore the balanced randomizer for the production survey.
5. After formal testing, export the final QSF from Qualtrics into this study directory to archive the tested survey.

The Tetris page submits a job to start generation early. The lineup page reuses the same job and polls for completion, so the browser does not need to hold a single image-generation request open for the full generation time. Retry reuses the same participant, description, and condition without automatically generating again. Failed or interrupted generation retains its error state for researcher review; participant retries cannot repeatedly incur generation charges. Network or image-loading failures can be retried to retrieve the same result.

Service environment variables:

| Variable | Purpose / default |
| --- | --- |
| `REFERENCE_DIR` | Override the default reference-image directory |
| `SUSPECT_PATH` | Override the default fixed-suspect image path |
| `LINEUP_STORAGE_DIR` | Persistent storage directory; defaults to `runtime/` in this study directory |
| `MAX_CONCURRENT_JOBS` | Maximum concurrent generation jobs; default `2` |
| `MAX_PENDING_JOBS` | Maximum total queued and generating jobs; default `40` |
| `OPENAI_API_KEY` | Server-side API credential |
| `IMAGE_MODEL` | Optional image-model override |
| `ALLOWED_ORIGINS` | Comma-separated allowed survey origins |

Tune the job limits for the expected participant volume, model limits, and measured generation time, and pilot the deployment before data collection. The current conversation model is configured in `lineup_service.py`, not through an environment variable.

`runtime/private/` stores raw generated images, prompts, descriptions, and research records; `runtime/jobs.sqlite3` stores jobs. Only `runtime/media/` serves standardized images publicly. The runtime directory is excluded by this directory's `.gitignore`. Preserve the job database and media to reproduce the lineup each participant actually saw, including when moving or restarting the service.

## Local validation

Install the optional local test dependencies if needed:

```powershell
python -m pip install pytest httpx
```

Run the tests that are present in this checkout:

```powershell
python -m pytest studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference/tests/test_lineup_service.py -q
node --test studies/Qualtrics_AI_Generate_Lineup_Mixing_with_Reference/tests/test_frontend.cjs
```

The tests use fake providers and do not send real prompt-processing or image-generation requests. They cover prompt-processing status, 23% target eye distance and canvas coverage, lineup size and roles in all four conditions, idempotent retries and persistence, CORS and media boundaries, and frontend selection and timing logic. Passing these checks verifies local behavior, not live model availability. The randomizer, complete survey flow, and Qualtrics exports are checked in the imported survey during formal testing.

Qualtrics documentation: [Importing QSF](https://www.qualtrics.com/support/survey-platform/survey-module/survey-tools/import-and-export-surveys/), [Randomizer](https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/standard-elements/randomizer/), [Embedded Data](https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/standard-elements/embedded-data/), and [JavaScript](https://www.qualtrics.com/support/survey-platform/survey-module/question-options/add-javascript/). The survey requires custom JavaScript to be enabled for the account.
