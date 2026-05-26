# Study 3 CFD CLIP Pilot

This folder contains thin command-line scripts for a feasibility pilot of CLIP
retrieval on the Chicago Face Database (CFD), matching the Study 1 logic in the
proposal at a smaller scale:

1. Build a manifest from CFD images and optional CFD norming metadata.
2. Generate proxy witness descriptions from CFD annotations, or replace this
   file with real free-recall descriptions.
3. Encode CFD images with CLIP and save an exact cosine-search index.
4. Evaluate whether the expected CFD target appears in the CLIP top-k results.
5. Export top-5 CLIP-matched filler sets for qualitative inspection.

The scripts are wrappers around `pyWitnessAI.cfd_clip_pilot`, so the reusable
logic stays in the package and the study folder only holds runnable entrypoints.

## Example

```powershell
python studies\study3_cfd_clip_pilot\01_build_cfd_manifest.py `
  --image-dir D:\Data\CFD\Images `
  --metadata-path D:\Data\CFD\CFD_norming.csv `
  --output studies\study3_cfd_clip_pilot\outputs\manifest.csv

python studies\study3_cfd_clip_pilot\02_generate_proxy_descriptions.py `
  --manifest studies\study3_cfd_clip_pilot\outputs\manifest.csv `
  --output studies\study3_cfd_clip_pilot\outputs\queries.csv

python studies\study3_cfd_clip_pilot\03_build_clip_index.py `
  --manifest studies\study3_cfd_clip_pilot\outputs\manifest.csv `
  --index-dir studies\study3_cfd_clip_pilot\outputs\clip_index `
  --model-name clip-ViT-B-32 `
  --batch-size 32 `
  --show-progress

python studies\study3_cfd_clip_pilot\04_evaluate_clip_retrieval.py `
  --index-dir studies\study3_cfd_clip_pilot\outputs\clip_index `
  --queries studies\study3_cfd_clip_pilot\outputs\queries.csv `
  --output-dir studies\study3_cfd_clip_pilot\outputs\evaluation `
  --top-k 50

python studies\study3_cfd_clip_pilot\05_build_clip_filler_sets.py `
  --index-dir studies\study3_cfd_clip_pilot\outputs\clip_index `
  --queries studies\study3_cfd_clip_pilot\outputs\queries.csv `
  --output studies\study3_cfd_clip_pilot\outputs\filler_sets.csv `
  --top-k 50 `
  --filler-count 5
```

## Inputs

The generated query file must contain:

- `description`: free text or proxy CFD description.
- `target_id` or `image_id`: the expected CFD target for evaluation.

For the main pilot, replace the proxy descriptions with human free-recall
descriptions collected from a small independent sample. Keep the same columns so
the evaluation script remains unchanged.

## Outputs

`04_evaluate_clip_retrieval.py` writes:

- `retrieval_results.csv`: long-form top-k CLIP retrievals.
- `per_query_metrics.csv`: one row per description with target rank and hit flags.
- `summary.json`: top-1/top-5/top-10/top-k hit rates and MRR.

`05_build_clip_filler_sets.py` writes one row per selected filler candidate. It
excludes the same CFD target by default, because Study 1 uses CLIP results as
fillers rather than as the suspect image.
