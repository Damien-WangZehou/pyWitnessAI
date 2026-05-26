from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cfd import build_cfd_manifest
from .clip_backend import SentenceTransformerClipEncoder
from .description import build_proxy_descriptions
from .evaluate import evaluate_retrieval, write_evaluation_outputs
from .index import ClipIndex
from .lineup import build_filler_sets


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CFD CLIP pilot utilities for manifest, indexing, retrieval, and filler selection."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    manifest = subparsers.add_parser("manifest", help="Build a CFD image manifest.")
    manifest.add_argument("--image-dir", required=True, help="Directory containing CFD images.")
    manifest.add_argument("--metadata-path", help="Optional CFD metadata CSV/XLSX path.")
    manifest.add_argument("--output", required=True, help="Output manifest CSV.")
    manifest.add_argument("--all-expressions", action="store_true", help="Keep all expressions instead of neutral only.")
    manifest.add_argument("--max-images", type=int, help="Optional cap for quick smoke tests.")
    manifest.set_defaults(func=_cmd_manifest)

    descriptions = subparsers.add_parser("descriptions", help="Generate proxy descriptions from CFD metadata.")
    descriptions.add_argument("--manifest", required=True, help="Manifest CSV.")
    descriptions.add_argument("--output", required=True, help="Output query CSV.")
    descriptions.add_argument("--unique-by", default="target_id", help="Column used to create one query per item.")
    descriptions.set_defaults(func=_cmd_descriptions)

    index = subparsers.add_parser("index", help="Build a CLIP image index.")
    index.add_argument("--manifest", required=True, help="Manifest CSV.")
    index.add_argument("--index-dir", required=True, help="Output index directory.")
    _add_model_args(index)
    index.add_argument("--show-progress", action="store_true")
    index.set_defaults(func=_cmd_index)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate CLIP retrieval against target IDs.")
    evaluate.add_argument("--index-dir", required=True, help="Existing CLIP index directory.")
    evaluate.add_argument("--queries", required=True, help="Query CSV with description and target_id/image_id.")
    evaluate.add_argument("--output-dir", required=True, help="Directory for result CSV/JSON files.")
    evaluate.add_argument("--top-k", type=int, default=50)
    _add_model_args(evaluate)
    evaluate.add_argument("--show-progress", action="store_true")
    evaluate.set_defaults(func=_cmd_evaluate)

    fillers = subparsers.add_parser("fillers", help="Build top-k CLIP filler sets.")
    fillers.add_argument("--index-dir", required=True, help="Existing CLIP index directory.")
    fillers.add_argument("--queries", required=True, help="Query CSV with description and target_id.")
    fillers.add_argument("--output", required=True, help="Output filler-set CSV.")
    fillers.add_argument("--top-k", type=int, default=50)
    fillers.add_argument("--filler-count", type=int, default=5)
    fillers.add_argument("--max-pairwise-clip-similarity", type=float)
    fillers.add_argument("--include-same-target", action="store_true")
    _add_model_args(fillers)
    fillers.add_argument("--show-progress", action="store_true")
    fillers.set_defaults(func=_cmd_fillers)

    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default="clip-ViT-B-32", help="sentence-transformers CLIP model name.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=32)


def _cmd_manifest(args: argparse.Namespace) -> None:
    manifest = build_cfd_manifest(
        image_dir=args.image_dir,
        metadata_path=args.metadata_path,
        neutral_only=not args.all_expressions,
        max_images=args.max_images,
    )
    _write_csv(manifest, args.output)
    print(f"Wrote {len(manifest)} manifest rows to {args.output}")


def _cmd_descriptions(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    descriptions = build_proxy_descriptions(manifest, unique_by=args.unique_by)
    _write_csv(descriptions, args.output)
    print(f"Wrote {len(descriptions)} descriptions to {args.output}")


def _cmd_index(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    encoder = SentenceTransformerClipEncoder(model_name=args.model_name, device=args.device)
    index = ClipIndex.build(
        manifest=manifest,
        encoder=encoder,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
    )
    index.save(args.index_dir)
    print(f"Wrote CLIP index with {len(manifest)} images to {args.index_dir}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    index = ClipIndex.load(args.index_dir)
    queries = pd.read_csv(args.queries)
    encoder = SentenceTransformerClipEncoder(model_name=args.model_name, device=args.device)
    retrieval_results, per_query, summary = evaluate_retrieval(
        index=index,
        queries=queries,
        encoder=encoder,
        top_k=args.top_k,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
    )
    write_evaluation_outputs(args.output_dir, retrieval_results, per_query, summary)
    print(f"Wrote evaluation outputs to {args.output_dir}")
    print(summary)


def _cmd_fillers(args: argparse.Namespace) -> None:
    index = ClipIndex.load(args.index_dir)
    queries = pd.read_csv(args.queries)
    encoder = SentenceTransformerClipEncoder(model_name=args.model_name, device=args.device)
    fillers = build_filler_sets(
        index=index,
        queries=queries,
        encoder=encoder,
        top_k=args.top_k,
        filler_count=args.filler_count,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
        exclude_same_target=not args.include_same_target,
        max_pairwise_clip_similarity=args.max_pairwise_clip_similarity,
    )
    _write_csv(fillers, args.output)
    print(f"Wrote {len(fillers)} filler rows to {args.output}")


def _write_csv(dataframe: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
