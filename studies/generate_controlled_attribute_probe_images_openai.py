from __future__ import annotations

import argparse
import base64
import csv
import os
import time
from pathlib import Path


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "pyWitnessAI").exists():
        return cwd
    if (cwd.parent / "src" / "pyWitnessAI").exists():
        return cwd.parent
    return Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_prompt(row: dict[str, str], include_negative_prompt: bool = True) -> str:
    prompt = row["prompt"].strip()
    negative = row.get("negative_prompt", "").strip()
    if include_negative_prompt and negative:
        prompt += "\n\nAvoid: " + negative
    return prompt


def output_path_for(row: dict[str, str], output_format: str) -> Path:
    path = Path(row["output_path"])
    if path.suffix.lower() != f".{output_format.lower()}":
        path = path.with_suffix(f".{output_format.lower()}")
    return path


def generate_image(
    client,
    *,
    model: str,
    prompt: str,
    output_path: Path,
    size: str,
    quality: str,
    output_format: str,
) -> None:
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=output_format,
    )
    image_base64 = response.data[0].b64_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(image_base64))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate controlled synthetic face images from OpenAI image prompts."
    )
    parser.add_argument(
        "--prompts-csv",
        type=Path,
        default=repo_root() / "data" / "controlled_clip_attribute_probe" / "generation_prompts.csv",
    )
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--output-format", default="png", choices=["png", "webp", "jpeg"])
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N pending images.")
    parser.add_argument("--start", type=int, default=0, help="Skip this many rows before generating.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-negative-prompt",
        action="store_true",
        help="Do not append the negative_prompt column as an Avoid clause.",
    )
    args = parser.parse_args()

    if not args.prompts_csv.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {args.prompts_csv}")

    rows = read_rows(args.prompts_csv)
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]

    pending = []
    for row in rows:
        output_path = output_path_for(row, args.output_format)
        if output_path.exists() and not args.overwrite:
            continue
        pending.append((row, output_path))

    print(f"Prompt CSV: {args.prompts_csv}")
    print(f"Pending images: {len(pending)}")
    print(f"Model: {args.model} | size={args.size} | quality={args.quality} | format={args.output_format}")

    if args.dry_run:
        for row, output_path in pending[:5]:
            print("\nOUT:", output_path)
            print(build_prompt(row, include_negative_prompt=not args.no_negative_prompt))
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running image generation.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK first: python -m pip install --upgrade openai") from exc

    client = OpenAI()
    for index, (row, output_path) in enumerate(pending, start=1):
        prompt = build_prompt(row, include_negative_prompt=not args.no_negative_prompt)
        print(f"[{index}/{len(pending)}] {row['mode']}/{row['stage']}/{row['label']} -> {output_path}")
        generate_image(
            client,
            model=args.model,
            prompt=prompt,
            output_path=output_path,
            size=args.size,
            quality=args.quality,
            output_format=args.output_format,
        )
        if args.sleep:
            time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
