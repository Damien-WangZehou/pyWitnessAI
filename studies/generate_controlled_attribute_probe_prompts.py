from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


FEATURE_ORDER = [
    "gender",
    "hair",
    "facial_hair",
    "eyes",
    "eyebrows",
    "nose",
    "build",
    "face_shape",
    "race",
    "forehead",
    "mouth",
    "ears",
    "jaw",
    "teeth",
]


BALANCED_VARIANTS = [
    "around 24 years old",
    "around 28 years old",
    "around 32 years old",
    "around 36 years old",
    "around 40 years old",
    "around 44 years old",
    "around 48 years old",
    "around 52 years old",
    "around 56 years old",
]


MODE_A_FEATURE_CONFIG = {
    "gender": {
        "positive_label": "male",
        "negative_label": "female",
        "positive_phrase": "a male adult person",
        "negative_phrase": "a female adult person",
        "query": "A frontal face photograph of a male person.",
    },
    "hair": {
        "positive_label": "long hair",
        "negative_label": "short hair",
        "positive_phrase": "an adult person with long hair",
        "negative_phrase": "an adult person with short hair",
        "query": "A frontal face photograph of a person with long hair.",
    },
    "facial_hair": {
        "positive_label": "full beard",
        "negative_label": "clean shaven",
        "positive_phrase": "an adult man with a full beard",
        "negative_phrase": "an adult man with no facial hair",
        "query": "A frontal face photograph of a person with a full beard.",
    },
    "eyes": {
        "positive_label": "blue eyes",
        "negative_label": "brown eyes",
        "positive_phrase": "an adult person with blue eyes",
        "negative_phrase": "an adult person with brown eyes",
        "query": "A frontal face photograph of a person with blue eyes.",
    },
    "eyebrows": {
        "positive_label": "thick eyebrows",
        "negative_label": "thin eyebrows",
        "positive_phrase": "an adult person with thick eyebrows",
        "negative_phrase": "an adult person with thin eyebrows",
        "query": "A frontal face photograph of a person with thick eyebrows.",
    },
    "nose": {
        "positive_label": "broad nose",
        "negative_label": "narrow nose",
        "positive_phrase": "an adult person with a broad nose",
        "negative_phrase": "an adult person with a narrow nose",
        "query": "A frontal face photograph of a person with a broad nose.",
    },
    "build": {
        "positive_label": "broad build",
        "negative_label": "slim build",
        "positive_phrase": "an adult person with a broad build",
        "negative_phrase": "an adult person with a slim build",
        "query": "A frontal face photograph of a person with a broad build.",
    },
    "face_shape": {
        "positive_label": "round face",
        "negative_label": "narrow face",
        "positive_phrase": "an adult person with a round face",
        "negative_phrase": "an adult person with a narrow face",
        "query": "A frontal face photograph of a person with a round face.",
    },
    "race": {
        "positive_label": "Asian",
        "negative_label": "White",
        "positive_phrase": "an Asian adult person",
        "negative_phrase": "a White adult person",
        "query": "A frontal face photograph of an Asian person.",
    },
    "forehead": {
        "positive_label": "high forehead",
        "negative_label": "low forehead",
        "positive_phrase": "an adult person with a high forehead",
        "negative_phrase": "an adult person with a low forehead",
        "query": "A frontal face photograph of a person with a high forehead.",
    },
    "mouth": {
        "positive_label": "full lips",
        "negative_label": "thin lips",
        "positive_phrase": "an adult person with full lips",
        "negative_phrase": "an adult person with thin lips",
        "query": "A frontal face photograph of a person with full lips.",
    },
    "ears": {
        "positive_label": "visible ears",
        "negative_label": "covered ears",
        "positive_phrase": "an adult person with clearly visible ears",
        "negative_phrase": "an adult person whose ears are covered by hair",
        "query": "A frontal face photograph of a person with visible ears.",
    },
    "jaw": {
        "positive_label": "strong jaw",
        "negative_label": "soft jaw",
        "positive_phrase": "an adult person with a strong square jaw",
        "negative_phrase": "an adult person with a soft rounded jawline",
        "query": "A frontal face photograph of a person with a strong jaw.",
    },
    "teeth": {
        "positive_label": "visible teeth",
        "negative_label": "no visible teeth",
        "positive_phrase": "an adult person smiling with visible teeth",
        "negative_phrase": "an adult person with a closed-mouth smile and no visible teeth",
        "query": "A frontal face photograph of a person with visible teeth.",
    },
}


MODE_B_STEPS = [
    ("gender", "male", "female", "subject"),
    ("hair", "long hair", "short hair", "detail"),
    ("facial_hair", "no facial hair", "a full beard", "detail"),
    ("eyes", "blue eyes", "brown eyes", "detail"),
    ("eyebrows", "thick eyebrows", "thin eyebrows", "detail"),
    ("nose", "a broad nose", "a narrow nose", "detail"),
    ("build", "a broad build", "a slim build", "detail"),
    ("face_shape", "a round face", "a narrow face", "detail"),
    ("race", "White", "Asian", "subject"),
    ("forehead", "a high forehead", "a low forehead", "detail"),
    ("mouth", "full lips", "thin lips", "detail"),
    ("ears", "visible ears", "ears covered by hair", "detail"),
    ("jaw", "a strong square jaw", "a soft rounded jawline", "detail"),
    ("teeth", "no visible teeth", "visible teeth", "detail"),
]

MODE_B_FEATURE_CONFIG = {
    **MODE_A_FEATURE_CONFIG,
    "race": {
        **MODE_A_FEATURE_CONFIG["race"],
        "positive_label": "White",
        "negative_label": "Asian",
        "positive_phrase": "a White adult person",
        "negative_phrase": "an Asian adult person",
        "query": "A frontal face photograph of a White person.",
    },
}


@dataclass(frozen=True)
class PromptRow:
    mode: str
    stage: str
    feature: str
    label: str
    attribute_label: str
    image_number: int
    output_path: str
    prompt: str
    negative_prompt: str
    query: str


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "pyWitnessAI").exists():
        return cwd
    if (cwd.parent / "src" / "pyWitnessAI").exists():
        return cwd.parent
    return Path(__file__).resolve().parents[1]


def make_prompt(subject_phrase: str, variant: str) -> str:
    return (
        "Photorealistic synthetic fictional adult face portrait, "
        f"{subject_phrase}, {variant}, frontal face, centered crop, shoulders visible, "
        "plain light gray studio background, even soft lighting, neutral camera angle, "
        "realistic skin texture, high detail, not a real person, not a celebrity."
    )


def negative_prompt() -> str:
    return (
        "real person, celebrity, child, teenager, side profile, extreme pose, sunglasses, "
        "hat, mask, heavy makeup, distorted face, duplicate face, extra head, extra eyes, "
        "cartoon, illustration, low resolution, blurry, watermark, text"
    )


def render_mode_b_subject(active_steps: list[tuple[str, str, str]]) -> str:
    subject_terms = []
    details = []
    by_feature = {feature: phrase for feature, phrase, role in active_steps}
    for feature in ["race", "gender"]:
        if feature in by_feature:
            subject_terms.append(by_feature[feature])
    for feature, phrase, role in active_steps:
        if role == "detail":
            details.append(phrase)
    subject = " ".join(subject_terms + ["adult person"])
    if details:
        subject += " with " + ", ".join(details)
    return subject


def mode_b_query(active_positive_steps: list[tuple[str, str, str]]) -> str:
    subject = render_mode_b_subject(active_positive_steps)
    article = "an" if subject[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"A frontal face photograph of {article} {subject}."


def build_mode_a_rows(data_root: Path) -> list[PromptRow]:
    rows = []
    mode = "mode_a_single_feature"
    for step_index, feature in enumerate(FEATURE_ORDER, start=1):
        cfg = MODE_A_FEATURE_CONFIG[feature]
        stage = f"{step_index:02d}_{feature}"
        for label, phrase_key, attribute_label in [
            ("positive", "positive_phrase", cfg["positive_label"]),
            ("negative", "negative_phrase", cfg["negative_label"]),
        ]:
            for image_number, variant in enumerate(BALANCED_VARIANTS, start=1):
                output_path = data_root / mode / stage / label / f"{label}_{image_number:02d}.png"
                rows.append(
                    PromptRow(
                        mode=mode,
                        stage=stage,
                        feature=feature,
                        label=label,
                        attribute_label=attribute_label,
                        image_number=image_number,
                        output_path=str(output_path),
                        prompt=make_prompt(cfg[phrase_key], variant),
                        negative_prompt=negative_prompt(),
                        query=cfg["query"],
                    )
                )
    return rows


def build_mode_b_rows(data_root: Path) -> list[PromptRow]:
    rows = []
    mode = "mode_b_conditional_ladder"
    positive_active: list[tuple[str, str, str]] = []
    for step_index, (feature, positive_phrase, negative_phrase, role) in enumerate(MODE_B_STEPS, start=1):
        stage = f"{step_index:02d}_{feature}"
        positive_active = positive_active + [(feature, positive_phrase, role)]
        negative_active = positive_active[:-1] + [(feature, negative_phrase, role)]
        cfg = MODE_B_FEATURE_CONFIG[feature]
        query = mode_b_query(positive_active)

        for label, active_steps, attribute_label in [
            ("positive", positive_active, cfg["positive_label"]),
            ("negative", negative_active, cfg["negative_label"]),
        ]:
            subject_phrase = render_mode_b_subject(active_steps)
            for image_number, variant in enumerate(BALANCED_VARIANTS, start=1):
                output_path = data_root / mode / stage / label / f"{label}_{image_number:02d}.png"
                rows.append(
                    PromptRow(
                        mode=mode,
                        stage=stage,
                        feature=feature,
                        label=label,
                        attribute_label=attribute_label,
                        image_number=image_number,
                        output_path=str(output_path),
                        prompt=make_prompt(subject_phrase, variant),
                        negative_prompt=negative_prompt(),
                        query=query,
                    )
                )
    return rows


def write_csv(path: Path, rows: list[PromptRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(PromptRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_stage_prompt_files(rows: list[PromptRow]) -> None:
    grouped: dict[tuple[str, str], list[PromptRow]] = {}
    for row in rows:
        grouped.setdefault((row.mode, row.stage), []).append(row)

    for (_, _), stage_rows in grouped.items():
        stage_folder = Path(stage_rows[0].output_path).parents[1]
        stage_folder.mkdir(parents=True, exist_ok=True)
        write_csv(stage_folder / "generation_prompts.csv", stage_rows)
        for row in stage_rows:
            output_path = Path(row.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path = output_path.with_suffix(".prompt.txt")
            prompt_path.write_text(row.prompt + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create prompt manifests and folders for the controlled CLIP attribute probe."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root() / "data" / "controlled_clip_attribute_probe",
    )
    args = parser.parse_args()

    rows = build_mode_a_rows(args.data_root) + build_mode_b_rows(args.data_root)
    args.data_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.data_root / "generation_prompts.csv", rows)
    write_stage_prompt_files(rows)

    print(f"Created prompt plan with {len(rows)} rows")
    print(f"Data root: {args.data_root}")
    print(f"Master CSV: {args.data_root / 'generation_prompts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
