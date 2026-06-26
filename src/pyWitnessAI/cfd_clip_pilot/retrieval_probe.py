from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

from .clip_backend import SentenceTransformerClipEncoder
from .index import ClipIndex
from .visualization import search_and_show_lineup

PromptRole = Literal["subject", "detail"]


@dataclass(frozen=True)
class FeaturePromptStep:
    """One phrase added to a cumulative text-description ladder."""

    feature: str
    phrase: str
    role: PromptRole = "detail"


@dataclass(frozen=True)
class DescriptionLadder:
    """Build cumulative CLIP text prompts from feature phrases."""

    steps: tuple[FeaturePromptStep, ...]
    base_details: tuple[str, ...] = ()
    prefix: str = "A frontal face photograph of"
    fallback_subject: str = "person"
    subject_order: tuple[str, ...] = ("race", "gender")
    base_feature: str | None = None

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[FeaturePromptStep | tuple[str, str] | tuple[str, str, PromptRole]],
        **kwargs,
    ) -> "DescriptionLadder":
        return cls(steps=tuple(_coerce_step(step) for step in steps), **kwargs)

    def to_frame(self) -> pd.DataFrame:
        records = []
        if self.base_feature is not None:
            records.append(
                {
                    "step": 0,
                    "feature": self.base_feature,
                    "added_phrase": " + ".join(self.base_details),
                    "description": self.render(()),
                }
            )

        for step_index in range(1, len(self.steps) + 1):
            active_steps = self.steps[:step_index]
            step = active_steps[-1]
            records.append(
                {
                    "step": step_index,
                    "feature": step.feature,
                    "added_phrase": step.phrase,
                    "description": self.render(active_steps),
                }
            )

        return pd.DataFrame.from_records(records)

    def render(self, active_steps: Sequence[FeaturePromptStep]) -> str:
        subject_values = {}
        details = list(self.base_details)

        for step in active_steps:
            if step.role == "subject":
                subject_values[step.feature] = step.phrase
            else:
                details.append(step.phrase)

        subject_terms = [
            subject_values[feature]
            for feature in self.subject_order
            if feature in subject_values
        ]
        subject = " ".join(subject_terms + [self.fallback_subject])
        sentence = f"{self.prefix} {_article_for(subject)} {subject}"
        if details:
            sentence += " with " + ", ".join(details)
        return sentence + "."


@dataclass
class ClipRetrievalProbe:
    """Run and visualize CLIP text-to-image retrieval probes over one manifest."""

    index: ClipIndex
    encoder: SentenceTransformerClipEncoder
    root: str | Path | None = None
    batch_size: int = 32

    @classmethod
    def from_manifest(
        cls,
        manifest: pd.DataFrame,
        index_dir: str | Path,
        encoder: SentenceTransformerClipEncoder,
        root: str | Path | None = None,
        batch_size: int = 32,
        show_progress: bool = False,
        rebuild_index: bool = False,
    ) -> "ClipRetrievalProbe":
        index = ensure_clip_index(
            manifest=manifest,
            index_dir=index_dir,
            encoder=encoder,
            batch_size=batch_size,
            show_progress=show_progress,
            rebuild=rebuild_index,
        )
        return cls(index=index, encoder=encoder, root=root, batch_size=batch_size)

    @property
    def manifest(self) -> pd.DataFrame:
        return self.index.manifest

    def find_target(
        self,
        *,
        image_id: str | None = None,
        target_id: str | None = None,
    ) -> pd.Series:
        return find_manifest_row(self.manifest, image_id=image_id, target_id=target_id)

    def run_ladder(
        self,
        ladder: DescriptionLadder | pd.DataFrame,
        *,
        target_image_id: str | None = None,
        target_id: str | None = None,
        top_k: int = 9,
        candidate_columns: Sequence[str] = ("expression",),
        show_progress: bool = False,
    ) -> pd.DataFrame:
        ladder_frame = _ladder_frame(ladder)
        results = self.index.search_texts(
            ladder_frame["description"].astype(str).tolist(),
            encoder=self.encoder,
            top_k=top_k,
            batch_size=self.batch_size,
            show_progress=show_progress,
        )
        return self._annotate_results(
            results=results,
            ladder_frame=ladder_frame,
            target_image_id=target_image_id,
            target_id=target_id,
            candidate_columns=candidate_columns,
        )

    def show_stage(
        self,
        ladder: DescriptionLadder | pd.DataFrame,
        stage_feature: str,
        *,
        target_image_id: str | None = None,
        target_id: str | None = None,
        top_k: int = 9,
        cols: int = 3,
        target_position: Literal["left", "top", "none"] = "left",
        candidate_columns: Sequence[str] = ("expression",),
        display_heading: bool = False,
        display_figure: bool = False,
        show_description_title: bool = True,
        show_table: bool = True,
    ):
        ladder_frame = _ladder_frame(ladder)
        matches = ladder_frame.loc[ladder_frame["feature"] == stage_feature]
        if matches.empty:
            valid = ", ".join(ladder_frame["feature"].astype(str).tolist())
            raise ValueError(f"Unknown stage_feature={stage_feature!r}. Valid values: {valid}")

        stage_row = matches.iloc[0]
        target_row = self.find_target(image_id=target_image_id, target_id=target_id)
        if display_heading:
            _display_stage_heading(stage_row)

        results, fig = search_and_show_lineup(
            index=self.index,
            encoder=self.encoder,
            description=stage_row["description"],
            target_image_path=target_row["image_path"],
            root=self.root,
            top_k=top_k,
            cols=cols,
            target_position=target_position,
            batch_size=self.batch_size,
            show_description_title=show_description_title,
        )
        annotated = self._annotate_results(
            results=results,
            ladder_frame=pd.DataFrame([stage_row]),
            target_image_id=str(target_row["image_id"]),
            target_id=str(target_row["target_id"]),
            candidate_columns=candidate_columns,
        )

        if display_figure:
            _display_figure(fig)

        if show_table:
            try:
                from IPython.display import display

                display(_display_columns(annotated))
            except ImportError:
                pass

        return annotated, fig

    def _annotate_results(
        self,
        *,
        results: pd.DataFrame,
        ladder_frame: pd.DataFrame,
        target_image_id: str | None,
        target_id: str | None,
        candidate_columns: Sequence[str],
    ) -> pd.DataFrame:
        annotated = results.copy()
        metadata = ladder_frame.reset_index(drop=True)
        for column in ("step", "feature", "added_phrase", "description"):
            if column in metadata.columns:
                annotated[column] = annotated["query_index"].map(metadata[column].to_dict())

        if target_image_id is None and target_id is None:
            for column in candidate_columns:
                if column in self.manifest.columns:
                    lookup = self.manifest.set_index("image_id")[column].to_dict()
                    annotated[f"candidate_{column}"] = annotated["image_id"].map(lookup)
            return annotated

        target_row = self.find_target(image_id=target_image_id, target_id=target_id)
        annotated["target_image_id"] = str(target_row["image_id"])
        annotated["target_identity_id"] = str(target_row["target_id"])
        annotated["is_target_image"] = annotated["image_id"].astype(str).eq(str(target_row["image_id"]))
        annotated["is_target_identity"] = annotated["target_id"].astype(str).eq(str(target_row["target_id"]))

        for column in candidate_columns:
            if column in self.manifest.columns:
                lookup = self.manifest.set_index("image_id")[column].to_dict()
                annotated[f"candidate_{column}"] = annotated["image_id"].map(lookup)

        return annotated


def ensure_clip_index(
    manifest: pd.DataFrame,
    index_dir: str | Path,
    encoder: SentenceTransformerClipEncoder,
    batch_size: int = 32,
    show_progress: bool = False,
    rebuild: bool = False,
) -> ClipIndex:
    """Load a CLIP index if present, otherwise build and save one."""
    root = Path(index_dir)
    embeddings_path = root / "image_embeddings.npy"
    manifest_path = root / "manifest.csv"
    if embeddings_path.exists() and manifest_path.exists() and not rebuild:
        return ClipIndex.load(root)

    index = ClipIndex.build(
        manifest=manifest,
        encoder=encoder,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    index.save(root)
    return index


def filter_expression_images(
    manifest: pd.DataFrame,
    expression_col: str = "expression",
    neutral_codes: Sequence[str] = ("", "N", "NEUTRAL"),
) -> pd.DataFrame:
    """Return manifest rows whose expression code is not neutral."""
    if expression_col not in manifest.columns:
        raise ValueError(f"Manifest is missing expression column: {expression_col}")
    neutral = {code.upper() for code in neutral_codes}
    expression = manifest[expression_col].fillna("").astype(str).str.upper()
    return manifest.loc[~expression.isin(neutral)].reset_index(drop=True)


def find_manifest_row(
    manifest: pd.DataFrame,
    *,
    image_id: str | None = None,
    target_id: str | None = None,
) -> pd.Series:
    """Find exactly one manifest row by image_id or target_id."""
    if image_id is None and target_id is None:
        raise ValueError("Provide image_id or target_id.")

    if image_id is not None:
        if "image_id" not in manifest.columns:
            raise ValueError("Manifest is missing image_id.")
        variants = _image_id_variants(image_id)
        matches = manifest.loc[manifest["image_id"].astype(str).isin(variants)]
        if not matches.empty:
            return matches.iloc[0]

    if target_id is not None:
        if "target_id" not in manifest.columns:
            raise ValueError("Manifest is missing target_id.")
        matches = manifest.loc[manifest["target_id"].astype(str) == str(target_id)]
        if not matches.empty:
            return matches.iloc[0]

    raise ValueError(f"No manifest row found for image_id={image_id!r}, target_id={target_id!r}.")


def _coerce_step(
    step: FeaturePromptStep | tuple[str, str] | tuple[str, str, PromptRole],
) -> FeaturePromptStep:
    if isinstance(step, FeaturePromptStep):
        return step
    if len(step) == 2:
        feature, phrase = step
        return FeaturePromptStep(feature=feature, phrase=phrase)
    if len(step) == 3:
        feature, phrase, role = step
        return FeaturePromptStep(feature=feature, phrase=phrase, role=role)
    raise ValueError(f"Cannot interpret prompt step: {step!r}")


def _ladder_frame(ladder: DescriptionLadder | pd.DataFrame) -> pd.DataFrame:
    frame = ladder.to_frame() if isinstance(ladder, DescriptionLadder) else ladder.copy()
    required = {"feature", "description"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ladder is missing required columns: {missing}")
    return frame.reset_index(drop=True)


def _article_for(text: str) -> str:
    return "an" if text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _image_id_variants(image_id: str) -> set[str]:
    stem = Path(str(image_id)).stem
    variants = {stem}
    if stem.upper().startswith("CFD-"):
        variants.add(stem[4:])
    else:
        variants.add(f"CFD-{stem}")
    return variants


def _display_columns(results: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "rank",
        "target_id",
        "image_id",
        "candidate_expression",
        "clip_score",
        "is_target_image",
        "is_target_identity",
    ]
    columns = [column for column in preferred if column in results.columns]
    return results[columns] if columns else results


def _display_stage_heading(stage_row: pd.Series) -> None:
    step = int(stage_row["step"]) if "step" in stage_row and pd.notna(stage_row["step"]) else 0
    feature = stage_row.get("feature", "")
    added_phrase = stage_row.get("added_phrase", "")
    description = stage_row.get("description", "")
    text = f"**Step {step:02d}: {feature} -> {added_phrase}**  \n{description}"
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text.replace("**", ""))


def _display_figure(fig) -> None:
    try:
        from IPython.display import display

        display(fig)
    except ImportError:
        pass
