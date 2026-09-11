"""Compose existing face tools and record a witness's successive choices.

create_lineup builds one research lineup. LineupSession uses the same tools for
neutral candidate portraits: a witness's confirmation never establishes TP/TA.
No web framework or model is loaded until its feature is requested.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from .FaceGenerator import FaceGenerator
from .LineupBuilder import Lineup, LineupBuilder, LineupMember

__all__ = ["create_lineup", "LineupSession", "LineupRound"]


def create_lineup(
    *, suspect=None, description=None, reference_image=None, filler_pool=None,
    target=None, size=6, rows=2, order="random", seed=None, similarity=75,
    generator_backend="openai", output_dir="./facesGenerated", parser="raw",
    generation_options=None,
) -> Lineup:
    """Build from an existing pool, or generate the required number of fillers.

    suspect is a member; reference_image is only a generation condition. For TA,
    omit suspect or supply an explicitly innocent suspect with target='absent'.
    Description and reference may be combined. A supplied pool is used directly;
    use FillerSelector first if it needs description-based selection.
    generation_options accepts model, size, quality and output_format.
    """
    builder = LineupBuilder(suspect, filler_pool, target=target, size=size,
                            rows=rows, order=order, seed=seed)
    if filler_pool is not None:
        if description is not None or reference_image is not None:
            raise ValueError("Use filler_pool or generation conditions, not both; select a pool before building.")
        return builder.build()
    if suspect is not None:
        _read_portrait(suspect)
    generator = _generator(
        description=description, reference_image=reference_image,
        n=size - int(suspect is not None), output_dir=output_dir,
        similarity=similarity, generator_backend=generator_backend,
        parser=parser, seed=seed, generation_options=generation_options,
    )
    builder.filler_pool = [face.path for face in generator.generate()]
    return builder.build()


@dataclass(frozen=True)
class LineupRound:
    """One fixed presentation and its optional, one-based witness response."""

    number: int
    lineup: Lineup
    directory: Path
    feedback: dict | None = None


class LineupSession:
    """A small, file-backed portrait refinement session.

    start() is repeatable. respond() records a round-specific response and
    either confirms it or generates the next round. The previous choice remains
    as an unchanged displayed image. Call the same response again after a
    generation failure; a live session reuses its partial generation batch.

    load() restores saved presentations and responses. An unfinished generation
    batch is not restored across processes and will start a new batch on retry.
    Instances are synchronous; callers must serialize access to one session.
    """

    def __init__(
        self, description: str, *, output_dir, reference_image=None, size=6,
        rows=2, similarity=75, max_rounds=10, seed=None, parser="raw",
        generator_backend="openai", generation_options=None,
    ):
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string.")
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer.")
        if not isinstance(parser, str):
            raise TypeError("A saved session requires a named parser.")
        LineupBuilder(None, [], size=size, rows=rows, seed=seed, target="unknown")
        self.output_dir = Path(output_dir).resolve()
        self.description = description
        self.reference_image = str(Path(reference_image).resolve()) if reference_image is not None else None
        self.size, self.rows, self.similarity = size, rows, similarity
        self.max_rounds, self.seed, self.parser = max_rounds, seed, parser
        self.generator_backend = generator_backend
        self.generation_options = dict(generation_options or {})
        # Validate the full generation plan before creating a session directory.
        self._pending_generator = self._new_generator(self.reference_image, size)
        if not self._pending_generator.backend.supports_reference_image:
            raise ValueError("LineupSession requires a backend that supports reference images.")
        self._history: list[LineupRound] = []
        self.status = "new"
        self._owns_directory = False

    @property
    def history(self) -> tuple[LineupRound, ...]:
        return tuple(self._history)

    @property
    def current(self) -> LineupRound | None:
        return self._history[-1] if self._history else None

    @property
    def confirmed_face(self) -> Path | None:
        if self.status != "confirmed":
            return None
        return self._chosen_path(self.current)

    def start(self) -> LineupRound:
        """Create the first round, or return the existing current presentation."""
        if self.current is not None:
            return self.current
        if self.status != "new":
            raise ValueError("This session has already stopped.")
        self._prepare_directory()
        return self._advance()

    def respond(self, choice: int, *, confirmed: bool = False, round_number: int) -> LineupRound:
        """Confirm a face, or reject the lineup and use choice as the closest face.

        round_number is mandatory so an old page cannot select a new face at the
        same position. Identical retries never consume another completed round.
        """
        if isinstance(round_number, bool) or not isinstance(round_number, int) or not 1 <= round_number <= len(self._history):
            raise ValueError("round_number must identify an existing round.")
        if isinstance(choice, bool) or not isinstance(choice, int) or not 1 <= choice <= self.size:
            raise ValueError("choice must be a one-based member position.")
        if not isinstance(confirmed, bool):
            raise ValueError("confirmed must be a boolean.")
        previous = self._history[round_number - 1]
        feedback = {"choice": choice, "confirmed": confirmed}
        if previous.feedback is not None:
            if previous.feedback != feedback:
                raise ValueError("This round already has a different response.")
            if previous is self.current and self.status == "active":
                return self._advance()
            return self.current
        if self.status != "active" or previous is not self.current:
            raise ValueError("Only the active current round can receive a response.")
        old_status = self.status
        self._history[-1] = replace(previous, feedback=feedback)
        if confirmed:
            self.status = "confirmed"
        elif len(self._history) >= self.max_rounds:
            self.status = "limit_reached"
        try:
            self._save()
        except BaseException:
            self._history[-1] = previous
            self.status = old_status
            raise
        if self.status == "active":
            return self._advance()
        return self.current

    def stop(self) -> None:
        """Stop without confirmation; repeated stops preserve the final outcome."""
        if self.status in {"new", "active"}:
            self._prepare_directory()
            old_status = self.status
            self.status = "stopped"
            try:
                self._save()
            except BaseException:
                self.status = old_status
                raise

    def _new_generator(self, reference, count):
        return _generator(
            description=self.description, reference_image=reference, n=count,
            output_dir=self.output_dir / "generated", similarity=self.similarity,
            generator_backend=self.generator_backend, parser=self.parser,
            seed=self.seed, generation_options=self.generation_options,
        )

    def _advance(self) -> LineupRound:
        retained = self._chosen_path(self.current) if self.current is not None else None
        if self._pending_generator is None:
            reference = retained or self.reference_image
            self._pending_generator = self._new_generator(reference, self.size - int(retained is not None))
        faces = self._pending_generator.generate()
        paths = ([retained] if retained is not None else []) + [face.path for face in faces]
        number = len(self._history) + 1
        lineup = LineupBuilder(
            None, paths, target="unknown", size=self.size, rows=self.rows,
            seed=None if self.seed is None else self.seed + number - 1,
        ).build()
        directory = self.output_dir / f"round_{number:04d}_{uuid4().hex[:12]}"
        lineup.save(directory)
        # Subsequent choices refer to saved display pixels, independent of any
        # mutable input photographs or generation output files.
        lineup = _load_presentation(directory, lineup.to_dict(include_roles=True))
        new_round = LineupRound(number, lineup, directory)
        self._history.append(new_round)
        old_status = self.status
        self.status = "active"
        try:
            self._save()
        except BaseException:
            self._history.pop()
            self.status = old_status
            raise
        self._pending_generator = None
        return new_round

    @staticmethod
    def _chosen_path(round_: LineupRound) -> Path:
        return round_.directory / f"member_{round_.feedback['choice']:02d}.png"

    def _prepare_directory(self):
        if self._owns_directory:
            return
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError("Use a new or empty session output_dir, or LineupSession.load().")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._owns_directory = True
        if self.reference_image is not None:
            reference = self.output_dir / "reference.png"
            _read_portrait(self.reference_image).save(reference)
            self.reference_image = str(reference)
            self._pending_generator = self._new_generator(reference, self.size)
        self._save()

    def to_dict(self, *, include_history=True) -> dict:
        """Private audit, including provenance. Do not serve it to participants."""
        result = {
            "version": 1, "status": self.status,
            "config": {
                "description": self.description, "reference_image": self.reference_image,
                "size": self.size, "rows": self.rows, "similarity": self.similarity,
                "max_rounds": self.max_rounds, "seed": self.seed, "parser": self.parser,
                "generation_options": self.generation_options,
            },
        }
        if include_history:
            result["rounds"] = [
                {"number": item.number, "directory": item.directory.name,
                 "feedback": item.feedback, "lineup": item.lineup.to_dict(include_roles=True)}
                for item in self._history
            ]
        return deepcopy(result)

    def _save(self):
        destination = self.output_dir / "session.json"
        pending = destination.with_suffix(".tmp")
        pending.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(destination)

    @classmethod
    def load(cls, output_dir, *, generator_backend="openai") -> LineupSession:
        """Resume saved rounds. Supply the same provider/backend configuration."""
        root = Path(output_dir).resolve()
        state = json.loads((root / "session.json").read_text(encoding="utf-8"))
        if state.get("version") != 1:
            raise ValueError("Unsupported session version.")
        session = cls(output_dir=root, generator_backend=generator_backend, **state["config"])
        session._owns_directory = True
        session._pending_generator = None
        session.status = state["status"]
        if session.status not in {"new", "active", "confirmed", "stopped", "limit_reached"}:
            raise ValueError("Invalid saved session status.")
        for index, record in enumerate(state["rounds"], 1):
            directory = (root / record["directory"]).resolve()
            if not directory.is_relative_to(root) or record["number"] != index:
                raise ValueError("Invalid saved round directory or number.")
            lineup = _load_presentation(directory, record["lineup"])
            if lineup.target != "unknown" or len(lineup) != session.size:
                raise ValueError("Saved session rounds must be neutral and retain their size.")
            session._history.append(LineupRound(index, lineup, directory, record["feedback"]))
        if (session.status in {"active", "confirmed", "limit_reached"}) and not session.current:
            raise ValueError("The saved session has no current round.")
        return session


def _generator(*, generation_options=None, **kwargs):
    options = dict(generation_options or {})
    unexpected = options.keys() - {"model", "size", "quality", "output_format"}
    if unexpected:
        raise ValueError(f"Unsupported generation_options: {sorted(unexpected)}")
    json.dumps(options)  # Saved configuration must also work outside this process.
    reference = kwargs.get("reference_image")
    if reference is not None:
        _read_portrait(reference)
    generator = FaceGenerator(**kwargs, **options)
    generator.generation_prompts()
    if reference is not None and not generator.backend.supports_reference_image:
        raise ValueError("The image backend does not support reference images.")
    return generator


def _read_portrait(source):
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _load_presentation(directory: Path, record: dict) -> Lineup:
    members = []
    for index, member in enumerate(record["members"], 1):
        path = directory / f"member_{index:02d}.png"
        metadata = dict(member.get("metadata", {}))
        metadata.setdefault("original_source_path", member.get("source_path"))
        members.append(LineupMember(_read_portrait(path), member["role"], str(path), metadata))
    return Lineup(tuple(members), target=record["target"], rows=record["rows"], seed=record["seed"])
