"""Build one lineup; keep candidate selection independent of presentation.

LineupBuilder samples and labels members. Lineup holds the finished images and
provides the same presentation for notebooks, image export, and video analysis.
No generator, selector, or vision model is loaded during construction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from io import BytesIO
import math
from pathlib import Path
import random
from typing import TYPE_CHECKING, Literal
import json

from PIL import Image, ImageCms, ImageDraw, ImageOps

from .ImageCatalog import image_manifest

if TYPE_CHECKING:
    from .FaceStandardizer import FaceStandardizer

__all__ = ["LineupBuilder", "Lineup", "LineupMember"]


@dataclass(frozen=True)
class LineupMember:
    """One portrait with a research role or neutral candidate label and source."""

    image: Image.Image = field(repr=False)
    role: Literal["guilty_suspect", "innocent_suspect", "filler", "candidate"]
    source_path: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Lineup:
    """A finished lineup returned by LineupBuilder.build().

    Positions are one-based; suspect_position is None for all-filler or unknown
    lineups. Unknown candidates carry no assumption about target presence.
    Member IDs describe positions, never filenames or roles. Standardization
    returns a new lineup; showing/saving never resamples.
    """

    members: tuple[LineupMember, ...]
    target: Literal["present", "absent", "unknown"] = "present"
    rows: int = 2
    seed: int | None = None

    def __post_init__(self):
        _validate_layout(len(self.members), self.rows)
        roles = [member.role for member in self.members]
        if self.target == "unknown":
            if any(role != "candidate" for role in roles):
                raise ValueError("An unknown lineup contains only neutral candidates, without a designated suspect.")
        else:
            expected = _suspect_role(self.target)
            allowed_counts = (1,) if self.target == "present" else (0, 1)
            if roles.count(expected) not in allowed_counts or any(role not in (expected, "filler") for role in roles):
                raise ValueError("A present lineup requires exactly one suspect; an absent lineup allows zero or one innocent suspect. All other members must be fillers.")
        if any(not isinstance(member.image, Image.Image) for member in self.members):
            raise TypeError("Each lineup member must contain a PIL image.")
        object.__setattr__(self, "members", tuple(self.members))

    def __len__(self) -> int:
        return len(self.members)

    @property
    def cols(self) -> int:
        return math.ceil(len(self) / self.rows)

    @property
    def suspect_position(self) -> int | None:
        return next((index for index, member in enumerate(self.members, 1)
                     if member.role in ("guilty_suspect", "innocent_suspect")), None)

    @property
    def images(self) -> dict[str, Image.Image]:
        """Independent sRGB presentation pixels, with transparency on white."""
        return {_member_id(i): _presentation_image(member.image) for i, member in enumerate(self.members, 1)}

    @property
    def roles(self) -> dict[str, str]:
        return {_member_id(i): member.role for i, member in enumerate(self.members, 1)}

    @property
    def source_paths(self) -> dict[str, str | None]:
        return {_member_id(i): member.source_path for i, member in enumerate(self.members, 1)}

    def standardize(self, standardizer: FaceStandardizer) -> Lineup:
        """Apply the same transform to every member, retaining order and QC.

        Any failed transform aborts the operation. Review flags are retained in
        member.metadata['standardization']; they are not silently discarded.
        Start a new version from the original lineup, not transformed pixels.
        """
        if any("standardization" in member.metadata or
               member.metadata.get("source", {}).get("standardization_version") for member in self.members):
            raise ValueError("Standardize the original lineup instead of standardizing it twice.")
        members = []
        for member in self.members:
            result = standardizer.transform(member.image.copy())
            metadata = deepcopy({**member.metadata, "standardization": result.metadata})
            members.append(replace(member, image=result.image.copy(), metadata=metadata))
        return replace(self, members=tuple(members))

    def render(
        self, *, annotate_roles: bool = False, tile_size: tuple[int, int] = (256, 320),
    ) -> Image.Image:
        """Render all members with neutral position numbers into a PIL image."""
        if len(tile_size) != 2:
            raise ValueError("tile_size must contain a width and height.")
        for value in tile_size:
            _positive_integer(value, "tile_size dimension")
        width, height = tile_size
        gap, label_height = 12, 32
        canvas = Image.new("RGB", (self.cols * (width + gap) + gap,
                                   self.rows * (height + label_height + gap) + gap), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (member, image) in enumerate(zip(self.members, self.images.values())):
            x = gap + (index % self.cols) * (width + gap)
            y = gap + (index // self.cols) * (height + label_height + gap)
            portrait = ImageOps.contain(image, tile_size)
            canvas.paste(portrait, (x + (width - portrait.width) // 2,
                                    y + (height - portrait.height) // 2))
            label = str(index + 1)
            if annotate_roles:
                label += f" ({member.role})"
            draw.text((x + width // 2, y + height + label_height // 2), label,
                      fill="black", anchor="mm")
        return canvas

    def show(self, *, annotate_roles: bool = False, display: bool = True) -> Image.Image:
        """Show the rendered lineup in a notebook or the local image viewer."""
        image = self.render(annotate_roles=annotate_roles)
        if display:
            try:
                from IPython import get_ipython
                from IPython.display import display as notebook_display
            except ImportError:
                image.show()
            else:
                if get_ipython() is not None and getattr(get_ipython(), "kernel", None) is not None:
                    notebook_display(image)
                else:
                    image.show()
        return image

    def to_dict(self, *, include_roles: bool = False) -> dict:
        """Return a JSON-ready manifest; research annotations are opt-in.

        Relative member image names correspond to save(). A future platform
        adapter can map those names to hosted URLs without rebuilding the lineup.
        """
        members = []
        for index, member in enumerate(self.members, 1):
            record = {"id": _member_id(index), "position": index, "image": f"{_member_id(index)}.png"}
            if include_roles:
                record.update(role=member.role, source_path=member.source_path, metadata=deepcopy(member.metadata))
            members.append(record)
        result = {"size": len(self), "rows": self.rows, "cols": self.cols, "members": members}
        if include_roles:
            result.update(target=self.target, seed=self.seed)
        return result

    def save(self, output_dir: str | Path) -> Path:
        """Save neutral member PNGs, lineup.png, and lineup.json to a new folder.

        Research annotations are available separately through to_dict() and are
        never included in this participant presentation bundle.
        """
        destination = Path(output_dir).resolve()
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError("Use a new or empty output_dir; existing files are never overwritten.")
        destination.mkdir(parents=True, exist_ok=True)
        for member_id, image in self.images.items():
            portrait = image.copy()
            portrait.info.clear()
            portrait.save(destination / f"{member_id}.png", icc_profile=image.info.get("icc_profile"))
        self.render().save(destination / "lineup.png")
        (destination / "lineup.json").write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8",
        )
        return destination


class LineupBuilder:
    """Build a lineup from a filler pool and an optional suspect.

    Pass suspect as a path, PIL image, or None. With None, all size members are
    fillers; otherwise the pool supplies size-1 fillers. filler_pool accepts a
    directory, iterable of paths, ImageCatalog, or DataFrame with image_path.
    A larger pool is sampled uniformly without replacement. order controls
    presentation only: 'given' keeps any suspect first and selected fillers in
    pool order; 'random' shuffles all members. An exact-size filler set is used
    in full.

    target defaults to 'present' with a suspect and 'absent' without one.
    Explicit 'absent' may include an innocent suspect or consist entirely of
    fillers. Explicit 'present' requires a suspect. No role is guessed from
    filenames, and no filler is implicitly promoted to suspect.
    Use 'unknown' for reconstruction rounds whose identities are unverified;
    all members are neutral candidates and suspect must be None.
    """

    def __init__(
        self, suspect: str | Path | Image.Image | None, filler_pool, *, size: int = 6,
        order: Literal["random", "given"] = "random", rows: int = 2,
        target: Literal["present", "absent", "unknown"] | None = None, seed: int | None = None,
    ) -> None:
        _validate_layout(size, rows)
        if target is None:
            target = "absent" if suspect is None else "present"
        if target == "unknown":
            if suspect is not None:
                raise ValueError("target='unknown' requires suspect=None; supply all candidates through filler_pool.")
        else:
            _suspect_role(target)
        if suspect is None and target == "present":
            raise ValueError("target='present' requires a suspect; use target='absent' or omit target with suspect=None.")
        if order not in ("random", "given"):
            raise ValueError("order must be 'random' or 'given'.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None.")
        self.suspect, self.filler_pool = suspect, filler_pool
        self.size, self.order, self.rows = size, order, rows
        self.target, self.seed = target, seed

    def build(self) -> Lineup:
        """Load and label the selected images; no AI calls or file writes."""
        has_suspect = self.suspect is not None
        source = _source_path(self.suspect) if has_suspect else None
        pool = image_manifest(self.filler_pool)
        if source is not None:
            pool = pool.loc[pool["image_path"].map(Path) != source].reset_index(drop=True)
        needed = self.size - int(has_suspect)
        if len(pool) < needed:
            raise ValueError(f"Need {needed} distinct fillers; found {len(pool)} usable candidates after excluding any supplied suspect.")
        seed = self.seed if self.seed is not None else random.SystemRandom().getrandbits(64)
        rng = random.Random(seed)
        selected = sorted(rng.sample(range(len(pool)), needed))
        members = []
        if has_suspect:
            members.append(LineupMember(_read_image(self.suspect), _suspect_role(self.target),
                                        str(source) if source is not None else None))
        # Normalize pandas/NumPy values, missing labels, and Paths for JSON export.
        records = json.loads(pool.iloc[selected].to_json(orient="records", default_handler=str, double_precision=15))
        for record in records:
            path = record.pop("image_path")
            metadata = {"source": record} if record else {}
            role = "candidate" if self.target == "unknown" else "filler"
            members.append(LineupMember(_read_image(path), role, path, metadata))
        if self.order == "random":
            rng.shuffle(members)
        return Lineup(tuple(members), target=self.target, rows=self.rows, seed=seed)


def _positive_integer(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_layout(size: int, rows: int) -> None:
    _positive_integer(size, "size")
    _positive_integer(rows, "rows")
    if size < 2:
        raise ValueError("size must include at least two lineup members (>= 2).")
    if rows > size:
        raise ValueError("rows cannot exceed lineup size.")


def _suspect_role(target: str) -> str:
    if target not in ("present", "absent"):
        raise ValueError("target must be 'present' or 'absent'.")
    return "guilty_suspect" if target == "present" else "innocent_suspect"


def _member_id(position: int) -> str:
    return f"member_{position:02d}"


def _source_path(image) -> Path | None:
    if isinstance(image, Image.Image):
        filename = getattr(image, "filename", None)
        return Path(filename).resolve() if filename else None
    if not isinstance(image, (str, Path)):
        raise TypeError("suspect must be an image path or PIL image.")
    path = Path(image).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return path


def _read_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).copy()
    with Image.open(image) as opened:
        return ImageOps.exif_transpose(opened).copy()


def _presentation_image(image: Image.Image) -> Image.Image:
    """Use identical RGB pixels for display, member export, and analysis."""
    alpha = image.convert("RGBA").getchannel("A")
    profile = image.info.get("icc_profile")
    if profile:
        rgb = ImageCms.profileToProfile(
            image, ImageCms.ImageCmsProfile(BytesIO(profile)), ImageCms.createProfile("sRGB"), outputMode="RGB",
        )
    else:
        rgb = image.convert("RGB")
    return Image.composite(rgb, Image.new("RGB", image.size, "white"), alpha)
