"""Optional local face restoration; never certifies identity or lineup suitability."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .FaceStandardizer import _eye_transform, _warp
from .API.models import download_model

__all__ = ["CodeFormerEnhancer"]

_REVISION = "b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b"
_WEIGHTS_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
_WEIGHTS_SHA256 = "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7"
# Hash the raw GitHub download bytes (LF), not a Windows Git checkout (CRLF).
_SOURCES = {
    "LICENSE": "cfd654022bdc44fdd809670d50239bcb1ebf2f4a661dca7e328f93c269538246",
    "basicsr/utils/registry.py": "21c23bd3bf727eb2decd99035120fc47316db92486d73116cca18e9fa8b6c854",
    "basicsr/archs/vqgan_arch.py": "5ef00125ce448e34c9c02655bdebff5dca51589597c7c43a0b4bf2326d32e8cb",
    "basicsr/archs/codeformer_arch.py": "dfccf9e9e6cd58b933301350e956418d1ffb15bb601ef5ce16d6dda3f01d020a",
}


class CodeFormerEnhancer:
    """Restore a softly blended face region using original eye measurements.

    Pass an instance as ``FaceStandardizer(enhancer=CodeFormerEnhancer())``.
    Fidelity 1 and strength 0.5 are conservative starting parameters, not an
    identity guarantee. Use ``enhancer=None`` to disable restoration entirely.
    Files download once on first inference; photos never leave the machine.
    Upstream code/weights use the S-Lab License 1.0 (non-commercial use).
    """

    def __init__(self, *, fidelity: float = 1.0, strength: float = 0.5,
                 device: str = "cpu", model_dir: str | Path | None = None):
        if not np.isfinite(fidelity) or not 0 <= fidelity <= 1:
            raise ValueError("fidelity must be finite and between 0 and 1.")
        if not np.isfinite(strength) or not 0 < strength <= 1:
            raise ValueError("strength must be finite, greater than 0 and at most 1.")
        if device != "cpu" and device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("device must be cpu, cuda, or cuda:<index>.")
        self.fidelity, self.strength, self.device = float(fidelity), float(strength), device
        self.model_dir = Path(model_dir or Path.home() / ".cache" / "pyWitnessAI" / "portraits" / "codeformer")
        self._model = None

    def prepare(self) -> None:
        """Check dependencies and pinned model files before starting a batch."""
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise ImportError('Install AI restoration with pip install "pyWitnessAI[enhancement]".') from exc
        if self.device.startswith("cuda"):
            index = int(self.device.split(":")[1]) if ":" in self.device else torch.cuda.current_device() if torch.cuda.is_available() else 0
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError(f"Requested device is unavailable: {self.device}")
        source_dir = self.model_dir / _REVISION
        for relative, checksum in _SOURCES.items():
            download_model(source_dir / relative,
                           f"https://raw.githubusercontent.com/sczhou/CodeFormer/{_REVISION}/{relative}", checksum)
        checkpoint = download_model(self.model_dir / "codeformer.pth", _WEIGHTS_URL, _WEIGHTS_SHA256)
        model_class = _load_architecture(source_dir)
        model = model_class(dim_embd=512, codebook_size=1024, n_head=8,
                            n_layers=9, connect_list=["32", "64", "128", "256"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["params_ema"]
        model.load_state_dict(state, strict=True)
        self._model = model.eval().to(self.device)

    def __call__(self, image: Image.Image, *, landmarks) -> Image.Image:
        """Return same-size RGB pixels; landmarks use oriented input coordinates."""
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("CodeFormerEnhancer expects a PIL RGB image.")
        points = np.asarray(landmarks, dtype=float)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise ValueError("Expected five finite facial landmarks.")
        if (points < 0).any() or (points >= image.size).any():
            raise ValueError("Landmarks must lie inside the image.")
        eyes = points[:2][np.argsort(points[:2, 0])]
        if np.linalg.norm(eyes[1] - eyes[0]) < 4:
            raise ValueError("Eye landmarks are too close to align.")
        # CodeFormer canonical eye locations; similarity alignment keeps proportions.
        matrix = _eye_transform(eyes, 125.92139, (255.942075, 240.07034))
        aligned = _warp(image, matrix, (512, 512), (224, 224, 224))
        restored = self._restore(aligned)
        if not isinstance(restored, Image.Image) or restored.mode != "RGB" or restored.size != (512, 512):
            raise ValueError("CodeFormer must produce a 512x512 RGB image.")
        inverse = np.linalg.inv(np.vstack((matrix, [0, 0, 1])))[:2]
        # Reproject only the model's changes: an identity restorer must not blur
        # the original through an unnecessary round trip of resampling.
        residual = np.asarray(restored, dtype=np.float32) - np.asarray(aligned, dtype=np.float32)
        change = np.stack([np.asarray(_warp(Image.fromarray(residual[:, :, channel]), inverse, image.size, 0))
                           for channel in range(3)], axis=2)
        # Keep frame/background/hair outside this feathered face region unchanged.
        region = Image.new("L", (512, 512))
        ImageDraw.Draw(region).ellipse((100, 150, 412, 475), fill=255)
        region = region.filter(ImageFilter.GaussianBlur(16))
        region = _warp(region, inverse, image.size, 0)
        blend = np.asarray(region, dtype=np.float32)[:, :, None] / 255 * self.strength
        pixels = np.asarray(image, dtype=np.float32) + change * blend
        return Image.fromarray(pixels.round().clip(0, 255).astype(np.uint8))

    def _restore(self, aligned: Image.Image) -> Image.Image:
        self.prepare()
        import torch
        pixels = np.asarray(aligned, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
        tensor = torch.from_numpy(pixels.copy()).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            restored = self._model(tensor, w=self.fidelity, adain=True)[0]
        if tuple(restored.shape) != (1, 3, 512, 512) or not torch.isfinite(restored).all():
            raise ValueError("CodeFormer returned invalid or non-finite pixels.")
        pixels = restored[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(((pixels + 1) * 127.5).round().astype(np.uint8))

    def recipe(self) -> dict[str, object]:
        """Serializable settings; no image upload, model load, or download."""
        recipe = dict(model="CodeFormer", revision=_REVISION, weights_sha256=_WEIGHTS_SHA256,
                      source_sha256=dict(_SOURCES), fidelity=self.fidelity, strength=self.strength,
                      device=self.device, alignment="original-eye-similarity-512",
                      blend="soft-face-residual-v1", identity_preservation="unverified",
                      license="S-Lab License 1.0; non-commercial")
        if self._model is not None:
            import torch
            recipe["torch"] = torch.__version__
        return recipe


def _load_architecture(source_dir: Path):
    """Load three verified upstream files without BasicSR's training dependencies.

    Rewrite only absolute package imports in memory, leaving cached source and
    license intact. A private namespace avoids replacing an installed basicsr.
    """
    namespace = f"_pywitnessai_codeformer_{uuid4().hex}"
    try:
        for suffix in ("", ".archs", ".utils"):
            module = ModuleType(namespace + suffix)
            module.__path__ = []
            sys.modules[module.__name__] = module
        sys.modules[namespace + ".utils"].get_root_logger = lambda: logging.getLogger(namespace)
        for relative in list(_SOURCES)[1:]:
            path = source_dir / relative
            name = namespace + "." + relative.removeprefix("basicsr/")[:-3].replace("/", ".")
            module = ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            source = path.read_text(encoding="utf-8").replace("from basicsr.", f"from {namespace}.")
            exec(compile(source, str(path), "exec"), module.__dict__)
        return module.CodeFormer
    except BaseException:
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]
        raise
