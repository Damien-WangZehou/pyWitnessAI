"""Download and verify the frozen study model before starting the live service.

Run ``python -m studies.qualtrics_cfd_clip.bootstrap_model`` with
``CFD_CLIP_LOCAL_MODEL_PATH`` set to a directory named after the pinned revision.
The final snapshot contains only the allowlisted model files. Hub cache data and
the cross-process lock live outside it, so the live selector's tree hash remains
unchanged. An existing snapshot is verified offline and is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "studies.qualtrics_cfd_clip"

import re
import shutil
import tempfile
import time
from typing import Any, Sequence

from filelock import FileLock

from pyWitnessAI.clip import MODEL_SNAPSHOT_HASH_VERSION, directory_content_sha256
from pyWitnessAI.model_files import file_sha256
from .profile import DEFAULT_PROFILE_PATH, LiveSelectionProfile


DEFAULT_MANIFEST_PATH = Path(__file__).with_name("model_download_manifest.json")
LOGGER = logging.getLogger(__name__)


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1:
        raise ValueError("Unsupported model download manifest version.")
    if manifest.get("snapshot_hash_version") != MODEL_SNAPSHOT_HASH_VERSION:
        raise ValueError("Unsupported model snapshot hash version.")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("revision", ""))):
        raise ValueError("Model download revision must be a full commit SHA.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Model download manifest must list its required files.")
    names = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("Invalid model download file record.")
        name = record.get("path")
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or ":" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name
            or name == "."
            or name.casefold() in names
        ):
            raise ValueError(f"Unsafe or duplicate model download path: {name!r}.")
        names.add(name.casefold())
        if type(record.get("size")) is not int or record["size"] < 0:
            raise ValueError(f"Invalid file size for {name}.")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
            raise ValueError(f"Invalid SHA-256 for {name}.")

    # This is the same name/size/content-digest contract used by clip.py.
    digest = hashlib.sha256(f"{MODEL_SNAPSHOT_HASH_VERSION}\0".encode("utf-8"))
    for record in sorted(files, key=lambda item: item["path"]):
        digest.update(
            f"{record['path']}\0{record['size']}\0{record['sha256']}\n".encode("utf-8")
        )
    if digest.hexdigest() != manifest.get("snapshot_sha256"):
        raise ValueError("Model file allowlist does not match its snapshot SHA-256.")
    return manifest


def _verify_file(path: Path, record: dict[str, Any]) -> None:
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError(f"Model file missing or wrong size: {record['path']}.")
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"Model file SHA-256 mismatch: {record['path']}.")


def _verify_snapshot(root: Path, manifest: dict[str, Any]) -> None:
    if not root.is_dir():
        raise ValueError(f"Model snapshot path is not a directory: {root}.")
    expected = {record["path"] for record in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not path.is_dir()
    }
    if actual != expected:
        raise ValueError(
            "Model snapshot file list mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )
    for record in manifest["files"]:
        _verify_file(root / record["path"], record)
    if directory_content_sha256(root) != manifest["snapshot_sha256"]:
        raise ValueError("Model snapshot SHA-256 does not match the frozen profile.")


def _hub_download(**kwargs: Any) -> str:
    # Import lazily: verified existing snapshots require no hub or network access.
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def _download_file(
    record: dict[str, Any],
    manifest: dict[str, Any],
    cache_dir: Path,
    max_attempts: int,
    retry_delay: float,
) -> Path:
    force_download = False
    for attempt in range(1, max_attempts + 1):
        try:
            downloaded = Path(
                _hub_download(
                    repo_id=manifest["repo_id"],
                    revision=manifest["revision"],
                    filename=record["path"],
                    cache_dir=str(cache_dir),
                    force_download=force_download,
                    library_name="pyWitnessAI",
                )
            )
            try:
                _verify_file(downloaded, record)
            except ValueError:
                # A damaged hub cache may be repaired, but never the final snapshot.
                force_download = True
                raise
            return downloaded
        except ImportError as exc:
            raise ImportError(
                'Automatic model download requires pip install "pyWitnessAI[clip,excel,api]".'
            ) from exc
        except Exception as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Could not obtain verified model file {record['path']} after "
                    f"{max_attempts} attempts. The download cache is retained; restart "
                    "to retry. The live service has not been started."
                ) from exc
            LOGGER.warning(
                "Retrying %s (%s/%s): %s", record["path"], attempt, max_attempts, exc
            )
            time.sleep(min(retry_delay * 2 ** (attempt - 1), 60))
    raise AssertionError("Unreachable download state")


def bootstrap_model(
    model_path: str | Path,
    *,
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
    cache_dir: str | Path | None = None,
    max_attempts: int = 3,
    retry_delay: float = 2,
    lock_timeout: float = 3600,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Return a verified snapshot, downloading only when the destination is absent.

    Interrupted HTTP transfers remain in the hub cache for the next attempt.
    Model files are copied into a private sibling temporary directory and checked
    before an atomic rename. Concurrent starts share a lock and recheck the
    destination after acquiring it. Existing corrupt snapshots require explicit
    operator repair; this function never deletes or replaces them.
    """
    if max_attempts < 1 or retry_delay < 0 or lock_timeout < 0:
        raise ValueError("Invalid download attempts, retry delay, or lock timeout.")
    manifest = _load_manifest(manifest_path)
    profile = LiveSelectionProfile.load(profile_path)
    expected_contract = {
        "repo_id": f"sentence-transformers/{profile.model_name}",
        "revision": profile.model_revision,
        "snapshot_sha256": profile.model_snapshot_sha256,
    }
    for field_name, expected in expected_contract.items():
        if manifest.get(field_name) != expected:
            raise ValueError(f"Model download manifest {field_name} differs from the frozen profile.")

    target = Path(model_path).expanduser().resolve()
    if target.name != profile.model_revision:
        raise ValueError(
            "CFD_CLIP_LOCAL_MODEL_PATH must end with the frozen revision "
            f"{profile.model_revision}."
        )
    cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else target.parent / ".pywitnessai-model-downloads"
    )
    if cache == target or target in cache.parents:
        raise ValueError("Model download cache must be outside the final snapshot directory.")

    def reuse_existing() -> bool:
        if not target.exists():
            return False
        LOGGER.info("Verifying existing frozen CLIP model: %s", target)
        try:
            _verify_snapshot(target, manifest)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Existing model snapshot is invalid: {target}. It has been left "
                "unchanged; restore the exact frozen files or choose a new empty "
                "parent directory. " + str(exc)
            ) from exc
        return True

    # Permit a valid snapshot mounted read-only, without even writing a lock file.
    if reuse_existing():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.bootstrap.lock"
    LOGGER.info("Waiting for model bootstrap lock: %s", lock_path)
    with FileLock(str(lock_path), timeout=lock_timeout):
        if reuse_existing():
            return target
        cache.mkdir(parents=True, exist_ok=True)
        # TemporaryDirectory owns only this newly created sibling, never user data.
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.bootstrap-", dir=target.parent
        ) as temporary:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            for record in manifest["files"]:
                LOGGER.info("Preparing frozen CLIP file: %s", record["path"])
                downloaded = _download_file(
                    record, manifest, cache, max_attempts, retry_delay
                )
                destination = staging / record["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(downloaded, destination)
            _verify_snapshot(staging, manifest)
            # Do not overwrite a directory created by an external process either.
            if target.exists():
                raise FileExistsError(f"Model snapshot appeared during bootstrap: {target}.")
            staging.rename(target)
    LOGGER.info("Frozen CLIP model is ready: %s", target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.getenv("CFD_CLIP_LOCAL_MODEL_PATH"))
    parser.add_argument(
        "--profile", default=os.getenv("CFD_CLIP_SELECTION_PROFILE") or str(DEFAULT_PROFILE_PATH)
    )
    parser.add_argument("--cache-dir", default=os.getenv("CFD_CLIP_MODEL_CACHE"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2)
    parser.add_argument("--lock-timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    if not args.model_path:
        parser.error("set CFD_CLIP_LOCAL_MODEL_PATH or pass --model-path")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        bootstrap_model(
            args.model_path,
            profile_path=args.profile,
            cache_dir=args.cache_dir,
            max_attempts=args.max_attempts,
            retry_delay=args.retry_delay,
            lock_timeout=args.lock_timeout,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        LOGGER.error("Model bootstrap failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
