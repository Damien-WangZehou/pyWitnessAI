import os

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Optional, Literal
from dataclasses import dataclass

from .Images import Images
from .LineupBuilder import Lineup
from .LineupAIDecision import LineupAIDecision


@dataclass
class PipelineConfig:
    analyzer_method: Literal["process", "process_verify", "process_georgia_pipeline"] = "process"
    distance_metric: str = "euclidean_l2"
    backend: str = "opencv"
    model: str = "VGG-Face"
    align: bool = False
    normalization: str = "base"
    enforce_detection: bool = False
    show_progress: bool = False
    # Use full frame or detected faces in each frame of the video to compare with lineup members
    use_full_frame: bool = True
    # Fallback distance when no face detected in a frame, preferably with detect_faces as "mtcnn"
    no_face_fill: float = 50.0


def export_for_pywitness(
    df: pd.DataFrame,
    output_csv: Optional[str] = None,
    targetLineup: Literal["targetPresent", "targetAbsent", "both"] = "both",
    lineupSize: int = 6,
    no_face_fill: Optional[float] = None,
) -> str:
    """
    Convert a long-form pipeline CSV into a minimal pyWitness-ready CSV.

    Input expectations: Per-frame TP/TA response and confidence summary fields.

    Parameters
    ------
    df: Pipeline results, either long or wide.
    output_csv: Where to write the new CSV.
    targetLineup : {'targetPresent','targetAbsent','both'}
        - 'targetPresent'   -> use TP_responseType / TP_conf
        - 'targetAbsent'    -> use TA_responseType / TA_conf
        - 'both'            -> split each input record into TWO output rows (TP and TA)
    lineupSize: Number of compared members. A paired TP/TA comparison containing
        both suspect roles excludes the other suspect from each output lineup.

    Output schema:
        responseType, confidence, targetLineup, lineupSize
    """
    if targetLineup is None:
        targetLineup = "both"
    if targetLineup not in ("targetPresent", "targetAbsent", "both"):
        raise ValueError("targetLineup must be 'targetPresent', 'targetAbsent', or 'both'.")
    if isinstance(lineupSize, bool) or not isinstance(lineupSize, int) or lineupSize < 1:
        raise ValueError("lineupSize must be a positive integer.")
    candidate_columns = any(str(column).endswith(("[candidate]", "[candidate]]")) for column in df.columns)
    candidate_roles = "role" in df.columns and df["role"].eq("candidate").any()
    unknown_target = any(column in df.columns and df[column].eq("unknown").any()
                         for column in ("target", "targetLineup"))
    if candidate_columns or candidate_roles or unknown_target:
        raise ValueError("pyWitness TP/TA export requires known target presence; unknown candidates cannot be exported as research outcomes.")

    # Sanity checks
    prefixes = ("TP", "TA") if targetLineup == "both" else (
        "TP" if targetLineup == "targetPresent" else "TA",
    )
    need_cols = {f"{prefix}_{field}" for prefix in prefixes for field in ("responseType", "conf")}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Paired comparisons may carry either long-form roles or wide distance labels.
    cols = [str(c) for c in df.columns]
    has_guilty = any(c.endswith("[guilty_suspect]") or c.endswith("[guilty_suspect]]") for c in cols)
    has_innocent = any(c.endswith("[innocent_suspect]") or c.endswith("[innocent_suspect]]") for c in cols)

    # When data is not wide-form, check "role" column
    if "role" in df.columns:
        unique_roles = set(df["role"].dropna().unique().tolist())
        has_guilty   = has_guilty   or ("guilty_suspect"   in unique_roles)
        has_innocent = has_innocent or ("innocent_suspect" in unique_roles)

    effective_lineup_size = lineupSize - int(has_guilty and has_innocent)
    if effective_lineup_size < 1:
        raise ValueError("A paired comparison requires both suspects and at least two members.")

    # We want one output row per group (frame, probe)
    group_keys = [c for c in ["frame", "probe"] if c in df.columns]
    summary_columns = [c for c in ("TA_responseType", "TP_responseType", "TA_conf", "TP_conf",
                                   "responseType", "confidence") if c in df.columns]
    if group_keys:
        summary = (
            df.groupby(group_keys, as_index=False)[summary_columns]
              .first()
        )
    else:
        summary = df[summary_columns].drop_duplicates()

    out_rows = []
    for _, row in summary.iterrows():
        # Legacy no-face records may only carry the generic summary fields.
        no_face = all(pd.isna(row.get(column)) for column in (
            "TA_responseType", "TP_responseType", "TA_conf", "TP_conf",
        ))
        for prefix in prefixes:
            response, confidence = row.get(f"{prefix}_responseType"), row.get(f"{prefix}_conf")
            if no_face:
                response, confidence = row.get("responseType"), row.get("confidence")
                response = response if pd.notna(response) else "fillerId"
                if pd.isna(confidence) and response != "rejectId":
                    confidence = no_face_fill if no_face_fill is not None else 50.0
            out_rows.append({
                "responseType": response if pd.notna(response) else None,
                "confidence": float(confidence) if pd.notna(confidence) else None,
                "targetLineup": "targetPresent" if prefix == "TP" else "targetAbsent",
                "lineupSize": effective_lineup_size,
            })

    out = pd.DataFrame(out_rows, columns=["responseType","confidence","targetLineup","lineupSize"])

    if output_csv is None:
        output_csv = "videoLineupResults_pywitness.csv"

    directory = os.path.dirname(output_csv)
    if directory:
        os.makedirs(directory, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv


class VideoLineupPipeline:
    """Compare video faces to a built Lineup and summarize its TP/TA decisions."""

    def __init__(self,
                 video_path: str,
                 lineup: Lineup,
                 cfg: PipelineConfig | dict | None = None):
        if not isinstance(lineup, Lineup):
            raise TypeError("lineup must be a Lineup returned by LineupBuilder.build().")
        if lineup.target == "unknown":
            raise ValueError("VideoLineupPipeline requires known target presence; an unknown reconstruction lineup cannot produce TP/TA statistics.")
        self.video_path = video_path
        self.lineup = lineup

        if cfg is None:
            self.cfg = PipelineConfig()
        elif isinstance(cfg, dict):
            self.cfg = PipelineConfig(**cfg)
        elif isinstance(cfg, PipelineConfig):
            self.cfg = cfg
        else:
            raise TypeError("cfg must be a PipelineConfig, dict, or None")

        self.identifier_obj = LineupAIDecision(targetLineup=self._target_lineup)
        self.identifier_enabled = True

        # Keep presentation order and neutral IDs, including in-memory portraits.
        self._row_il = Images([])
        self._row_il.images = dict(lineup.images)
        self._row_il.path_to_images = {
            name: path or f"{name}.png" for name, path in lineup.source_paths.items()
        }
        self.roles = dict(lineup.roles)

        from .ImagesAI import ImagesAI
        self._analyzer_template = ImagesAI(
            column_images=Images([]),
            row_images=self._row_il,
            distance_metric=self.cfg.distance_metric,
            backend=self.cfg.backend,
            enforce_detection=self.cfg.enforce_detection,
            model=self.cfg.model,
            align=self.cfg.align,
            normalization=self.cfg.normalization,
            show_progress=self.cfg.show_progress
        )

    @property
    def _target_lineup(self) -> str:
        if self.lineup.target == "unknown":
            raise ValueError("TP/TA statistics require known target presence.")
        return "targetPresent" if self.lineup.target == "present" else "targetAbsent"

    def _faces_from_frame(self, frame_bgr: np.ndarray) -> List[Image.Image]:
        """
        Return a list of PIL faces for this frame.
        - "fullframe": treat whole frame as a single face (current default behaviour)
        - "mtcnn": use facenet-pytorch MTCNN; if none found, return []
        """
        img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        if self.cfg.use_full_frame:
            return [pil]

        method = getattr(self.cfg, "analyzer_method", "process")
        backend = getattr(self.cfg, "backend", "mtcnn")

        faces: List[Image.Image] = []

        if method == "process_georgia_pipeline":
            try:
                boxes, probs = self._analyzer_template.mtcnn.detect(pil)
                if boxes is None or probs is None:
                    return []
                # Filter boxes by size/prob if needed
                for (x1, y1, x2, y2), p in zip(boxes, probs):
                    if p is None:
                        continue
                    crop = pil.crop((int(x1), int(y1), int(x2), int(y2))).convert("RGB")
                    faces.append(crop)
            except Exception:
                return []

        else:
            # DeepFace path for process / process_verify / fallback
            from deepface import DeepFace

            try:
                dets = DeepFace.extract_faces(
                    img_path=np.array(pil),
                    detector_backend=backend,
                    enforce_detection=False
                ) or []
                for d in dets:
                    face_arr = d.get("face")
                    region = d.get("facial_area") or d.get("region")
                    if face_arr is not None:
                        np_face = np.asarray(face_arr)
                        if np_face.ndim == 3:
                            if np_face.max() <= 1.0:
                                np_face = (np_face * 255.0).clip(0, 255).astype("uint8")
                            else:
                                np_face = np_face.astype("uint8")
                            faces.append(Image.fromarray(np_face).convert("RGB"))
                        else:
                            # Fallback to region cropping if face_arr is not HWC
                            if isinstance(region, dict):
                                x = int(region.get("x", 0));
                                y = int(region.get("y", 0))
                                w = int(region.get("w", 0));
                                h = int(region.get("h", 0))
                                if w > 0 and h > 0:
                                    faces.append(pil.crop((x, y, x + w, y + h)).convert("RGB"))
                    elif isinstance(region, dict):
                        x = int(region.get("x", 0));
                        y = int(region.get("y", 0))
                        w = int(region.get("w", 0));
                        h = int(region.get("h", 0))
                        if w > 0 and h > 0:
                            faces.append(pil.crop((x, y, x + w, y + h)).convert("RGB"))
            except Exception:
                return []

        return faces

    def _normalize_summary(self, summary: dict[str, object]) -> dict[str, object]:
        """
        Ensure all expected keys are present in the summary dict.
        """
        base = {
            "filler_min": None,
            "selected": None,
            "responseType": None,
            "confidence": None,
            "TA_selected": None,
            "TP_selected": None,
            "TA_responseType": None,
            "TP_responseType": None,
            "TA_conf": None,
            "TP_conf": None,
        }
        base.update(summary or {})
        prefix = "TP" if self.lineup.target == "present" else "TA"
        for field, source in (("selected", "selected"), ("responseType", "responseType"), ("conf", "confidence")):
            if base[f"{prefix}_{field}"] is None:
                base[f"{prefix}_{field}"] = base[source]
        return base

    def _decide_summary(self, sim_df: pd.DataFrame) -> dict[str, object]:
        """Decide using the built lineup's explicit member roles and scenario."""
        if not self.identifier_enabled or self.identifier_obj is None:
            return {}
        return self._normalize_summary(self.identifier_obj.decide_for_pipeline(sim_df, self.lineup))

    def _no_face_records(self, frame: int) -> list[dict]:
        distance = float(self.cfg.no_face_fill)
        summary = self._normalize_summary({
            "filler_min": distance,
            "selected": None,
            "responseType": "fillerId",
            "confidence": distance,
        }) if self.identifier_enabled else {}
        return [{
            "frame": frame,
            "probe": f"frame{frame}_noface",
            "lineup_member": name,
            "distance": distance,
            "role": role,
            **summary,
        } for name, role in self.roles.items()]

    def _long_to_wide(self, df: pd.DataFrame, by=("frame", "probe"),
                      include_role_mins=True,
                      name_by_role=True,  # True: dist[member[role]]; False: dist[member]
    ) -> pd.DataFrame:
        """
        # Convert long-form df to wide-form for intuitionistic and easier analysis.
        # Columns to keep as index (frame, probe) or just (frame)
        # include_role_mins: whether to include min distances by role as separate columns
        # name_by_role: whether to include role in the distance column names
        """
        dist_wide = (
            df.pivot_table(index=list(by), columns="lineup_member", values="distance", aggfunc="min")
            .sort_index(axis=1)
        )

        # The pattern dist[member[role]] or dist[member]
        if name_by_role:
            # Get role map from df (not self.roles, in case some members missing)
            role_map = (
                df[["lineup_member", "role"]]
                .dropna()
                .drop_duplicates(subset=["lineup_member"], keep="first")
                .set_index("lineup_member")["role"]
                .to_dict()
            )
            dist_wide.columns = [f"{member}[{role_map.get(member, 'unknown')}]" for member in dist_wide.columns]
        else:
            dist_wide.columns = [f"dist[{c}]" for c in dist_wide.columns]

        # Summary columns
        candidate_summary_cols = [
            "filler_min", "TA_selected", "TP_selected", "TA_responseType", "TP_responseType", "TA_conf",
            "TP_conf" , "selected", "responseType", "confidence",
        ]
        summary_cols = [c for c in candidate_summary_cols if c in df.columns]
        summary = df.groupby(list(by), as_index=True)[summary_cols].first()

        # Optionally include min distances by role
        if include_role_mins and "role" in df.columns:
            role_min = df.groupby(list(by) + ["role"])["distance"].min().unstack("role")
            role_min = role_min.add_prefix("min_by_role[")
            role_min.columns = [c + "]" for c in role_min.columns]
            out = pd.concat([summary, role_min, dist_wide], axis=1).reset_index()
        else:
            out = pd.concat([summary, dist_wide], axis=1).reset_index()

        return out

    def run(self,
            frame_start: int = 0,
            frame_end: Optional[int] = None,
            frame_stride: int = 1):
        cap = cv.VideoCapture(self.video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            return self._run_capture(cap, frame_start, frame_end, frame_stride)
        finally:
            cap.release()

    def _run_capture(self, cap, frame_start: int, frame_end: Optional[int], frame_stride: int):
        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
        if n_frames <= 0:
            n_frames = 10 ** 12

        # Support negative indexing for frame_start and frame_end
        if frame_start < 0:
            frame_start = max(0, n_frames + frame_start)
        if frame_end is None:
            frame_end = n_frames - 1
        elif frame_end < 0:
            frame_end = max(-1, n_frames + frame_end)

        frame_start = max(0, frame_start)
        frame_end = max(-1, frame_end)
        if frame_end >= 0:
            frame_end = min(frame_end, n_frames - 1)

        if frame_start > frame_end and frame_end >= 0:
            # If start > end (and end is valid), no frames to process
            return pd.DataFrame()

        frame_stride = max(1, int(frame_stride))

        # Set the video to start frame
        # Be tolerant to FakeCap.set() with simplified signature
        try:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_start)
        except Exception:
            try:
                cap.set()
            except Exception:
                pass

        records = []
        logical_frame_idx = frame_start - 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            logical_frame_idx += 1

            if logical_frame_idx > frame_end:
                break

            if (logical_frame_idx - frame_start) % frame_stride != 0:
                continue

            if getattr(self.cfg, "use_full_frame", True):
                img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                pil_full = Image.fromarray(img_rgb)
                faces = [pil_full]
            # Only detect faces if not using full frame
            else:
                faces = self._faces_from_frame(frame)

            # Build column image for this frame
            probe_names = [f"frame{logical_frame_idx}_face{i}" for i in range(len(faces))] if faces else []
            col_il = Images([])
            if faces:
                col_il.images = {n: im for n, im in zip(probe_names, faces)}
                col_il.path_to_images = {n: f"{n}.png" for n in probe_names}  # Virtual paths
            else:
                # No faces -> {} -> return empty sim_df -> no-face logic
                col_il.images = {}
                col_il.path_to_images = {}

            # If analyzer template is missing (e.g., tests bypass __init__), fall back to no-face logic
            analyzer = getattr(self, "_analyzer_template", None)
            if analyzer is None:
                records.extend(self._no_face_records(logical_frame_idx))
                continue

            analyzer.column_images = col_il
            analyzer.show_progress = self.cfg.show_progress

            # maintain consistent methods: process / process_verify / process_georgia_pipeline
            getattr(analyzer, self.cfg.analyzer_method)()
            sim_df = analyzer.dataframe()
            sim_df = sim_df.T  # Rows: probe, Cols: lineup member

            # Check if there are any faces by seeing if sim_df is empty or all NaN
            # If no faces detected at all, or all distances are NaN, treat as no-face frame
            no_valid_dist = (sim_df.empty) or (sim_df.isna().all().all())

            if no_valid_dist:
                records.extend(self._no_face_records(logical_frame_idx))
                continue

            # Gneral case: at least one face with valid distances
            summary = {}
            if self.identifier_enabled:
                summary = self._decide_summary(sim_df)

            for probe, row in sim_df.iterrows():
                for lm_name, dist in row.items():
                    role = self.roles.get(lm_name, "filler")
                    base_row = {
                        "frame": logical_frame_idx,
                        "probe": probe,
                        "lineup_member": lm_name,
                        "distance": (None if pd.isna(dist) else float(dist)),
                        "role": role,
                    }
                    if self.identifier_enabled:
                        records.append({**base_row, **summary})
                    else:
                        records.append(base_row)

        df = pd.DataFrame.from_records(records)
        return self._long_to_wide(df, by=("frame", "probe")) if not df.empty else df

    def save(self, df: pd.DataFrame, output_csv: str = "videoLineupResults.csv"):
        """
        Save the pipeline dataframe to CSV.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty.")

        dirpath = os.path.dirname(output_csv)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)


        df.to_csv(output_csv, index=False)
        print(f"Saved pipeline results to {output_csv}")

    def savepywitness(self,
                      df: pd.DataFrame,
                      output_csv: str = "videoLineupResults_pywitness.csv",
                      targetLineup: Optional[Literal["targetPresent", "targetAbsent"]] = None,
                      lineupSize: Optional[int] = None):
        """
        Save results using the built lineup's target condition and member count.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty.")
        check_target = targetLineup or self._target_lineup
        if check_target != self._target_lineup:
            raise ValueError("targetLineup must match the built lineup; build and run TP and TA lineups separately.")
        if lineupSize is None:
            lineupSize = len(self.lineup)

        result = export_for_pywitness(
            df=df,
            output_csv=output_csv,
            targetLineup=check_target,
            lineupSize=lineupSize,
            no_face_fill=float(self.cfg.no_face_fill)
        )
        print(f"Saved pipeline results to {output_csv}")
        return result

    def plot_role_histograms(self, df: pd.DataFrame,
                             roles=("filler", "guilty_suspect", "innocent_suspect"),
                             bins=40, xlim=None, alpha=0.55,
                             figsize=(7, 4), title=None):
        """
        Draw histograms of distances by role.
        """
        from .utils.plots import plot_role_histograms as _plot
        if title is None:
            title = "Similarity histograms by role (min distance per role)"
        return _plot(df, roles=roles, bins=bins, xlim=xlim, alpha=alpha,
                     figsize=figsize, title=title, show_legend=True)
