import cv2 as cv
from typing import List, Optional, Literal
from dataclasses import dataclass

from .ImagesAI import *
from .Lineup import Lineup
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

    Input expectations: Columns include the per-frame summary fields (TA_responseType, TP_responseType, TA_conf, TP_conf)

    Parameters
    ------
    input_csv: Path to the pipeline CSV file.
    output_csv: Where to write the new CSV. If None, will write next to input as '*_pywitness.csv'.
    targetLineup : {'targetPresent','targetAbsent','both'}
        - 'targetPresent'   -> use TP_responseType / TP_conf
        - 'targetAbsent'    -> use TA_responseType / TA_conf
        - 'both'            -> split each input record into TWO output rows (TP and TA)
    lineupSize: Number written into the `lineupSize` column for every output row.

    Output schema:
        responseType, confidence, targetLineup, lineupSize
    """
    if targetLineup is None:
        targetLineup = "both"

    # Sanity checks
    need_cols = {"TA_responseType","TP_responseType","TA_conf","TP_conf"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    has_guilty = False
    has_innocent = False

    # Check if guilty_suspect and innocent_suspect exist in the data
    cols = [str(c) for c in df.columns]
    dist_cols = [c for c in cols if c.startswith("dist[")]
    if dist_cols:
        has_guilty   = any("[guilty_suspect]"   in c for c in dist_cols)
        has_innocent = any("[innocent_suspect]" in c for c in dist_cols)

    # When data is not wide-form, check "role" column
    if ("role" in df.columns) and (not dist_cols):
        unique_roles = set(df["role"].dropna().unique().tolist())
        has_guilty   = has_guilty   or ("guilty_suspect"   in unique_roles)
        has_innocent = has_innocent or ("innocent_suspect" in unique_roles)

    effective_lineup_size_both = int(lineupSize) - 1 if (targetLineup == "both") else int(lineupSize)
    # Avoid non-positive lineup size
    if effective_lineup_size_both < 1:
        effective_lineup_size_both = 1

    # We want one output row per group (frame, probe)
    group_keys = [c for c in ["frame", "probe"] if c in df.columns]
    if group_keys:
        summary = (
            df.groupby(group_keys, as_index=False)[["TA_responseType","TP_responseType","TA_conf","TP_conf"]]
              .first()
        )
    else:
        # If neither frame nor probe exists, just take the first row
        summary = df[["TA_responseType","TP_responseType","TA_conf","TP_conf"]].drop_duplicates().head(1)
        summary["__dummy__"] = 0
        group_keys = ["__dummy__"]

    def _fallback(row):
        base_resp = row.get("responseType")
        base_conf = row.get("confidence")
        resp = (base_resp if pd.notna(base_resp) else "fillerId")
        if pd.notna(base_conf):
            conf = float(base_conf)
        else:
            conf = float(no_face_fill if no_face_fill is not None else 50.0)
        return resp, conf

    out_rows = []

    def emit_row(resp: str, conf, tl: str, lineup_size=None):
        out_rows.append({
            "responseType": resp if pd.notna(resp) else None,
            "confidence":   float(conf) if pd.notna(conf) else None,
            "targetLineup": tl,
            "lineupSize":   int(lineup_size if lineup_size is not None else lineupSize),
        })

    # For each summary row, emit one or two output rows depending on targetLineup
    for _, row in summary.iterrows():
        ta_resp, ta_conf = row.get("TA_responseType"), row.get("TA_conf")
        tp_resp, tp_conf = row.get("TP_responseType"), row.get("TP_conf")

        # Check if both TA and TP responses are missing
        is_no_face_like = (pd.isna(ta_resp) and pd.isna(tp_resp) and pd.isna(ta_conf) and pd.isna(tp_conf))

        if targetLineup == "targetPresent":
            resp, conf = (tp_resp, tp_conf)
            if pd.isna(resp) or pd.isna(conf):
                if is_no_face_like:
                    resp, conf = _fallback(row)
            emit_row(resp, conf, "targetPresent")

        elif targetLineup == "targetAbsent":
            resp, conf = (ta_resp, ta_conf)
            if pd.isna(resp) or pd.isna(conf):
                if is_no_face_like:
                    resp, conf = _fallback(row)
            emit_row(resp, conf, "targetAbsent")

        elif targetLineup == "both":
            resp_tp, conf_tp = (tp_resp, tp_conf)
            if pd.isna(resp_tp) or pd.isna(conf_tp):
                if is_no_face_like:
                    resp_tp, conf_tp = _fallback(row)
            emit_row(resp_tp, conf_tp, "targetPresent", effective_lineup_size_both)

            resp_ta, conf_ta = (ta_resp, ta_conf)
            if pd.isna(resp_ta) or pd.isna(conf_ta):
                if is_no_face_like:
                    resp_ta, conf_ta = _fallback(row)
            emit_row(resp_ta, conf_ta, "targetAbsent", effective_lineup_size_both)

        else:
            raise ValueError("targetLineup must be 'targetPresent', 'targetAbsent', or 'both'.")

    out = pd.DataFrame(out_rows, columns=["responseType","confidence","targetLineup","lineupSize"])

    if output_csv is None:
        output_csv = "videoLineupResults_pywitness.csv"

    out.to_csv(output_csv, index=False)
    return output_csv


class VideoLineupPipeline:
    """
    For each frame:
      - Row data: frame + lineup member distances + TA/TP summary (if identifier is set)
      - Distance is calculated by ImageAnalyzer between detected faces and lineup members
      - Summary is calculated by LineupIdentifier from the distance matrix
    """

    def __init__(self,
                 video_path: str,
                 lineup: Lineup,
                 cfg: PipelineConfig | dict = PipelineConfig()):
        self.video_path = video_path
        self.lineup_loader = lineup_loader

        if isinstance(cfg, dict):
            self.cfg = PipelineConfig(**cfg)
        elif isinstance(cfg, PipelineConfig):
            self.cfg = cfg
        else:
            raise TypeError("cfg must be a PipelineConfig or dict")

        # Identifier used to be a choice, now it is always enabled
        self.identifier_obj = LineupAIDecision()
        self.identifier_enabled = True
        # identifier: Optional[LineupIdentifier] = None
        # # Normalize identifier switch/object
        # if isinstance(identifier, LineupIdentifier):
        #     self.identifier_obj: Optional[LineupIdentifier] = identifier
        #     self.identifier_enabled: bool = True
        # elif identifier is True:
        #     self.identifier_obj = LineupIdentifier()  # default threshold/targetLineup/target
        #     self.identifier_enabled = True
        # else:
        #     self.identifier_obj = None
        #     self.identifier_enabled = False

        # Initialize the lineup rows from LineupLoader
        self._row_il = Images([*self.lineup_loader.lineup])  # paths -> PIL
        # Roles map: image base name -> role
        self.roles = LineupAIDecision._role_map_from_lineuploader(self.lineup_loader)

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

    def _has_face(self, pil_img: Image.Image) -> bool:
        """
        Face presence check that matches the configured analyzer_method:
        - process_georgia_pipeline -> MTCNN (facenet-pytorch)
        - process/process_verify   -> DeepFace.represent(detector_backend=cfg.backend)
        - fallback                 -> DeepFace.extract_faces
        Returns True if at least one face is detected, else False.
        """
        method = getattr(self.cfg, "analyzer_method", "process")
        backend = getattr(self.cfg, "backend", "mtcnn")
        show = bool(getattr(self.cfg, "show_progress", False))

        # Helper: silence stdout/stderr when show_progress is False
        def _maybe_silent(callable_, *args, **kwargs):
            if show:
                return callable_(*args, **kwargs)
            with io.StringIO() as _buf1, io.StringIO() as _buf2, redirect_stdout(_buf1), redirect_stderr(_buf2):
                return callable_(*args, **kwargs)

        # 1. Georgia pipeline: use MTCNN detection
        # Ensure analyzer's mtcnn exists and matches current show_progress
        if method == "process_georgia_pipeline":
            try:
                # Recreate MTCNN if show_progress changed
                boxes, probs = self._analyzer_template.mtcnn.detect(pil_img)
                if boxes is None or probs is None:
                    return False
                return True
                w, h = pil_img.size
                img_area = float(w * h)
                for (x1, y1, x2, y2), p in zip(boxes, probs):
                    if p is None:
                        continue
                    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    if (p >= 0.90) and (0.01 * img_area <= area <= 0.80 * img_area):
                        return True
                return False
            except Exception:
                return False

        # 2 DeepFace.represent path for process / process_verify
        if method in ("process", "process_verify"):
            try:
                arr = np.array(pil_img)
                # Use represent to keep consistent with analyzer settings
                reps = _maybe_silent(
                    DeepFace.represent,
                    arr,
                    model_name=self._analyzer_template.model,
                    detector_backend=backend,
                    enforce_detection=self._analyzer_template.enforce_detection,
                    align=self._analyzer_template.align,
                    normalization=self._analyzer_template.normalization
                )
                if not reps:
                    return False

                return True
                # If there is facial_area/region info, use it to filter out too small/large faces
                w, h = pil_img.size
                img_area = float(w * h)
                min_area = 0.01 * img_area
                max_area = 0.80 * img_area
                ok = False
                for r in reps:
                    fa = r.get("facial_area") or r.get("region") or {}
                    if isinstance(fa, dict):
                        rw = float(fa.get("w", 0.0))
                        rh = float(fa.get("h", 0.0))
                        area = rw * rh
                        if min_area <= area <= max_area:
                            ok = True
                            break
                    else:
                        # If no region info, just accept it
                        ok = True
                        break
                return ok
            except Exception:
                return False

        # 3 Fallback: DeepFace.extract_faces
        try:
            arr = np.array(pil_img)
            faces = _maybe_silent(
                DeepFace.extract_faces,
                img_path=arr,
                detector_backend=backend,
                enforce_detection=False
            ) or []
            if not faces:
                return False
            return True
            w, h = pil_img.size
            img_area = float(w * h)
            min_area = 0.01 * img_area
            max_area = 0.80 * img_area
            for f in faces:
                conf = float(f.get("confidence", 0.0))
                region = f.get("facial_area") or f.get("region") or {}
                if isinstance(region, dict):
                    area = float(region.get("w", 0.0)) * float(region.get("h", 0.0))
                else:
                    area = 0.0
                if (conf >= 0.90) and (min_area <= area <= max_area):
                    return True
            return False
        except Exception:
            return False

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

    def _normalize_summary(self, summary: Dict[str, object]) -> Dict[str, object]:
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
        return base

    def _decide_summary(self, sim_df: pd.DataFrame) -> Dict[str, object]:
        """
        Decide the summary for the frame based on the identifier settings and roles.
        Choices: decide_macro, decide_tp_or_ta, decide (simple)
        """
        if not self.identifier_enabled or self.identifier_obj is None:
            return {}

        roles = self.roles
        has_guilty = any(v == "guilty_suspect" for v in roles.values())
        has_innocent = any(v == "innocent_suspect" for v in roles.values())

        # Case1: no targetLineup, both guilty and innocent exist -> macro
        if self.identifier_obj.targetLineup is None and has_guilty and has_innocent:
            return self._normalize_summary(self.identifier_obj.decide_macro(sim_df, roles))

        # Case2: targetLineup is set to TP/TA -> TP/TA strategy
        if self.identifier_obj.targetLineup in ("targetPresent", "targetAbsent"):
            return self._normalize_summary(self.identifier_obj.decide_tp_or_ta(sim_df, roles))

        # Case3: neither guilty nor innocent exists, or only one exists but targetLineup is not set
        return self._normalize_summary(self.identifier_obj.decide_for_pipeline(sim_df, self.lineup_loader))

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
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

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
                noface_probe = f"frame{logical_frame_idx}_noface"
                for lm_path in self.lineup_loader.lineup:
                    lm_name = lm_path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
                    role = self.roles.get(lm_name, "filler")

                    base_row = {
                        "frame": logical_frame_idx,
                        "probe": noface_probe,
                        "lineup_member": lm_name,
                        "distance": float(self.cfg.no_face_fill),
                        "role": role,
                    }

                    if self.identifier_enabled:
                        summary = self._normalize_summary({
                            "filler_min": float(self.cfg.no_face_fill),
                            "selected": None,
                            "responseType": "fillerId",
                            "confidence": float(self.cfg.no_face_fill),
                        })
                        records.append({**base_row, **summary})
                    else:
                        records.append(base_row)
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
                noface_probe = f"frame{logical_frame_idx}_noface"
                for lm_path in self.lineup_loader.lineup:
                    lm_name = lm_path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
                    role = self.roles.get(lm_name, "filler")

                    base_row = {
                        "frame": logical_frame_idx,
                        "probe": noface_probe,
                        "lineup_member": lm_name,
                        "distance": float(self.cfg.no_face_fill),
                        "role": role,
                    }

                    if self.identifier_enabled:
                        summary = self._normalize_summary({
                            "filler_min": float(self.cfg.no_face_fill),
                            "selected": None,
                            "responseType": "fillerId",
                            "confidence": float(self.cfg.no_face_fill),
                        })
                        records.append({**base_row, **summary})
                    else:
                        records.append(base_row)
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

        cap.release()
        df = pd.DataFrame.from_records(records)
        wide: bool = True
        if wide:
            df = self._long_to_wide(df, by=("frame", "probe"))
        return df

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
                      targetLineup: Literal["targetPresent", "targetAbsent", "both"] = "both",
                      lineupSize: Optional[int] = None):
        """
        Save a pyWitness-compatible CSV from the pipeline dataframe.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty.")
        if lineupSize is None:
            lineupSize = len(self.lineup_loader.lineup)

        dirpath = os.path.dirname(output_csv)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        check_target = (
            self.identifier_obj.targetLineup
            if (self.identifier_obj and self.identifier_obj.targetLineup in ("targetPresent", "targetAbsent"))
            else (targetLineup or "both")
        )

        export_for_pywitness(
            df=df,
            output_csv=output_csv,
            targetLineup=check_target,
            lineupSize=lineupSize,
            no_face_fill=float(self.cfg.no_face_fill)
        )
        print(f"Saved pipeline results to {output_csv}")

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
