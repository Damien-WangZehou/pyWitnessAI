from pyWitnessAI import VideoLineupPipeline, export_for_pywitness
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import types

class FakeCap:
    """
    Fabricate a cv2.videoCapture class, yielding 3 empty frames then End-Of-File.
    """
    def __init__(self, path):
        self.path = path
        self._i = 0
        self._n = 3

    def isOpened(self):
        return True

    def get(self, prop):
        if int(prop) == int(cv.CAP_PROP_FRAME_COUNT):
            return float(self._n)
        return 0.0

    def set(self):
        return True

    def read(self):
        if self._i < self._n:
            self._i += 1
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            return True, frame
        return False, None

    def release(self):
        return True


def _make_min_pipeline(no_face_fill=50.0, lineup_names=("A", "B")) -> VideoLineupPipeline:
    """
    A minimal VideoLineupPipeline instance, with no real processing.
    """
    pipe = object.__new__(VideoLineupPipeline)

    pipe.video_path = "dummy.mp4"
    pipe.lineup_loader = types.SimpleNamespace(
        lineup=[str(Path("x") / f"{n}.png") for n in lineup_names]
    )

    pipe.roles = {n: ("innocent_suspect" if i == 0 else "filler") for i, n in enumerate(lineup_names)}

    class Cfg:
        pass
    pipe.cfg = Cfg()
    pipe.cfg.no_face_fill = float(no_face_fill)
    pipe.cfg.show_progress = False
    pipe.cfg.use_full_frame = True

    # No-face
    pipe.identifier_enabled = True
    pipe.identifier_obj = None

    pipe._faces_from_frame = lambda frame: []

    return pipe


def test_save_with_basename_writes_in_cwd(tmp_path, monkeypatch):
    df = pd.DataFrame({"x": [1, 2, 3]})
    monkeypatch.chdir(tmp_path)

    pipe = _make_min_pipeline()
    pipe.save(df=df, output_csv="results.csv")

    out = tmp_path / "results.csv"
    assert out.exists(), "results.csv should be created in current working directory"
    loaded = pd.read_csv(out)
    assert not loaded.empty
    assert list(loaded.columns) == ["x"]


def test_export_for_pywitness_both_and_fallback(tmp_path):
    df_in = pd.DataFrame(
        {
            "frame": [0],
            "probe": ["p0"],
            "TA_responseType": [pd.NA],
            "TP_responseType": [pd.NA],
            "TA_conf": [pd.NA],
            "TP_conf": [pd.NA],
        }
    )
    out = tmp_path / "pw.csv"
    export_for_pywitness(
        df=df_in,
        output_csv=str(out),
        targetLineup="both",
        lineupSize=6,
        no_face_fill=33.0,
    )

    assert out.exists()
    df_out = pd.read_csv(out)
    # both -> TA & TP
    assert len(df_out) == 2
    # lineupSize -> 6-1
    assert set(df_out["lineupSize"].tolist()) == {5}
    assert set(df_out["confidence"].tolist()) == {33.0}
    assert set(df_out["targetLineup"].tolist()) == {"targetPresent", "targetAbsent"}


def test_savepywitness_writes_file(tmp_path):
    # General case: both TA & TP exist
    df_in = pd.DataFrame(
        {
            "frame": [0],
            "probe": ["p0"],
            "TA_responseType": ["fillerId"],
            "TP_responseType": ["suspectId"],
            "TA_conf": [12.0],
            "TP_conf": [34.5],
        }
    )
    pipe = _make_min_pipeline()
    out = tmp_path / "pw2.csv"
    pipe.savepywitness(
        df=df_in,
        output_csv=str(out),
        targetLineup="both",
        lineupSize=6,
    )
    assert out.exists()
    df_out = pd.read_csv(out)
    assert len(df_out) == 2
    assert set(df_out["targetLineup"].tolist()) == {"targetPresent", "targetAbsent"}


def test_run_noface_path_produces_summary(monkeypatch):
    # Mock cv2.VideoCapture to use FakeCap
    monkeypatch.setattr(cv, "VideoCapture", lambda path: FakeCap(path))

    pipe = _make_min_pipeline(no_face_fill=77.0, lineup_names=("A", "B"))
    df = pipe.run(frame_start=0, frame_end=2)

    assert not df.empty
    assert "responseType" in df.columns
    assert "confidence" in df.columns
    assert set(df["responseType"].dropna().unique().tolist()) == {"fillerId"}
    assert set(df["confidence"].dropna().round(1).unique().tolist()) == {77.0}