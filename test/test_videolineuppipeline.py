from pyWitnessAI import VideoLineupPipeline, export_for_pywitness
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import sys
import types

from pyWitnessAI.LineupBuilder import Lineup, LineupBuilder, LineupMember
from pyWitnessAI.LineupAIDecision import LineupAIDecision

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
    pipe.lineup = Lineup(tuple(
        LineupMember(Image.new("RGB", (10, 10)), "innocent_suspect" if i == 0 else "filler")
        for i, _ in enumerate(lineup_names)
    ), target="absent")
    pipe.roles = pipe.lineup.roles

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
    # No evidence of a paired comparison: the requested size stays unchanged.
    assert set(df_out["lineupSize"].tolist()) == {6}
    assert set(df_out["confidence"].tolist()) == {33.0}
    assert set(df_out["targetLineup"].tolist()) == {"targetPresent", "targetAbsent"}


def test_savepywitness_infers_target_and_actual_size(tmp_path):
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
    )
    assert out.exists()
    df_out = pd.read_csv(out)
    assert len(df_out) == 1
    assert df_out["targetLineup"].tolist() == ["targetAbsent"]
    assert df_out["confidence"].tolist() == [12.0]
    assert df_out["lineupSize"].tolist() == [2]

    with pytest.raises(ValueError, match="must match"):
        pipe.savepywitness(df_in, str(out), targetLineup="both")


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
    assert df["TA_responseType"].tolist() == ["fillerId"] * 3
    assert df["TA_conf"].tolist() == [77.0] * 3


@pytest.mark.parametrize("target", ["present", "absent"])
def test_pipeline_keeps_built_images_roles_and_single_scenario(tmp_path, monkeypatch, target):
    # Same-basename sources must survive as separate neutral member IDs.
    paths = []
    for index in range(3):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / "face.png"
        Image.new("RGB", (8, 10), (index * 70, 20, 80)).save(path)
        paths.append(path)
    lineup = LineupBuilder(paths[0], paths[1:], size=3, order="given", target=target).build()

    class FakeAnalyzer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def process(self):
            pass

        def dataframe(self):
            # A filler is more similar than the suspect.
            return pd.DataFrame(
                {name: [0.5, 0.1, 0.8] for name in self.column_images.images},
                index=self.row_images.images,
            )

    monkeypatch.setitem(sys.modules, "pyWitnessAI.ImagesAI", types.SimpleNamespace(ImagesAI=FakeAnalyzer))
    monkeypatch.setattr(cv, "VideoCapture", lambda path: FakeCap(path))
    pipe = VideoLineupPipeline("dummy.mp4", lineup)
    assert list(pipe._row_il.images) == list(lineup.images)
    assert len(pipe._row_il.images) == 3
    assert pipe.roles == lineup.roles

    result = pipe.run(frame_end=0)
    prefix = "TP" if target == "present" else "TA"
    assert result[f"{prefix}_responseType"].tolist() == ["fillerId"]
    assert result[f"{prefix}_selected"].tolist() == ["member_02"]
    output = tmp_path / "export.csv"
    pipe.savepywitness(result, str(output))
    exported = pd.read_csv(output)
    assert exported["lineupSize"].tolist() == [3]
    assert exported["confidence"].tolist() == [0.1]
    assert exported["targetLineup"].tolist() == ["targetPresent" if target == "present" else "targetAbsent"]


def test_pipeline_accepts_in_memory_suspect(monkeypatch):
    lineup = Lineup((
        LineupMember(Image.new("RGB", (8, 10), "red"), "guilty_suspect"),
        LineupMember(Image.new("RGB", (8, 10), "blue"), "filler"),
    ))
    monkeypatch.setitem(sys.modules, "pyWitnessAI.ImagesAI", types.SimpleNamespace(
        ImagesAI=lambda **kwargs: types.SimpleNamespace(**kwargs),
    ))
    pipe = VideoLineupPipeline("dummy.mp4", lineup)
    assert pipe._row_il.images["member_01"].getpixel((0, 0)) == (255, 0, 0)
    assert len(pipe._row_il.path_to_images) == 2


@pytest.mark.parametrize("scenario,size", [("match", 3), ("reject", 4), ("no_face", 2)])
def test_pipeline_without_suspect_decides_and_exports(tmp_path, monkeypatch, scenario, size):
    paths = []
    for index in range(size):
        path = tmp_path / f"face_{index}.png"
        Image.new("RGB", (8, 10), (index * 50, 20, 80)).save(path)
        paths.append(path)
    lineup = LineupBuilder(None, paths, size=size, order="given").build()

    class FakeAnalyzer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def process(self):
            pass

        def dataframe(self):
            distances = [0.5, 0.1] + [0.8] * (size - 2)
            return pd.DataFrame(
                {name: distances for name in self.column_images.images},
                index=self.row_images.images,
            )

    monkeypatch.setitem(sys.modules, "pyWitnessAI.ImagesAI", types.SimpleNamespace(ImagesAI=FakeAnalyzer))
    monkeypatch.setattr(cv, "VideoCapture", lambda path: FakeCap(path))
    pipe = VideoLineupPipeline("dummy.mp4", lineup, cfg={
        "use_full_frame": scenario != "no_face", "no_face_fill": 77.0,
    })
    if scenario == "no_face":
        monkeypatch.setattr(pipe, "_faces_from_frame", lambda frame: [])
    if scenario == "reject":
        pipe.identifier_obj.threshold = 0.1

    assert set(pipe.roles.values()) == {"filler"}
    assert len(pipe._row_il.images) == size
    result = pipe.run(frame_end=0)
    response = "rejectId" if scenario == "reject" else "fillerId"
    assert result["responseType"].tolist() == [response]
    assert result["TA_responseType"].tolist() == [response]
    assert result["TP_responseType"].isna().all()
    if scenario == "match":
        assert result["TA_selected"].tolist() == ["member_02"]
    else:
        assert result["TA_selected"].isna().all()

    output = tmp_path / "all_fillers.csv"
    pipe.savepywitness(result, str(output))
    exported = pd.read_csv(output)
    assert exported["responseType"].tolist() == [response]
    assert exported["targetLineup"].tolist() == ["targetAbsent"]
    assert exported["lineupSize"].tolist() == [size]
    if scenario == "reject":
        assert exported["confidence"].isna().all()
    else:
        assert exported["confidence"].tolist() == [77.0 if scenario == "no_face" else 0.1]


@pytest.mark.parametrize("wide", [False, True])
def test_export_paired_comparison_excludes_alternate_suspect(tmp_path, wide):
    result = pd.DataFrame({
        "frame": [0, 0], "probe": ["p0", "p0"],
        "TA_responseType": ["designateId"] * 2, "TP_responseType": ["fillerId"] * 2,
        "TA_conf": [0.1] * 2, "TP_conf": [0.2] * 2,
        "role": ["guilty_suspect", "innocent_suspect"],
    })
    if wide:
        result = result.drop(columns="role").head(1).assign(**{
            "member_01[guilty_suspect]": [0.3], "member_02[innocent_suspect]": [0.1],
        })
    output = tmp_path / "paired.csv"
    export_for_pywitness(result, str(output), lineupSize=7)
    exported = pd.read_csv(output)
    assert exported["lineupSize"].tolist() == [6, 6]
    assert exported["responseType"].tolist() == ["fillerId", "designateId"]


def test_explicit_suspect_identifies_role_without_excluding_fillers():
    decision = LineupAIDecision(targetLineup="targetPresent", target="suspect")
    result = decision.decide(pd.DataFrame({"suspect": [0.7], "filler": [0.2]}))
    assert result["responseType"] == "fillerId"
    assert result["selected"] == "filler"


def test_decision_uses_selected_members_only():
    lineup = Lineup((
        LineupMember(Image.new("RGB", (8, 10)), "guilty_suspect"),
        LineupMember(Image.new("RGB", (8, 10)), "filler"),
    ))
    result = LineupAIDecision(targetLineup="targetPresent").decide_for_pipeline(
        pd.DataFrame({"member_01": [0.1], "member_02": [0.2]}), lineup,
    )
    assert result["responseType"] == "suspectId"


def test_unknown_lineup_is_rejected_before_loading_video_models(monkeypatch):
    lineup = Lineup(tuple(
        LineupMember(Image.new("RGB", (8, 10)), "candidate") for _ in range(2)
    ), target="unknown")
    monkeypatch.setitem(sys.modules, "pyWitnessAI.ImagesAI", None)
    with pytest.raises(ValueError, match="known target presence"):
        VideoLineupPipeline("unused.mp4", lineup)
    with pytest.raises(ValueError, match="known target presence"):
        LineupAIDecision(targetLineup="targetPresent").decide_for_pipeline(
            pd.DataFrame({"member_01": [0.1], "member_02": [0.2]}), lineup,
        )


@pytest.mark.parametrize("method", ["decide_macro", "decide_tp_or_ta"])
def test_direct_tp_ta_decisions_reject_neutral_candidate_roles(method):
    decision = LineupAIDecision(targetLineup="targetPresent")
    with pytest.raises(ValueError, match="neutral candidates"):
        getattr(decision, method)(
            pd.DataFrame({"member_01": [0.1], "member_02": [0.2]}),
            {"member_01": "candidate", "member_02": "candidate"},
        )


@pytest.mark.parametrize("marker", ["role", "wide", "target", "targetLineup"])
def test_pywitness_export_rejects_unknown_outcomes(tmp_path, marker):
    result = pd.DataFrame({"TP_responseType": ["fillerId"], "TP_conf": [0.1]})
    if marker == "wide":
        result["member_01[candidate]"] = 0.1
    else:
        result[marker] = "candidate" if marker == "role" else "unknown"
    destination = tmp_path / "unknown.csv"
    with pytest.raises(ValueError, match="known target presence"):
        export_for_pywitness(result, str(destination), targetLineup="targetPresent")
    assert not destination.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_export_preserves_rejection_with_missing_confidence(tmp_path, legacy):
    result = pd.DataFrame({
        "TP_responseType": [None if legacy else "rejectId"],
        "TP_conf": [None],
        "responseType": ["rejectId"],
        "confidence": [None],
    })
    output = tmp_path / "rejection.csv"
    export_for_pywitness(result, str(output), targetLineup="targetPresent")
    exported = pd.read_csv(output)
    assert exported["responseType"].tolist() == ["rejectId"]
    assert exported["confidence"].isna().all()


def test_run_releases_capture_when_analysis_raises(monkeypatch):
    capture = FakeCap("dummy.mp4")
    released = []
    capture.release = lambda: released.append(True)
    monkeypatch.setattr(cv, "VideoCapture", lambda _: capture)
    pipe = _make_min_pipeline()
    pipe.cfg.analyzer_method = "process"

    def fail():
        raise RuntimeError("analysis failed")

    pipe._analyzer_template = types.SimpleNamespace(process=fail)
    with pytest.raises(RuntimeError, match="analysis failed"):
        pipe.run(frame_end=0)
    assert released == [True]
