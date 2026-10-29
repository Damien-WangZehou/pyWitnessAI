from pathlib import Path


_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "georgia_state_video1"


def test_Images_loading_singlefile(tmp_path, monkeypatch):
    import pyWitnessAI

    monkeypatch.chdir(tmp_path)
    il = pyWitnessAI.Images(str(_FIXTURES / "Video1_Mugshot2.png"))

    assert len(il.images) == 1


def test_Images_loading_directory(tmp_path, monkeypatch):
    import pyWitnessAI

    monkeypatch.chdir(tmp_path)
    il = pyWitnessAI.Images(str(_FIXTURES))

    assert len(il.images) == 8


def test_Images_loading_wildcard(tmp_path, monkeypatch):
    import pyWitnessAI

    monkeypatch.chdir(tmp_path)
    il = pyWitnessAI.Images(str(_FIXTURES / "*Mugshot*"))

    assert len(il.images) == 6

def test_Images_loading_list(tmp_path, monkeypatch):
    import pyWitnessAI

    monkeypatch.chdir(tmp_path)
    il = pyWitnessAI.Images([
        str(_FIXTURES / "*Mugshot*"),
        str(_FIXTURES / "Video1_ProbeImage.png"),
    ])

    assert len(il.images) == 7
