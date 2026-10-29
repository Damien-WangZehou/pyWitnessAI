"""Offline coverage of composition and the persisted human-feedback loop."""

import json
from pathlib import Path

from PIL import Image
import pytest

from pyWitnessAI.LineupSession import LineupSession, create_lineup
from pyWitnessAI.face_generation_backend import ImageGenerationBackend


class TinySessionBackend(ImageGenerationBackend):
    provider = "session-test"
    default_model = "tiny-session-portraits"
    supports_reference_image = True

    def __init__(self):
        self.requests = []
        self.successful_paths = []
        self.fail_at_calls = set()

    def generate(self, request):
        self.requests.append(request)
        call = len(self.requests)
        if call in self.fail_at_calls:
            request.output_path.write_bytes(b"interrupted image")
            raise RuntimeError("simulated generation failure")
        Image.new("RGB", (12, 16), (call % 256, 70, 130)).save(request.output_path)
        self.successful_paths.append(request.output_path)


@pytest.fixture
def backend():
    return TinySessionBackend()


@pytest.fixture
def faces(tmp_path):
    paths = []
    for index in range(7):
        path = tmp_path / f"existing_{index}.png"
        Image.new("RGB", (12, 16), (210 + index, 30, 80)).save(path)
        paths.append(path.resolve())
    return paths


def make_session(tmp_path, backend, **kwargs):
    options = dict(output_dir=tmp_path / "session", generator_backend=backend,
                   size=4, rows=2, seed=37)
    options.update(kwargs)
    return LineupSession("Dark hair, a small scar beneath the left eyebrow.", **options)


def displayed_pixels(round_, choice):
    return list(round_.lineup.images.values())[choice - 1].tobytes()


def saved_member(round_, choice):
    return Path(round_.directory) / f"member_{choice:02d}.png"


def assert_neutral(round_):
    assert round_.lineup.target == "unknown"
    assert round_.lineup.suspect_position is None
    assert set(round_.lineup.roles.values()) == {"candidate"}


@pytest.mark.parametrize("target, role", [("present", "guilty_suspect"), ("absent", "innocent_suspect")])
def test_compose_existing_pool_keeps_explicit_suspect_without_generation(faces, backend, target, role):
    lineup = create_lineup(suspect=faces[0], filler_pool=faces[1:], target=target,
                           order="given", generator_backend=backend)
    assert len(lineup) == 6
    assert lineup.target == target
    assert lineup.suspect_position == 1
    assert lineup.roles["member_01"] == role
    assert lineup.source_paths["member_01"] == str(faces[0])
    assert backend.requests == []


def test_description_only_creates_full_absent_lineup(tmp_path, backend):
    lineup = create_lineup(description="a person with a small scar", generator_backend=backend,
                           output_dir=tmp_path / "faces", size=4)
    assert len(lineup) == len(backend.requests) == 4
    assert lineup.target == "absent"
    assert lineup.suspect_position is None
    assert set(lineup.roles.values()) == {"filler"}
    assert all("a person with a small scar" in request.prompt for request in backend.requests)
    assert all(request.reference_image is None for request in backend.requests)


@pytest.mark.parametrize("target, expected_fillers", [("present", 5), ("absent", 6)])
def test_six_person_lineup_generates_individual_portraits(tmp_path, faces, backend, target, expected_fillers):
    description = "Dark hair; center-parted hairstyle; looked nervous"
    lineup = create_lineup(
        suspect=faces[0] if target == "present" else None,
        description=description, target=target, size=6, order="given",
        generator_backend=backend, output_dir=tmp_path / "generated",
    )

    assert len(lineup) == 6
    assert len(backend.requests) == expected_fillers
    assert all(request.total == expected_fillers for request in backend.requests)
    assert all(request.prompt.startswith("Make a photo of ") for request in backend.requests)
    assert all(" photos " not in request.prompt for request in backend.requests)
    assert all(description.lower() in request.prompt for request in backend.requests)
    assert lineup.target == target
    assert list(lineup.roles.values()).count("filler") == expected_fillers
    if target == "present":
        assert lineup.suspect_position == 1
        assert lineup.roles["member_01"] == "guilty_suspect"
        assert lineup.source_paths["member_01"] == str(faces[0])
    else:
        assert lineup.suspect_position is None
        assert str(faces[0]) not in lineup.source_paths.values()


@pytest.mark.parametrize("description", [None, "dark hair and a broad nose"])
def test_generation_reference_is_not_implicitly_a_lineup_member(tmp_path, faces, backend, description):
    lineup = create_lineup(suspect=faces[0], reference_image=faces[1], description=description,
                           generator_backend=backend, output_dir=tmp_path / "faces",
                           similarity=42, size=4, order="given")
    assert len(backend.requests) == 3
    assert lineup.source_paths["member_01"] == str(faces[0])
    assert str(faces[1]) not in lineup.source_paths.values()
    assert all(request.reference_image == faces[1] and request.similarity == 42
               for request in backend.requests)
    if description:
        assert all(description in request.prompt for request in backend.requests)


@pytest.mark.parametrize("extra", [{"description": "dark hair"}, {"reference_image": "reference"}])
def test_pool_rejects_generation_inputs_that_would_be_ignored(faces, backend, extra):
    if "reference_image" in extra:
        extra = {"reference_image": faces[0]}
    with pytest.raises((TypeError, ValueError)):
        create_lineup(filler_pool=faces, generator_backend=backend, **extra)
    assert backend.requests == []


@pytest.mark.parametrize("invalid", [
    {"target": "present"}, {"target": "both"}, {"size": 1}, {"size": True},
    {"rows": 0}, {"rows": 7}, {"order": "shuffle"}, {"seed": True},
    {"description": " "}, {"description": 123}, {"similarity": -1},
    {"similarity": float("nan")}, {"parser": "missing-parser"},
    {"generation_options": {"n": 100}}, {"generation_options": {"output_format": "gif"}},
])
def test_invalid_composition_fails_before_generation(tmp_path, backend, invalid):
    options = dict(description="brown eyes", generator_backend=backend, output_dir=tmp_path / "faces")
    options.update(invalid)
    with pytest.raises((TypeError, ValueError)):
        create_lineup(**options)
    assert backend.requests == []


@pytest.mark.parametrize("argument", ["suspect", "reference_image"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_invalid_input_images_fail_before_generation(tmp_path, backend, argument, corrupt):
    path = tmp_path / "invalid.png"
    if corrupt:
        path.write_text("this is not an image", encoding="utf-8")
    with pytest.raises((OSError, ValueError, TypeError)):
        create_lineup(description="brown eyes", generator_backend=backend,
                      output_dir=tmp_path / "faces", **{argument: path})
    assert backend.requests == []


def test_three_round_loop_preserves_selected_pixels_and_original_description(tmp_path, backend):
    session = make_session(tmp_path, backend)
    assert session.status == "new"
    assert session.current is None
    assert session.history == ()
    assert backend.requests == []

    first = session.start()
    assert first.number == 1
    assert session.status == "active"
    assert len(first.lineup) == len(backend.requests) == 4
    assert first.feedback is None
    first_pixels = displayed_pixels(first, 2)
    first_reference = saved_member(first, 2).resolve()
    second = session.respond(2, round_number=1)
    first = session.history[0]
    assert second.number == 2
    assert first.feedback["choice"] == 2
    assert first.feedback["confirmed"] is False
    assert len(backend.requests) == 7
    assert sum(image.tobytes() == first_pixels for image in second.lineup.images.values()) == 1
    assert all(request.reference_image == first_reference for request in backend.requests[4:])

    second_pixels = displayed_pixels(second, 3)
    third = session.respond(3, round_number=2)
    second = session.history[1]
    assert third.number == 3
    assert len(backend.requests) == 10
    assert sum(image.tobytes() == second_pixels for image in third.lineup.images.values()) == 1
    assert all(request.reference_image == saved_member(second, 3).resolve()
               for request in backend.requests[7:])
    assert all("dark hair, a small scar beneath the left eyebrow." in request.prompt
               for request in backend.requests)
    assert session.description == "Dark hair, a small scar beneath the left eyebrow."
    assert session.history == (first, second, third)
    for round_ in session.history:
        assert_neutral(round_)
        assert Path(round_.directory, "lineup.png").is_file()
        assert Path(round_.directory, "lineup.json").is_file()

    final_pixels = displayed_pixels(third, 1)
    confirmed_round = session.respond(1, confirmed=True, round_number=3)
    assert confirmed_round.number == third.number
    assert session.status == "confirmed"
    assert len(backend.requests) == 10
    assert session.confirmed_face == saved_member(third, 1).resolve()
    with Image.open(session.confirmed_face) as confirmed:
        assert confirmed.tobytes() == final_pixels
    assert_neutral(confirmed_round)


def test_initial_reference_guides_first_round_without_being_retained(tmp_path, faces, backend):
    session = make_session(tmp_path, backend, reference_image=faces[0], similarity=61)
    first = session.start()
    assert all(request.similarity == 61 for request in backend.requests)
    with Image.open(faces[0]) as original:
        for request in backend.requests:
            with Image.open(request.reference_image) as copied_reference:
                assert copied_reference.tobytes() == original.tobytes()
    assert str(faces[0]) not in first.lineup.source_paths.values()
    assert_neutral(first)


def test_repeated_feedback_is_idempotent_but_conflicting_feedback_is_rejected(tmp_path, backend):
    session = make_session(tmp_path, backend)
    session.start()
    second = session.respond(2, round_number=1)
    calls = len(backend.requests)
    assert session.respond(2, round_number=1) is second
    assert len(backend.requests) == calls
    with pytest.raises(ValueError):
        session.respond(3, round_number=1)
    with pytest.raises(ValueError):
        session.respond(2, confirmed=True, round_number=1)
    assert session.current is second
    assert len(backend.requests) == calls


@pytest.mark.parametrize("choice, confirmed, round_number", [
    (0, False, 1), (5, False, 1), (True, False, 1), (1.5, False, 1),
    (1, "yes", 1), (1, False, 0), (1, False, 2), (1, False, True),
])
def test_invalid_feedback_does_not_mutate_or_generate(tmp_path, backend, choice, confirmed, round_number):
    session = make_session(tmp_path, backend)
    first = session.start()
    snapshot = session.to_dict()
    with pytest.raises((TypeError, ValueError)):
        session.respond(choice, confirmed=confirmed, round_number=round_number)
    assert first.feedback is None
    assert session.to_dict() == snapshot
    assert len(backend.requests) == 4


def test_last_round_records_closest_face_and_stops_without_generation(tmp_path, backend):
    session = make_session(tmp_path, backend, max_rounds=2)
    session.start()
    second = session.respond(1, round_number=1)
    final = session.respond(3, round_number=2)
    assert final.number == second.number
    assert session.status == "limit_reached"
    assert final.feedback["choice"] == 3
    assert final.feedback["confirmed"] is False
    assert session.confirmed_face is None
    assert len(backend.requests) == 7
    assert session.respond(3, round_number=2) is final
    assert len(backend.requests) == 7


@pytest.mark.parametrize("started", [False, True])
def test_stop_closes_session_without_generating(tmp_path, backend, started):
    session = make_session(tmp_path, backend)
    if started:
        session.start()
    before = len(backend.requests)
    session.stop()
    session.stop()
    assert session.status == "stopped"
    assert session.confirmed_face is None
    with pytest.raises((ValueError, RuntimeError)):
        session.respond(1, round_number=1)
    assert len(backend.requests) == before


def test_failed_stop_save_rolls_back_status_and_can_be_retried(tmp_path, backend, monkeypatch):
    session = make_session(tmp_path, backend)
    first = session.start()
    save = session._save

    def fail_save():
        raise OSError("simulated disk failure")

    monkeypatch.setattr(session, "_save", fail_save)
    with pytest.raises(OSError, match="simulated disk failure"):
        session.stop()
    assert session.status == "active"
    assert session.current is first
    assert first.feedback is None

    monkeypatch.setattr(session, "_save", save)
    session.stop()
    assert session.status == "stopped"
    restored = LineupSession.load(tmp_path / "session", generator_backend=backend)
    assert restored.status == "stopped"
    assert restored.current.number == first.number
    assert restored.current.feedback is None
    assert len(backend.requests) == 4


def test_feedback_before_start_is_rejected(tmp_path, backend):
    session = make_session(tmp_path, backend)
    with pytest.raises((ValueError, RuntimeError)):
        session.respond(1, round_number=1)
    assert session.status == "new"
    assert backend.requests == []


def test_failed_next_round_records_feedback_and_retry_reuses_completed_outputs(tmp_path, backend):
    session = make_session(tmp_path, backend)
    first = session.start()
    backend.fail_at_calls.add(6)  # The first image of round two already succeeded.
    with pytest.raises(RuntimeError, match="simulated generation failure"):
        session.respond(2, round_number=1)
    assert session.current.number == first.number
    first = session.current
    assert session.history == (first,)
    assert first.feedback["choice"] == 2
    assert first.feedback["confirmed"] is False
    persisted = json.loads((tmp_path / "session" / "session.json").read_text(encoding="utf-8"))
    assert persisted["rounds"][0]["feedback"]["choice"] == 2
    with pytest.raises(ValueError):
        session.respond(3, round_number=1)

    second = session.respond(2, round_number=1)
    assert second.number == 2
    assert session.history == (first, second)
    assert len(backend.requests) == 8
    assert len(backend.successful_paths) == len(set(backend.successful_paths)) == 7
    assert all(request.reference_image == saved_member(first, 2).resolve()
               for request in backend.requests[4:])


def test_failed_first_round_is_retryable_without_partial_history(tmp_path, backend):
    session = make_session(tmp_path, backend)
    backend.fail_at_calls.add(2)
    with pytest.raises(RuntimeError, match="simulated generation failure"):
        session.start()
    assert session.current is None
    assert session.history == ()
    assert session.status == "new"
    first = session.start()
    assert first.number == 1
    assert len(backend.requests) == 5
    assert len(backend.successful_paths) == len(set(backend.successful_paths)) == 4


def test_load_restores_order_pixels_feedback_and_can_continue(tmp_path, backend):
    session = make_session(tmp_path, backend)
    first = session.start()
    second = session.respond(2, round_number=1)
    before = session.to_dict()
    restored_backend = TinySessionBackend()
    restored = LineupSession.load(tmp_path / "session", generator_backend=restored_backend)
    assert restored_backend.requests == []
    assert restored.to_dict() == before
    assert restored.status == "active"
    assert len(restored.history) == 2
    assert restored.current.number == 2
    for original, loaded in zip(session.history, restored.history):
        assert original.feedback == loaded.feedback
        assert original.lineup.source_paths == loaded.lineup.source_paths
        assert [image.tobytes() for image in original.lineup.images.values()] == [
            image.tobytes() for image in loaded.lineup.images.values()]
        assert_neutral(loaded)
    # An old request may be retried after a lost HTTP response and process restart.
    assert restored.respond(2, round_number=1) is restored.current
    assert restored_backend.requests == []
    result = restored.respond(3, confirmed=True, round_number=2)
    assert result.number == second.number
    assert restored.status == "confirmed"
    assert restored.confirmed_face == saved_member(second, 3).resolve()
    assert restored_backend.requests == []
    confirmed = LineupSession.load(tmp_path / "session", generator_backend=restored_backend)
    assert confirmed.status == "confirmed"
    assert confirmed.confirmed_face == restored.confirmed_face
    assert_neutral(confirmed.current)


def test_resumed_session_generates_with_saved_choice_and_description(tmp_path, backend):
    session = make_session(tmp_path, backend)
    first = session.start()
    next_backend = TinySessionBackend()
    restored = LineupSession.load(tmp_path / "session", generator_backend=next_backend)
    second = restored.respond(2, round_number=1)
    assert len(next_backend.requests) == 3
    assert all(request.reference_image == saved_member(first, 2).resolve()
               for request in next_backend.requests)
    assert all("dark hair, a small scar beneath the left eyebrow." in request.prompt
               for request in next_backend.requests)
    assert_neutral(second)


@pytest.mark.parametrize("invalid", [
    {"size": 1}, {"rows": 5}, {"max_rounds": 0}, {"max_rounds": True},
    {"similarity": 101}, {"seed": True}, {"generation_options": {"n": 5}},
])
def test_invalid_session_configuration_fails_before_generation(tmp_path, backend, invalid):
    with pytest.raises((TypeError, ValueError)):
        make_session(tmp_path, backend, **invalid)
    assert backend.requests == []


def test_session_requires_reference_capability_before_starting_first_batch(tmp_path, backend):
    backend.supports_reference_image = False
    with pytest.raises(ValueError, match="reference"):
        make_session(tmp_path, backend)
    assert backend.requests == []


def test_audit_snapshot_is_json_serializable_and_independent(tmp_path, backend):
    session = make_session(tmp_path, backend, generation_options={"quality": "low"})
    session.start()
    first = session.respond(2, confirmed=True, round_number=1)
    snapshot = session.to_dict()
    json.dumps(snapshot, allow_nan=False)
    snapshot["rounds"][0]["feedback"]["choice"] = 4
    snapshot["config"]["generation_options"]["quality"] = "high"
    assert first.feedback["choice"] == 2
    assert session.generation_options == {"quality": "low"}
    compact = session.to_dict(include_history=False)
    assert "rounds" not in compact
    assert compact["status"] == "confirmed"
