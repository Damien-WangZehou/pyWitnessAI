import base64
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pyWitnessAI.API.openai_prompt_processor import (
    DEFAULT_CONVERSATION_MODEL,
    OpenAIPromptProcessor,
)
from pyWitnessAI.prompt_processing import (
    PromptProcessingRequest,
    PromptProcessingResult,
    PromptProcessor,
)


def _request(**overrides):
    values = {
        "original_description": "Short brown hair; no beard; perhaps blue eyes.",
        "parsed_description": "short brown hair; blue eyes",
        "attributes": {"hair_colour": "brown", "eye_colour": "blue"},
        "base_prompt": "A distinctive portrait with short brown hair and blue eyes.",
        "portrait_instructions": "Front view, shoulders up, neutral expression, grey background.",
    }
    return PromptProcessingRequest(**{**values, **overrides})


def _payload(**overrides):
    return {
        "prompt": "A front view portrait with short brown hair, no beard and a grey background.",
        "summary": "Preserved the witness description and avoided a definite eye colour.",
        "constraints": ["short brown hair", "no beard", "front view", "grey background"],
        "conflicts": ["Omitted the parser's definite blue eyes because the witness was uncertain."],
        **overrides,
    }


def _response(**overrides):
    return SimpleNamespace(status="completed", output=[], output_text=json.dumps(_payload()), **overrides)


def _client(response=None):
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response or _response())))


def test_result_is_validated_and_immutable():
    result = PromptProcessingResult(**_payload(), model="conversation-model")

    assert isinstance(result.constraints, tuple)
    assert isinstance(result.conflicts, tuple)
    assert result.provider == ""
    with pytest.raises(FrozenInstanceError):
        result.prompt = "changed"


@pytest.mark.parametrize("field,value", [
    ("prompt", None), ("summary", []), ("model", 5), ("provider", None),
    ("constraints", "short hair"), ("constraints", [1]),
    ("conflicts", None), ("conflicts", ["conflict", False]),
])
def test_result_rejects_invalid_types(field, value):
    values = {**_payload(), "model": "conversation-model", field: value}
    with pytest.raises(TypeError, match=field):
        PromptProcessingResult(**values)


@pytest.mark.parametrize("field", ["prompt", "model"])
def test_result_rejects_empty_required_strings(field):
    with pytest.raises(ValueError, match=field):
        PromptProcessingResult(**{**_payload(), "model": "conversation-model", field: "  "})


def test_construction_is_lazy_and_processor_uses_the_protocol(monkeypatch):
    create_client = Mock(side_effect=AssertionError("must remain lazy"))
    monkeypatch.setattr(OpenAIPromptProcessor, "_create_client", create_client)

    processor = OpenAIPromptProcessor()

    assert processor.model == DEFAULT_CONVERSATION_MODEL
    assert isinstance(processor, PromptProcessor)
    create_client.assert_not_called()


@pytest.mark.parametrize("model", ["", "  ", None, 5])
def test_model_must_be_a_non_empty_string(model):
    with pytest.raises(ValueError, match="model"):
        OpenAIPromptProcessor(model=model)


def test_process_preserves_full_source_and_returns_auditable_output():
    description = "他没有胡须；眼睛可能是蓝色，也可能是绿色。He was running away."
    request = _request(original_description=description, filler_profile="Different jawline.", index=2, total=3)
    client = _client(_response(model="conversation-model-2026-01-01"))

    result = OpenAIPromptProcessor(model="conversation-model", client=client).process(request)

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "conversation-model"
    assert kwargs["store"] is False
    assert "tools" not in kwargs
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["strict"] is True
    assert kwargs["text"]["format"]["schema"]["additionalProperties"] is False
    assert len(kwargs["input"]) == 1
    assert kwargs["input"][0]["role"] == "user"
    content = kwargs["input"][0]["content"]
    assert len(content) == 1
    supplied = json.loads(content[0]["text"])
    assert supplied["original_description"] == description
    assert supplied["parsed_description"] == request.parsed_description
    assert supplied["attributes"] == request.attributes
    assert supplied["base_prompt"] == request.base_prompt
    assert supplied["portrait_instructions"] == request.portrait_instructions
    assert supplied["filler_profile"] == request.filler_profile
    assert supplied["portrait_index"] == 2
    assert supplied["portrait_total"] == 3
    assert supplied["reference_image_supplied"] is False
    assert supplied["requested_similarity"] is None
    assert supplied["attributes_authoritative"] is False
    assert result.prompt == _payload()["prompt"]
    assert result.model == "conversation-model-2026-01-01"
    assert result.provider == "openai"
    assert result.conflicts == tuple(_payload()["conflicts"])
    instructions = " ".join(kwargs["instructions"].split())
    assert "same-priority witness statements" in instructions
    assert "not chain-of-thought" in instructions
    assert "Do not turn observed" in instructions
    assert "including negation" in instructions


def test_explicit_experimental_controls_are_flagged_as_authoritative():
    client = _client()

    OpenAIPromptProcessor(client=client).process(_request(attributes_authoritative=True))

    kwargs = client.responses.create.call_args.kwargs
    supplied = json.loads(kwargs["input"][0]["content"][0]["text"])
    assert supplied["attributes_authoritative"] is True
    assert "explicit experimental" in kwargs["instructions"]
    assert "Preserve these controls exactly" in kwargs["instructions"]


@pytest.mark.parametrize("extension,mime_type", [
    ("png", "image/png"), ("jpg", "image/jpeg"),
    ("webp", "image/webp"), ("gif", "image/gif"),
])
def test_reference_pixels_and_similarity_are_sent_to_the_conversation_model(tmp_path: Path, extension, mime_type):
    reference = tmp_path / f"reference.{extension}"
    image_bytes = b"reference pixels"
    reference.write_bytes(image_bytes)
    client = _client()
    request = _request(
        original_description="", parsed_description="", attributes={},
        reference_image=reference, similarity=63.5,
    )

    OpenAIPromptProcessor(client=client).process(request)

    content = client.responses.create.call_args.kwargs["input"][0]["content"]
    assert json.loads(content[0]["text"])["requested_similarity"] == 63.5
    assert json.loads(content[0]["text"])["reference_image_supplied"] is True
    assert content[1] == {
        "type": "input_image",
        "image_url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}",
    }
    assert str(reference) not in content[0]["text"]


def test_combined_description_and_reference_has_no_numeric_similarity(tmp_path: Path):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference pixels")
    client = _client()

    OpenAIPromptProcessor(client=client).process(_request(reference_image=reference))

    supplied = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"][0]["text"])
    assert supplied["original_description"]
    assert supplied["requested_similarity"] is None


@pytest.mark.parametrize("overrides,error", [
    ({"original_description": None}, TypeError),
    ({"base_prompt": ""}, ValueError),
    ({"portrait_instructions": " "}, ValueError),
    ({"attributes": {"hair": False}}, TypeError),
    ({"attributes_authoritative": "yes"}, TypeError),
    ({"index": 0}, ValueError),
    ({"total": True}, ValueError),
    ({"index": 2, "total": 1}, ValueError),
    ({"similarity": -1}, ValueError),
    ({"similarity": float("nan")}, ValueError),
    ({"similarity": "75"}, ValueError),
    ({"similarity": True}, ValueError),
    ({"similarity": 75}, ValueError),
])
def test_invalid_inputs_are_rejected_before_a_provider_call(overrides, error):
    client = _client()
    with pytest.raises(error):
        OpenAIPromptProcessor(client=client).process(_request(**overrides))
    client.responses.create.assert_not_called()


def test_combined_input_rejects_reference_only_numeric_guidance(tmp_path: Path):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference pixels")
    client = _client()
    with pytest.raises(ValueError, match="reference-only"):
        OpenAIPromptProcessor(client=client).process(_request(reference_image=reference, similarity=75))
    client.responses.create.assert_not_called()


@pytest.mark.parametrize("kind,error", [("missing", FileNotFoundError), ("directory", FileNotFoundError), ("empty", ValueError), ("unsupported", ValueError)])
def test_bad_reference_inputs_fail_before_a_provider_call(tmp_path: Path, kind, error):
    reference = tmp_path / ("reference.bmp" if kind == "unsupported" else "reference.png")
    if kind == "directory":
        reference.mkdir()
    elif kind != "missing":
        reference.write_bytes(b"pixels" if kind == "unsupported" else b"")
    client = _client()

    with pytest.raises(error):
        OpenAIPromptProcessor(client=client).process(_request(reference_image=reference))

    client.responses.create.assert_not_called()


@pytest.mark.parametrize("response,match", [
    (SimpleNamespace(status="incomplete", output_text=json.dumps(_payload())), "did not complete"),
    (SimpleNamespace(status="failed", output_text=""), "did not complete"),
    (SimpleNamespace(status="completed", error={"message": "failed"}, output_text=""), "returned an error"),
    (SimpleNamespace(output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])], output_text=json.dumps(_payload())), "was refused"),
    (SimpleNamespace(output_text=""), "no output text"),
    (SimpleNamespace(output_text=None), "no output text"),
    (SimpleNamespace(output_text="```json\n{}\n```"), "invalid JSON"),
    (SimpleNamespace(output_text="[]"), "invalid result shape"),
    (SimpleNamespace(output_text=json.dumps({**_payload(), "extra": "value"})), "invalid result shape"),
    (SimpleNamespace(output_text=json.dumps({"prompt": "text"})), "invalid result shape"),
    (SimpleNamespace(output_text=json.dumps(_payload(prompt=" "))), "invalid result values"),
    (SimpleNamespace(output_text=json.dumps(_payload(constraints=[1]))), "invalid result values"),
    (SimpleNamespace(output_text=json.dumps(_payload(conflicts="none"))), "invalid result values"),
    (SimpleNamespace(output_text=json.dumps(_payload(summary=None))), "invalid result values"),
])
def test_provider_failures_and_invalid_outputs_do_not_return_a_fallback(response, match):
    client = _client(response)

    with pytest.raises(RuntimeError, match=match):
        OpenAIPromptProcessor(client=client).process(_request())

    client.responses.create.assert_called_once()


def test_api_errors_propagate():
    client = _client()
    client.responses.create.side_effect = RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        OpenAIPromptProcessor(client=client).process(_request())


def test_lazy_client_is_created_once_and_reused(monkeypatch):
    client = _client()
    create_client = Mock(return_value=client)
    monkeypatch.setattr(OpenAIPromptProcessor, "_create_client", create_client)
    processor = OpenAIPromptProcessor()

    first = processor.process(_request())
    second = processor.process(_request())

    create_client.assert_called_once_with()
    assert client.responses.create.call_count == 2
    assert first.model == second.model == DEFAULT_CONVERSATION_MODEL


def test_missing_api_key_has_a_friendly_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY.*processing"):
        OpenAIPromptProcessor().process(_request())


def test_old_sdk_has_a_friendly_error():
    client = SimpleNamespace(images=SimpleNamespace())

    with pytest.raises(RuntimeError, match="Responses API support.*Upgrade"):
        OpenAIPromptProcessor(client=client).process(_request())
