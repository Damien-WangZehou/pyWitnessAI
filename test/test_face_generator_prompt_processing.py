"""Optional conversational prompt processing without live model requests."""

from __future__ import annotations

import base64
import builtins
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import pytest

from pyWitnessAI.Description import ParsedDescription
from pyWitnessAI.API.openai import OpenAIImageBackend
from pyWitnessAI.API.openai_prompt_processor import OpenAIPromptProcessor
from pyWitnessAI.FaceAttributeSchema import _PORTRAIT_INSTRUCTIONS
from pyWitnessAI.FaceGenerator import FaceGenerator
from pyWitnessAI.face_generation_backend import ImageGenerationBackend
from pyWitnessAI.prompt_processing import (
    PromptProcessor,
    PromptProcessingRequest,
    PromptProcessingResult,
)


class TinyPromptImageBackend(ImageGenerationBackend):
    provider = "prompt-image-test"
    default_model = "tiny-image-v1"
    supports_reference_image = True

    def __init__(self, *, fail_once_at=None):
        self.requests = []
        self.fail_once_at = fail_once_at

    def generate(self, request):
        self.requests.append(request)
        if request.index == self.fail_once_at:
            self.fail_once_at = None
            request.output_path.write_bytes(b"incomplete image")
            raise RuntimeError("simulated image failure")
        Image.new("RGB", (8, 8), (request.index, 80, 120)).save(request.output_path)


class FakePromptProcessor(PromptProcessor):
    provider = "fake-conversation"
    model = "fake-text-v1"

    def __init__(self, *, fail_once_at=None, prompt=None):
        self.requests = []
        self.fail_once_at = fail_once_at
        self.prompt = prompt

    def process(self, request: PromptProcessingRequest) -> PromptProcessingResult:
        self.requests.append(request)
        if request.index == self.fail_once_at:
            self.fail_once_at = None
            raise RuntimeError("simulated conversation failure")
        return PromptProcessingResult(
            prompt=self.prompt if self.prompt is not None else (
                f"Rewritten portrait {request.index}. {request.base_prompt}"
            ),
            summary=f"Prepared portrait {request.index}",
            constraints=("Keep the witness's eye description", "Use one portrait"),
            conflicts=("Supplementary eye colour yields to witness description",),
            model=self.model,
            provider=self.provider,
        )


class OfflineOpenAISDK:
    """Exercise both real adapters using only in-memory SDK-shaped replies."""

    def __init__(self):
        self.events = []
        self.text_calls = []
        self.image_calls = []
        self.responses = SimpleNamespace(create=self.create_response)
        self.images = SimpleNamespace(generate=self.generate_image, edit=self.edit_image)

    def create_response(self, **arguments):
        self.events.append("text")
        self.text_calls.append(arguments)
        return SimpleNamespace(
            status="completed",
            model="text-test-snapshot",
            output_text=json.dumps({
                "prompt": "Photograph one person with blue eyes.",
                "summary": "Retained eye colour and one-person composition.",
                "constraints": ["blue eyes", "one person"],
                "conflicts": [],
            }),
        )

    def generate_image(self, **arguments):
        self.events.append("image_generate")
        self.image_calls.append(arguments)
        return self._image_reply()

    def edit_image(self, **arguments):
        self.events.append("image_edit")
        image_file = arguments.pop("image")
        self.image_calls.append({
            **arguments,
            "reference_bytes": image_file.read(),
            "reference_path": Path(image_file.name).resolve(),
        })
        return self._image_reply()

    @staticmethod
    def _image_reply():
        buffer = BytesIO()
        Image.new("RGB", (8, 8), (10, 80, 120)).save(buffer, format="PNG")
        return SimpleNamespace(data=[SimpleNamespace(
            b64_json=base64.b64encode(buffer.getvalue()).decode("ascii"),
        )])


@pytest.fixture
def processor_reference_image(tmp_path: Path) -> Path:
    path = tmp_path / "reference.png"
    Image.new("RGB", (10, 10), (90, 120, 160)).save(path)
    return path


def test_disabled_processing_keeps_direct_generation_and_requested_labels(tmp_path):
    backend = TinyPromptImageBackend()
    generator = FaceGenerator(backend, tmp_path / "faces", description="a man with blue eyes")
    original_prompts = generator.generation_prompts()

    results = generator.generate()

    assert [request.prompt for request in backend.requests] == original_prompts
    assert results[0].prompt == original_prompts[0]
    assert results[0].prompt_processing is None
    assert generator.prompt_processing_results == []
    assert generator.to_manifest().iloc[0]["attribute_label_source"] == "requested"


def test_construction_local_preview_and_dry_run_do_not_process_or_write(tmp_path):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    output_dir = tmp_path / "faces"
    generator = FaceGenerator(
        backend, output_dir, 2, description="blue eyes", prompt_processor=processor,
    )

    prompts = generator.generation_prompts()
    assert generator.generate(dry_run=True) == []

    assert len(prompts) == 2
    assert processor.requests == backend.requests == []
    assert generator.prompt_processing_results == []
    assert generator.results == []
    assert not output_dir.exists()


def test_explicit_processed_preview_is_cached_and_used_by_generation(tmp_path):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    output_dir = tmp_path / "faces"
    generator = FaceGenerator(
        backend, output_dir, 2, description="blue eyes", prompt_processor=processor,
    )
    base_prompts = generator.generation_prompts()

    processed = generator.generation_prompts(process=True)

    assert [request.index for request in processor.requests] == [1, 2]
    assert [request.base_prompt for request in processor.requests] == base_prompts
    assert len(generator.prompt_processing_results) == 2
    assert generator.generation_prompts() == processed
    assert generator.generation_prompts(process=True) == processed
    assert backend.requests == []
    assert not output_dir.exists()

    results = generator.generate()
    generator.generate()

    assert len(processor.requests) == len(backend.requests) == 2
    assert [request.prompt for request in backend.requests] == processed
    assert [result.prompt for result in results] == processed
    assert [result.base_prompt for result in results] == base_prompts
    assert [result.prompt_processing.prompt for result in results] == processed
    assert len(pd.read_csv(output_dir / "manifest.csv")) == 2


def test_each_portrait_receives_full_description_hints_profile_and_reference(
    tmp_path, processor_reference_image,
):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    description = "A man with blue eyes, a star-shaped birthmark and perhaps long hair."
    profiles = ["brown eyes and a broad nose", "brown eyes and a narrow nose"]
    generator = FaceGenerator(
        backend, tmp_path / "faces", 2, description=description,
        filler_profiles=profiles, reference_image=processor_reference_image,
        similarity=23, prompt_processor=processor,
    )
    base_prompts = generator.generation_prompts()

    generator.generate()

    assert len(processor.requests) == 2
    for index, request in enumerate(processor.requests, 1):
        assert request.original_description == description
        assert "blue eyes" in str(request.parsed_description)
        assert request.attributes == generator.parsed_description.schema_attributes
        assert request.attributes_authoritative is False
        assert request.base_prompt == base_prompts[index - 1]
        assert request.portrait_instructions == _PORTRAIT_INSTRUCTIONS
        assert request.filler_profile == profiles[index - 1]
        assert profiles[index - 1] in request.base_prompt
        assert request.reference_image == processor_reference_image.resolve()
        # Combined-input generation has no numeric resemblance instruction.
        assert request.similarity is None
        assert request.index == index
        assert request.total == 2
    assert all(request.reference_image == processor_reference_image.resolve()
               for request in backend.requests)


def test_reference_only_processing_receives_numeric_resemblance(
    tmp_path, processor_reference_image,
):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    generator = FaceGenerator(
        backend, tmp_path / "faces", reference_image=processor_reference_image,
        similarity=23, prompt_processor=processor,
    )

    generator.generate()

    request = processor.requests[0]
    assert request.original_description == ""
    assert request.reference_image == processor_reference_image.resolve()
    assert request.similarity == 23
    assert "23/100" in request.base_prompt


def test_processing_can_use_original_description_unrecognised_by_schema(tmp_path):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    description = "a star-shaped birthmark"
    generator = FaceGenerator(
        backend, tmp_path / "faces", description=description, prompt_processor=processor,
    )

    results = generator.generate()

    assert processor.requests[0].original_description == description
    assert processor.requests[0].attributes == {}
    assert results[0].attributes == {}
    assert len(backend.requests) == 1


def test_preparsed_experimental_attributes_are_authoritative_for_processing(tmp_path):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    label = "opaque experimental eye label"
    parsed = ParsedDescription.from_attributes({"eyes": label}, original_text="source wording")
    generator = FaceGenerator(
        backend, tmp_path / "faces", description=parsed, prompt_processor=processor,
    )

    result = generator.generate()[0]
    request = processor.requests[0]
    row = generator.to_manifest().iloc[0]

    assert generator.parsed_description is parsed
    assert request.original_description == "source wording"
    assert request.attributes == {"eyes": label}
    assert request.attributes_authoritative is True
    assert label in request.base_prompt
    assert result.attributes == {"eyes": label}
    assert row["eyes"] == label
    assert row["parser"] == "preparsed"
    assert row["attribute_label_source"] == "requested"


def test_random_sampled_attributes_remain_authoritative(tmp_path):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    generator = FaceGenerator(
        backend, tmp_path / "faces", 2, seed=42, prompt_processor=processor,
    )

    results = generator.generate()

    for request, result in zip(processor.requests, results):
        assert request.attributes_authoritative is True
        assert request.attributes == result.attributes
        assert result.attributes
        assert all(value in request.base_prompt for value in result.attributes.values())
    assert generator.to_manifest().attribute_label_source.eq("requested").all()


def test_fixed_portrait_instructions_are_enforced_in_cached_and_recorded_prompt(tmp_path):
    backend = TinyPromptImageBackend()
    processor = FakePromptProcessor(prompt="Photograph the described person.")
    generator = FaceGenerator(
        backend, tmp_path / "faces", description="blue eyes", prompt_processor=processor,
    )

    preview = generator.generation_prompts(process=True)[0]
    result = generator.generate()[0]

    assert "Photograph the described person." in preview
    assert _PORTRAIT_INSTRUCTIONS in preview
    assert preview == generator.prompt_processing_results[0].prompt
    assert preview == result.prompt_processing.prompt == result.prompt
    assert preview == backend.requests[0].prompt
    assert preview == generator.to_manifest().iloc[0]["prompt"]
    assert len(processor.requests) == 1


@pytest.mark.parametrize("invalid_reference", ["missing", "unsupported"])
def test_reference_validation_precedes_conversation_requests(
    tmp_path, processor_reference_image, invalid_reference,
):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    if invalid_reference == "missing":
        reference = tmp_path / "missing.png"
        expected_error = FileNotFoundError
    else:
        reference = processor_reference_image
        backend.supports_reference_image = False
        expected_error = ValueError
    output_dir = tmp_path / "faces"

    with pytest.raises(expected_error):
        generator = FaceGenerator(
            backend, output_dir, description="blue eyes", reference_image=reference,
            prompt_processor=processor,
        )
        generator.generate()

    assert processor.requests == backend.requests == []
    assert not output_dir.exists()


def test_conversation_model_and_custom_processor_cannot_both_be_selected(tmp_path):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()

    with pytest.raises(ValueError):
        FaceGenerator(
            backend, tmp_path / "faces", description="blue eyes",
            conversation_model="fake-text-v1", prompt_processor=processor,
        )

    assert processor.requests == backend.requests == []
    assert not (tmp_path / "faces").exists()


def test_conversation_failure_blocks_images_and_retry_reuses_successful_processing(tmp_path):
    backend = TinyPromptImageBackend()
    processor = FakePromptProcessor(fail_once_at=2)
    generator = FaceGenerator(
        backend, tmp_path / "faces", 3, description="blue eyes", prompt_processor=processor,
    )

    with pytest.raises(RuntimeError, match="simulated conversation failure"):
        generator.generate()

    assert backend.requests == []
    assert generator.results == []
    assert not (generator.output_dir / "manifest.csv").exists()

    results = generator.generate()

    assert [request.index for request in processor.requests] == [1, 2, 2, 3]
    assert [request.index for request in backend.requests] == [1, 2, 3]
    assert len(results) == len(generator.prompt_processing_results) == 3


def test_image_failure_and_retry_keep_processed_prompts_and_partial_manifest(tmp_path):
    backend, processor = TinyPromptImageBackend(fail_once_at=2), FakePromptProcessor()
    generator = FaceGenerator(
        backend, tmp_path / "faces", 3, description="blue eyes", prompt_processor=processor,
    )

    with pytest.raises(RuntimeError, match="simulated image failure"):
        generator.generate()

    assert len(processor.requests) == 3
    assert [request.index for request in backend.requests] == [1, 2]
    assert len(generator.results) == 1
    assert len(pd.read_csv(generator.output_dir / "manifest.csv")) == 1
    assert not backend.requests[-1].output_path.exists()
    prompts = generator.generation_prompts()

    results = generator.generate()
    rows = pd.read_csv(generator.output_dir / "manifest.csv")

    assert len(processor.requests) == 3
    assert [request.index for request in backend.requests] == [1, 2, 2, 3]
    assert [result.prompt for result in results] == prompts
    assert len(rows) == 3
    assert rows.image_id.is_unique
    for result in results:
        row = rows.loc[rows.image_id == result.image_id].iloc[0]
        assert row["prompt"] == result.prompt
        assert row["base_prompt"] == result.base_prompt


def test_manifest_records_actual_prompt_audit_and_keeps_attribute_provenance(tmp_path):
    backend, processor = TinyPromptImageBackend(), FakePromptProcessor()
    generator = FaceGenerator(
        backend, tmp_path / "faces", description="a man with blue eyes",
        prompt_processor=processor,
    )
    base_prompt = generator.generation_prompts()[0]

    result = generator.generate()[0]
    row = pd.read_csv(generator.output_dir / "manifest.csv").iloc[0]
    processing = result.prompt_processing

    assert row["prompt"] == result.prompt == backend.requests[0].prompt
    assert row["base_prompt"] == result.base_prompt == base_prompt
    assert row["prompt_processing_provider"] == processing.provider == processor.provider
    assert row["prompt_processing_model"] == processing.model == processor.model
    assert row["prompt_processing_summary"] == processing.summary
    assert json.loads(row["prompt_processing_constraints"]) == list(processing.constraints)
    assert json.loads(row["prompt_processing_conflicts"]) == list(processing.conflicts)
    assert row["model"] == backend.default_model
    assert row["provider"] == backend.provider
    assert row["attribute_label_source"] == "parsed_attribute_hints"
    assert not row["attributes_verified"]
    assert row["eyes"] == result.attributes["eyes"] == "blue eyes"
    assert row["original_description"] == "a man with blue eyes"


@pytest.mark.parametrize("use_reference", [False, True])
def test_real_openai_adapters_form_text_then_image_pipeline_with_mock_sdk(
    tmp_path, processor_reference_image, use_reference,
):
    sdk = OfflineOpenAISDK()
    processor = OpenAIPromptProcessor(client=sdk, model="text-test")
    backend = OpenAIImageBackend(client=sdk)
    reference = processor_reference_image if use_reference else None
    generator = FaceGenerator(
        backend, tmp_path / "faces", description="a man with blue eyes",
        reference_image=reference, model="image-test", prompt_processor=processor,
    )

    result = generator.generate()[0]

    assert sdk.events == ["text", "image_edit" if use_reference else "image_generate"]
    text_call, image_call = sdk.text_calls[0], sdk.image_calls[0]
    assert text_call["model"] == "text-test"
    assert image_call["model"] == result.model == "image-test"
    assert image_call["prompt"] == result.prompt == result.prompt_processing.prompt
    assert "Photograph one person with blue eyes." in result.prompt
    assert _PORTRAIT_INSTRUCTIONS in result.prompt
    assert result.prompt_processing.model == "text-test-snapshot"
    assert result.prompt_processing.constraints == ("blue eyes", "one person")
    assert result.prompt_processing.conflicts == ()
    content = text_call["input"][0]["content"]
    payload = json.loads(content[0]["text"])
    assert payload["original_description"] == "a man with blue eyes"
    assert payload["reference_image_supplied"] is use_reference
    if use_reference:
        image_input = next(item for item in content if item["type"] == "input_image")
        assert base64.b64decode(image_input["image_url"].split(",", 1)[1]) == reference.read_bytes()
        assert image_call["reference_bytes"] == reference.read_bytes()
        assert image_call["reference_path"] == reference.resolve()
    else:
        assert all(item["type"] != "input_image" for item in content)
        assert "reference_bytes" not in image_call
    with Image.open(result.path) as generated:
        assert generated.size == (8, 8)


def test_conversation_model_convenience_is_lazy_and_uses_real_processor(
    tmp_path, monkeypatch,
):
    sdk = OfflineOpenAISDK()
    created_clients = []

    def create_client():
        created_clients.append(sdk)
        return sdk

    original_import = builtins.__import__

    def no_live_sdk(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise AssertionError("Offline prompt processing must not import the OpenAI SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_live_sdk)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(OpenAIPromptProcessor, "_create_client", staticmethod(create_client))
    backend = TinyPromptImageBackend()
    generator = FaceGenerator(
        backend, tmp_path / "faces", description="blue eyes", conversation_model="text-test",
    )

    assert isinstance(generator.prompt_processor, OpenAIPromptProcessor)
    assert generator.prompt_processor.client is None
    generator.generation_prompts()
    generator.generate(dry_run=True)
    assert created_clients == sdk.text_calls == backend.requests == []
    assert not generator.output_dir.exists()

    result = generator.generate()[0]
    generator.generate()

    assert created_clients == [sdk]
    assert len(sdk.text_calls) == len(backend.requests) == 1
    assert sdk.text_calls[0]["model"] == "text-test"
    assert result.prompt_processing.provider == "openai"


def test_direct_mode_never_creates_conversation_client(tmp_path, monkeypatch):
    def unexpected_text_client():
        pytest.fail("Direct image generation must not create a conversation client")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        OpenAIPromptProcessor, "_create_client", staticmethod(unexpected_text_client),
    )
    backend = TinyPromptImageBackend()
    generator = FaceGenerator(backend, tmp_path / "faces", description="blue eyes")

    generator.generation_prompts()
    generator.generate(dry_run=True)
    result = generator.generate()[0]

    assert generator.prompt_processor is None
    assert result.prompt_processing is None
    assert len(backend.requests) == 1


def test_lineup_session_preserves_conversation_model_when_saved_and_resumed(tmp_path, monkeypatch):
    from pyWitnessAI.LineupSession import LineupSession

    sdk = OfflineOpenAISDK()
    monkeypatch.setattr(OpenAIPromptProcessor, "_create_client", staticmethod(lambda: sdk))
    backend = TinyPromptImageBackend()
    session = LineupSession(
        "blue eyes", output_dir=tmp_path / "session", size=2, rows=1,
        generator_backend=backend, generation_options={"conversation_model": "text-test"},
    )
    assert sdk.text_calls == backend.requests == []

    session.start()
    call_count = len(sdk.text_calls)
    assert call_count == 2
    resumed = LineupSession.load(session.output_dir, generator_backend=backend)

    assert resumed.generation_options == {"conversation_model": "text-test"}
    assert resumed.current.number == 1
    assert len(sdk.text_calls) == call_count
