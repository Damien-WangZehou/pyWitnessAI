"""Public imports must work from a clean interpreter without optional models."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
_BLOCK_OPTIONAL = """
import importlib.abc
import sys

optional_roots = {
    'cv', 'cv2', 'deepface', 'facenet_pytorch', 'tensorflow', 'keras',
    'tf_keras', 'mtcnn', 'retinaface', 'torch', 'sentence_transformers',
    'transformers', 'tokenizers', 'huggingface_hub', 'matplotlib',
    'IPython', 'openai', 'fastapi', 'starlette', 'uvicorn', 'pydantic', 'onnxruntime',
}

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in optional_roots:
            raise ModuleNotFoundError('Optional dependency deliberately absent', name=fullname)

sys.meta_path.insert(0, BlockOptional())
"""


def _run_fresh(source: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_SOURCE_ROOT), environment.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_plain_import_and_discovery_do_not_import_implementation_modules():
    _run_fresh(_BLOCK_OPTIONAL + """
import pyWitnessAI

assert isinstance(pyWitnessAI.__version__, str) and pyWitnessAI.__version__
assert 'pyWitnessAI.FaceSearch' not in sys.modules
assert 'pyWitnessAI.Images' not in sys.modules
assert 'pyWitnessAI.API.cli' not in sys.modules
names = dir(pyWitnessAI)
assert 'FrameAnalyzerDeepface' in names
assert 'FrameProcessorCropper' in names
assert 'FillerSelector' in names
assert 'pyWitnessAI.VideoAI' not in sys.modules
assert not optional_roots.intersection(name.split('.')[0] for name in sys.modules)
""")


def test_star_import_has_a_stable_core_surface_without_optional_dependencies():
    _run_fresh(_BLOCK_OPTIONAL + """
import pyWitnessAI

before = tuple(pyWitnessAI.__all__)
namespace = {}
exec('from pyWitnessAI import *', namespace)
assert tuple(pyWitnessAI.__all__) == before
assert set(before) <= namespace.keys()
assert {'Images', 'Lineup', 'LineupBuilder', 'LineupMember', 'image_paths', 'image_manifest', 'select_by_similarity', 'FaceSearch', 'ImageCatalog', 'FillerSelector',
        'FaceGenerator', 'GeneratedFace', 'FaceStandardizer', 'StandardizedFace', 'CodeFormerEnhancer', 'SchemaDescriptionParser',
        'ImageGenerationBackend', 'ImageGenerationRequest', 'OpenAIImageBackend',
        'ParsedDescription', 'ContrastScoring',
        'FillerSelectionBenchmark', 'AttributeRetrievalBenchmark'} <= set(before)
assert {'ImagesAI', 'Video', 'VideoAI', 'VideoProcessor', 'FrameAnalyzerDeepface',
        'FrameProcessorCropper', 'VideoLineupPipeline'}.isdisjoint(before)
assert not optional_roots.intersection(name.split('.')[0] for name in sys.modules)
assert callable(namespace['FillerSelector'])
assert callable(namespace['FaceSearch'])
""")


def test_api_package_and_core_interfaces_do_not_load_service_or_provider_adapters():
    _run_fresh(_BLOCK_OPTIONAL + """
import pyWitnessAI.API
from pyWitnessAI.face_generation_backend import (
    ImageGenerationBackend, available_image_generation_providers,
)
from pyWitnessAI.API.models import download_model

assert 'openai' in available_image_generation_providers()
assert 'pyWitnessAI.API.http' not in sys.modules
assert 'pyWitnessAI.API.openai' not in sys.modules
assert not optional_roots.intersection(name.split('.')[0] for name in sys.modules)
""")


def test_generation_exports_use_only_new_modules_without_loading_the_sdk():
    _run_fresh(_BLOCK_OPTIONAL + """
import importlib
import importlib.util
import pyWitnessAI
from pyWitnessAI import FaceGenerator, GeneratedFace, ImageGenerationBackend, OpenAIImageBackend

backend_module = importlib.import_module('pyWitnessAI.face_generation_backend')
assert ImageGenerationBackend is backend_module.ImageGenerationBackend
assert OpenAIImageBackend is importlib.import_module('pyWitnessAI.API.openai').OpenAIImageBackend
assert GeneratedFace.__module__ == 'pyWitnessAI.FaceGenerator'
generator = FaceGenerator(description='blue eyes')
assert 'blue eyes' in generator.generation_prompts()[0]
assert generator.results == []
for removed in ('FillerGenerator', 'GeneratedFiller'):
    assert removed not in pyWitnessAI.__all__, removed
    assert removed not in dir(pyWitnessAI), removed
    assert not hasattr(pyWitnessAI, removed), removed
for removed_module in ('FillerGenerator', 'filler_generation_backends'):
    assert importlib.util.find_spec('pyWitnessAI.' + removed_module) is None, removed_module
assert not optional_roots.intersection(name.split('.')[0] for name in sys.modules)
""")


@pytest.mark.parametrize("name", [
    "FaceSearch", "ImageCatalog", "FillerSelector", "FaceAttributeSchema", "Images", "FaceGenerator", "FaceStandardizer", "LineupBuilder", "LineupSession",
])
@pytest.mark.parametrize("order", ["submodule_first", "public_first"])
def test_class_and_submodule_names_are_stable_across_import_order(name, order):
    _run_fresh(_BLOCK_OPTIONAL + f"""
import importlib
from types import ModuleType
import pyWitnessAI

name = {name!r}
if {order!r} == 'public_first':
    original = getattr(pyWitnessAI, name)
module = importlib.import_module('pyWitnessAI.' + name)
assert isinstance(module, ModuleType)
expected = getattr(module, name)
assert getattr(pyWitnessAI, name) is expected
namespace = {{}}
exec('from pyWitnessAI import ' + name, namespace)
assert namespace[name] is expected
if {order!r} == 'public_first':
    assert original is expected
# Other modules may import and assign the same submodule to the package.
importlib.import_module('pyWitnessAI.FillerSelectionBenchmark')
assert getattr(pyWitnessAI, name) is expected
assert importlib.import_module('pyWitnessAI.' + name) is module
""")


def test_analyzer_processor_and_pipeline_exports_resolve_their_actual_owner():
    _run_fresh("""
import sys
from types import ModuleType
import pyWitnessAI

before = tuple(pyWitnessAI.__all__)
owners = {
    'VideoAI': (
        'FrameAnalyzerMTCNNIndependent', 'FrameAnalyzerOpenCVIndependent',
        'FrameAnalyzerDeepface', 'FrameAnalyzerMTCNN', 'FrameAnalyzerOpenCV',
        'FrameAnalyzerFastMTCNN', 'SimilarityAnalyzer', 'LineupLoader',
    ),
    'VideoProcessor': (
        'FrameProcessorCropper', 'FrameProcessorMonochrome',
        'FrameProcessorNormalizer', 'FrameProcessorRemover',
        'FrameProcessorVideoWriter', 'FrameProcessorDisplayer',
        'FrameProcessorHistogramEqualization', 'FrameProcessorNoiseReduction',
        'FrameProcessorGammaCorrection', 'FrameProcessorSharpening',
    ),
    'VideoLineupPipeline': ('PipelineConfig', 'VideoLineupPipeline', 'export_for_pywitness'),
}
for module_name, names in owners.items():
    module = ModuleType('pyWitnessAI.' + module_name)
    for name in names:
        setattr(module, name, type(name, (), {'__module__': module.__name__}))
    sys.modules[module.__name__] = module
    for name in names:
        namespace = {}
        exec('from pyWitnessAI import ' + name, namespace)
        assert namespace[name] is getattr(module, name)
    if module_name in {'VideoAI', 'VideoProcessor'}:
        assert getattr(pyWitnessAI, module_name) is module
assert tuple(pyWitnessAI.__all__) == before
assert 'FrameAnalyzerFaceIdentity' not in dir(pyWitnessAI)
""")


def test_missing_optional_dependency_reports_extra_and_preserves_cause():
    _run_fresh(_BLOCK_OPTIONAL + """
import pyWitnessAI

try:
    from pyWitnessAI import FrameAnalyzerOpenCV
except ModuleNotFoundError as exc:
    assert exc.name == 'cv2'
    assert 'FrameAnalyzerOpenCV' in str(exc)
    assert 'pip install "pyWitnessAI[vision]"' in str(exc)
    assert isinstance(exc.__cause__, ModuleNotFoundError)
else:
    raise AssertionError('Missing optional dependencies must remain visible')
assert 'FrameAnalyzerOpenCV' not in pyWitnessAI._EXPORT_CACHE
""")


def test_only_known_missing_dependency_roots_are_relabelled_as_extras():
    _run_fresh("""
import pyWitnessAI

expected = {
    'sentence_transformers': 'clip', 'matplotlib': 'plot',
    'IPython': 'notebook', 'openai': 'generation', 'fastapi': 'api',
}
for dependency, extra in expected.items():
    error = ModuleNotFoundError('missing dependency', name=dependency)
    def missing(_module, error=error):
        raise error
    pyWitnessAI._import_module = missing
    try:
        pyWitnessAI.FillerSelector
    except ModuleNotFoundError as exc:
        assert f'pyWitnessAI[{extra}]' in str(exc)
        assert exc.__cause__ is error
    else:
        raise AssertionError(dependency)
""")


def test_real_implementation_errors_are_not_hidden_or_relabelled():
    _run_fresh("""
import pyWitnessAI

errors = (
    RuntimeError('implementation failed'),
    ValueError('invalid implementation configuration'),
    ImportError('installed dependency has incompatible API'),
    ModuleNotFoundError('internal module missing', name='pyWitnessAI.missing_helper'),
    ModuleNotFoundError('installed torch is broken', name='torch._C'),
    ModuleNotFoundError('base dependency missing', name='numpy'),
    ModuleNotFoundError('unknown import typo', name='misspelled_internal_helper'),
)
for error in errors:
    def broken(_module, error=error):
        raise error
    pyWitnessAI._import_module = broken
    try:
        pyWitnessAI.ParsedDescription
    except Exception as exc:
        assert exc is error
    else:
        raise AssertionError(type(error).__name__)
assert 'ParsedDescription' not in pyWitnessAI._EXPORT_CACHE
""")


def test_unknown_exports_raise_attribute_error_without_importing_anything():
    _run_fresh("""
import pyWitnessAI

def unexpected(_module):
    raise AssertionError('Unknown names must not trigger imports')

pyWitnessAI._import_module = unexpected
try:
    pyWitnessAI.unknown_public_name
except AttributeError as exc:
    assert 'unknown_public_name' in str(exc)
else:
    raise AssertionError('Unknown export accepted')
""")


@pytest.mark.parametrize("installed_version", ["9.8.7", None])
def test_version_falls_back_to_distribution_metadata_without_generated_file(installed_version):
    _run_fresh(f"""
import importlib.abc
import importlib.metadata
import sys

class NoGeneratedVersion(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyWitnessAI._version':
            raise ModuleNotFoundError('no generated version file', name=fullname)

def version(_name):
    if {installed_version!r} is None:
        raise importlib.metadata.PackageNotFoundError(_name)
    return {installed_version!r}

sys.meta_path.insert(0, NoGeneratedVersion())
importlib.metadata.version = version
import pyWitnessAI
assert pyWitnessAI.__version__ == {installed_version or '0.0.1'!r}
""")


def test_main_is_a_lazy_public_callable():
    _run_fresh("""
import sys
from types import ModuleType
import pyWitnessAI

assert 'pyWitnessAI.API.cli' not in sys.modules
module = ModuleType('pyWitnessAI.API.cli')
module.main = lambda: 0
sys.modules[module.__name__] = module
from pyWitnessAI import main
assert main is module.main
""")
