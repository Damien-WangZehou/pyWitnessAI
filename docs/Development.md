# Development and release builds

Run the following commands from the repository root with Python 3.10 or 3.11.
Install the development tools in your chosen environment:

```powershell
python -m pip install -e ".[dev]"
```

The base package remains NumPy, pandas and Pillow. Install additional extras
only for the integrations you intend to run; their setup is described in the
[README](../README.md).

## Tests

```powershell
python -m pytest
```

Default discovery is limited to the existing `test/` directory. Individual
tests that exercise optional integrations may require their corresponding
extras. Study-specific tests should be run explicitly from their study's own
instructions; removing a study does not change the core test-discovery paths.

For the generation and portrait-standardization regression suite:

```powershell
python -m pytest test/test_face_generator.py test/test_face_standardizer.py test/test_face_attribute_schema.py test/test_prompt_processing.py test/test_face_generator_prompt_processing.py test/test_filler_profiles.py test/test_lineup_session.py test/test_public_imports.py -q
```

These tests use local fixtures and fake providers instead of paid image or text
generation requests.

## Release artifacts

```powershell
python -m build --outdir dist
```

Use the default build mode: it creates a source distribution, extracts it into
a temporary directory, and builds the wheel from that source distribution.
This checks that the source archive contains the files needed to build the
package and keeps old files under the working tree's `build/lib/` out of the
wheel. `--wheel`, or explicitly passing both `--sdist` and `--wheel`, builds the
wheel directly from the working tree and bypasses this check.

`MANIFEST.in` excludes study directories, local datasets, notebooks, build
outputs and the unused FaceNet weight file. The wheel should contain the
current `pyWitnessAI` modules and four OpenCV XML resources. Source archives
also include documentation, regression tests and the small test PNG fixtures.
Check the resulting archives before publishing; package configuration alone
does not verify the contents of an old build directory.

`src/pyWitnessAI.egg-info/` and `src/pyWitnessAI/_version.py` are generated
metadata. Rebuild them through the packaging tools after source or README
changes; do not edit cached file lists or copied package descriptions by hand.

## Historical local artifacts

Old indexes, validation results, virtual environments and build copies can be
retained under the ignored `build/` workspace for research provenance. They do
not describe the current source tree and should not be added to `PYTHONPATH`,
used as current smoke tests, or copied into a release. Preserve any records
needed to reproduce past results before deciding whether to remove them.

For a retired study, use Git history to locate its source and the matching
study-specific configuration. The core package does not supply the removed
study's candidate pool, frozen scoring profile or service. Current reusable
interfaces are documented in the [module map](CoreModuleMap.zh-CN.md),
[selection guide](FillerSelection.md), and [lineup/session guide](LineupSession.md).
