# Regression fixtures

`georgia_state_video1/` contains eight PNG files copied unchanged from the
repository's existing `data/01_Georgia_State_Video1/` files. They are retained as
fixed inputs for image-loading tests and the existing ImagesAI similarity
regressions. Their filenames and image contents preserve those reference results.

These fixtures accompany the source-distribution tests and are not installed
with the library wheel. Research datasets and generated results belong in the
ignored `data/` directory; model files use their configured local caches.
