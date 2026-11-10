# Changelog

All notable changes to Cleanslate are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-11

### Added
- **Batch tab**: paint the watermark once on a reference image and apply the same mask to every image
  in the folder. Results go to `cleaned/`.
  - Mask is stored relative to the image, so it can **scale** to photos of a different resolution
    (or those images can be **skipped** instead).
  - Existing outputs can be **skipped** (resume a stopped run) or **overwritten**.
  - One failing image never stops the run; it is logged and the batch continues.
  - Live progress bar, status line and log. Gallery captions show `✓` done, `–` skipped, `✗` failed, `…` running.
  - **Stop** button (takes effect after the image currently being processed).
- **Thumbnail size slider** (64-256 px) that re-flows the gallery.
- Folder **auto-loads**: thumbnails appear right after Browse or pressing Enter in the folder field.
  A progress bar shows thumbnail creation on large folders.
- `cleanslate --version`.
- `batch.py` module (`run_batch`) and unit tests for batching, mask scaling and output paths.
- `CHANGELOG.md` and a much more detailed `README.md`.

### Changed
- **New layout**: folder and gallery on the left, work area on the right.
- Work area split into **Single image** and **Batch** tabs. In Single image, **Mask** and **Result** are
  sub-tabs; the view switches to Result after Remove and back to Mask on Continue editing.
- The **Load** button was removed (loading is automatic).
- OpenAI API key and prompt moved into a collapsed **API settings** accordion in the sidebar, shared by both tabs.
- Thumbnails are generated at 320 px (was 256) so large sizes stay sharp.

### Fixed
- If an unreadable image was skipped while building thumbnails, clicking later thumbnails could open the
  wrong file. Gallery and file list are now always built together.
- Opening a new image no longer leaves the previous result in memory, which could save the old result
  under the new file's name.

## [0.2.0] - 2026-11-10

### Added
- Renamed the project to **Cleanslate** (package `cleanslate`, command `cleanslate`).
- Automatic LaMa model download on first use, with a progress bar and fallback URLs. The app checks for an
  existing file first (`LAMA_MODEL`, Cleanslate's folder, then the PyTorch hub cache).
- `cleanslate --download-models`, plus `LAMA_MODEL`, `LAMA_MODEL_URL` and `CLEANSLATE_DEVICE` settings.
- Unit tests for model download logic and LaMa helper functions.

### Changed
- LaMa is now loaded directly from the TorchScript weights (always mapped to CPU first, then moved to the
  chosen device). Only `torch` is required: `torchvision` and `simple-lama-inpainting` were dropped.
- LaMa runs on a crop around the mask instead of the whole image, which is much faster on CPU.
- Backend failures are shown as readable messages in the UI instead of raw tracebacks.

### Fixed
- `Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend` when loading LaMa on a
  machine with a CPU-only PyTorch build.

## [0.1.0] - 2026-11-10

### Added
- First release: Gradio app to pick a folder, browse thumbnails, open an image, paint a mask and remove it.
- Backends: LaMa (local), OpenCV Telea (local) and OpenAI `gpt-image-1` (API), behind a small backend interface.
- Results saved to a `cleaned/` subfolder; only the masked area is changed.
- Project scaffolding: `pyproject.toml`, tests, CI workflow, MIT license.

[0.3.0]: https://github.com/<your-user>/cleanslate/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/<your-user>/cleanslate/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/<your-user>/cleanslate/releases/tag/v0.1.0
