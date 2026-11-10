# Cleanslate

Paint over a watermark or an unwanted object and wipe it away. Cleanslate is a local Gradio app with a
Photoshop-style "remove" workflow, plus a **batch mode** that applies one mask to a whole folder.

- **Folder + gallery on the left**, work area on the right
- **Single image**: open a photo at full resolution, brush over the area, remove it
- **Batch**: paint the watermark once, clean every image in the folder with the same mask
- **Three backends**: LaMa (local AI model), OpenCV (instant, local), OpenAI `gpt-image-1` (API)
- Only the masked area is changed; the rest of the image stays pixel-identical
- Originals are never touched: results go to a `cleaned/` subfolder
- Runs on CPU. No GPU required

<!-- Add screenshots: docs/single.png, docs/batch.png -->

## Contents

- [Requirements](#requirements) · [Installation](#installation) · [Quick start](#quick-start)
- [Using Cleanslate](#using-cleanslate): [Folder and gallery](#folder-and-gallery) · [Single image](#single-image) · [Batch](#batch)
- [Backends](#backends) · [Model download](#model-download) · [Configuration](#configuration)
- [Performance tips](#performance-tips) · [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure) · [Adding a backend](#adding-a-backend) · [Development](#development)
- [Responsible use](#responsible-use) · [License](#license)

## Requirements

- Python 3.10 or newer
- About 1 GB of disk space if you use LaMa (PyTorch plus a ~200 MB model file)
- Internet access once, to download the LaMa model (not needed afterwards), or always for the OpenAI backend
- Windows, macOS or Linux. The **Browse…** button uses the system folder dialog (tkinter), so it works when
  the app runs on your own machine. You can always paste a path instead.

## Installation

```bash
git clone https://github.com/<your-user>/cleanslate.git
cd cleanslate
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[all]"
```

| Install | Gives you |
|---|---|
| `pip install -e .` | UI + OpenCV backend |
| `pip install -e ".[lama]"` | adds the local LaMa backend (`torch`) |
| `pip install -e ".[openai]"` | adds the OpenAI API backend |
| `pip install -e ".[all]"` | everything |

**PyTorch notes**
- On Windows, the default `torch` wheel from PyPI is the CPU build. That is all you need without an NVIDIA GPU.
  (Intel/AMD integrated graphics are not used by PyTorch; Cleanslate runs on the CPU.)
- NVIDIA GPU: install a CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then the extras.
- Windows shortcut: double-click `run.bat`. It creates a virtual environment on first run, installs everything and starts the app.

## Quick start

```bash
cleanslate              # or: python -m cleanslate
```

The app opens in your browser at http://127.0.0.1:7860. Then:

1. Press **Browse…** and pick a folder. Thumbnails load automatically.
2. Click a thumbnail, paint over the watermark, press **Remove**.
3. Happy with it? **Save to /cleaned**. Have a lot of images with the same watermark? Use the **Batch** tab.

Command-line options:

| Option | Meaning |
|---|---|
| `--host`, `--port` | where to listen (default `127.0.0.1:7860`) |
| `--no-browser` | do not open a browser tab |
| `--share` | create a temporary public Gradio link (anyone with it can use your app. Use with care) |
| `--download-models` | download local model files now, then exit |
| `--version` | print the version |

## Using Cleanslate

### Folder and gallery

The left column is shared by both tabs.

- **Image folder**: press **Browse…**, or paste a path and press **Enter**. The gallery fills automatically.
  Supported files: JPG, PNG, WebP, BMP, TIFF. Subfolders are not scanned.
- **Thumbnail size** (64-256 px): changes how many thumbnails fit per row. Thumbnails are cached, so reopening a
  folder is fast.
- **Clicking a thumbnail** opens it in the Single image editor *and* sets it as the reference image on the Batch tab.
- **API settings** (collapsed): OpenAI key and prompt, used by whichever tab runs the OpenAI backend.

### Single image

1. Click a thumbnail. The image opens at full resolution on the **Mask** sub-tab.
2. Paint over the watermark with the red brush. Use the eraser to fix mistakes. Slightly oversized strokes work better than tight ones.
3. Choose a **Backend** and a **Grow mask** amount (expands your mask by a few pixels, because watermark edges are soft).
4. Press **Remove**. The view switches to **Result**.
5. Then either **Save to /cleaned**, or **Continue editing** (loads the result back into the editor so you can mask what is left and run again).

### Batch

Use this when many images carry the same watermark in the same place.

1. Load the folder and click one representative image. It becomes the **reference**.
2. On the **Batch** tab, paint the watermark on the reference image once.
3. Pick the backend and **Grow mask**, and decide the two policies below.
4. Press **Clean all images**. Progress, a status line and a log update live. Gallery captions show
   `✓` done, `–` skipped, `✗` failed and `…` for the image being processed.
5. **Stop** ends the run after the image currently being processed.

**If an image has a different size**
- **Scale mask** (default): the mask is stored relative to the image (as percentages), so it is resized to each image and
  follows a watermark that sits in the same relative position. *Grow mask* is scaled the same way.
- **Skip image**: only images with exactly the reference size are processed. Others are logged as skipped.

**If a cleaned file already exists**
- **Skip it** (default): lets you stop and resume a long run without redoing finished images.
- **Overwrite**: re-cleans everything.

**Good to know**
- Output goes to `<your folder>/cleaned/` with the **same file name and format** as the original. Originals are never changed.
- One failing image never stops the run. It is logged as failed and the batch continues.
- Batch runs unattended, so it cannot check that the watermark really is under the mask in every photo. If the
  watermark position or aspect ratio varies, run separate batches for each group of similar photos
  (use a different reference image for each).
- Spot-check a few results in `cleaned/` after a run. Use *Overwrite* after adjusting the mask or backend.

## Backends

| Backend | Runs | Strengths | Limits |
|---|---|---|---|
| **LaMa** | locally (CPU or GPU) | best local quality for textured backgrounds and larger marks. Free, private | ~200 MB model, slower on CPU |
| **OpenCV (Telea)** | locally | instant, nothing to download | only suits thin lines and tiny marks on simple backgrounds |
| **OpenAI API** (`gpt-image-1`) | cloud | strong on complex scenes | costs money, uploads your images, resized for the model then pasted back |

How masks are applied: every backend only changes the masked area (with a soft edge). The rest of the image stays
pixel-identical, even when the API resizes the image internally.

## Model download

The LaMa weights (`big-lama.pt`, ~200 MB) are **downloaded automatically the first time you use LaMa**, with a progress
bar in the UI. Cleanslate checks for an existing file first and skips the download if it finds one. It looks, in order, in:

1. the path in the `LAMA_MODEL` environment variable,
2. `%LOCALAPPDATA%\cleanslate\models` (Windows) or `~/.cache/cleanslate/models` (macOS/Linux),
3. the PyTorch hub cache (`~/.cache/torch/hub/checkpoints`).

If the download fails (offline, blocked), the error message in the UI shows the exact path where you can place the
file manually. A partially downloaded file is detected and replaced.

## Configuration

Environment variables:

| Variable | Purpose |
|---|---|
| `LAMA_MODEL` | path to your own `big-lama.pt` (also where it is downloaded if missing) |
| `LAMA_MODEL_URL` | extra download mirror, tried before the built-in URLs |
| `CLEANSLATE_DEVICE` | force `cpu` or `cuda` for LaMa (default: CUDA if available, else CPU) |
| `OPENAI_API_KEY` | OpenAI key (alternative to typing it in the UI) |
| `TORCH_HOME` | PyTorch cache location, also used when looking for an existing model |

You can put variables in a `.env` file for your own reference (see `.env.example`), but Cleanslate does not load it
automatically. Set them in your shell, e.g. `set LAMA_MODEL=D:\models\big-lama.pt` on Windows.

## Performance tips

- LaMa runs on a **crop around your mask** (plus context) instead of the whole photo, so a corner watermark on a
  large image takes seconds on CPU, not minutes. Keep masks tight, and use separate runs for marks far apart.
- Very large crops are downscaled to at most 1600 px for inference, then the result is pasted back at full resolution.
- For batches, expect roughly a few seconds per image on a modern CPU. Use *Skip it* so interrupted runs can resume.
- OpenCV is the fastest option and is fine for thin text-like marks over smooth areas.

## Troubleshooting

| Problem | What to do |
|---|---|
| `Could not run 'aten::empty_strided' ... 'CUDA' backend` | Your PyTorch is CPU-only but something asked for CUDA. Cleanslate 0.2+ loads the model on the CPU. Make sure you are on the latest version, or set `CLEANSLATE_DEVICE=cpu`. |
| No model download happens | Watch the UI progress bar and the terminal. If the download is blocked, download `big-lama.pt` yourself and set `LAMA_MODEL`, or place it in the folder named in the error message. |
| `LaMa failed to load: ...` | The model file may be corrupt: delete it and run again. Check `python -c "import torch; print(torch.__version__)"`. |
| **Browse…** does nothing | tkinter is missing (some Linux installs: `sudo apt install python3-tk`) or the app runs on a remote machine. Paste the path and press Enter. |
| Results leave a faint ghost of the watermark | Increase **Grow mask**, paint a little beyond the edges, then use **Continue editing** for a second pass. |
| Result looks smeared on a busy background | Try LaMa instead of OpenCV, or the OpenAI backend with a descriptive prompt. |
| Batch skips everything | A `cleaned/` copy already exists (set *If a cleaned file already exists* to **Overwrite**), or sizes differ and *Skip image* is selected. |
| Out of memory | Mask a smaller area, or close other apps. LaMa falls back to CPU automatically if a GPU run fails. |

## Project structure

```
src/cleanslate/
  app.py          Gradio UI (layout, events) and command line
  batch.py        run_batch(): apply one mask to many images, yields progress events
  imaging.py      folder scan, thumbnails, mask scaling/growing, compositing, output paths
  models.py       model lookup and automatic download
  backends/
    base.py       Backend interface
    lama.py       local LaMa (TorchScript)
    opencv_telea.py
    openai_api.py
tests/            unit tests (pytest)
```

## Adding a backend

Create `src/cleanslate/backends/my_backend.py`:

```python
from .base import Backend

class MyBackend(Backend):
    name = "My model"            # label shown in the UI
    composite = True             # paste only the masked area back onto the original
    feather = 3                  # edge softness when compositing

    def ensure_ready(self, progress=None):
        ...                      # optional: download files; call progress(fraction, message)

    def inpaint(self, image, mask, *, api_key=None, prompt=None):
        ...                      # image: RGB PIL; mask: uint8 (h, w), 255 = remove; return RGB PIL
```

Add an instance to the tuple in `backends/__init__.py`. It appears in both tabs automatically.
A ComfyUI, Replicate or Stability backend fits this pattern.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The core logic (`imaging`, `batch`, `models`) has no Gradio dependency and is covered by unit tests, including
batching with a fake backend. CI runs the tests on Python 3.10 and 3.12.
See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Responsible use

Only remove watermarks from images you own or have permission to edit. Watermarks often indicate ownership, and
removing them from other people's work may violate copyright or licensing terms. Batch mode makes this easy to do at
scale, so be especially sure you have the rights to every image in the folder.

## License

MIT. See [LICENSE](LICENSE).
