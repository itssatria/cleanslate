# Watermark Remover

A small Gradio app for removing watermarks and unwanted objects from images, with a Photoshop-style
workflow: pick a folder, click a thumbnail, paint over the area, and remove it.

- Browse a folder and get a thumbnail gallery
- Open any image at full resolution
- Paint a mask with a brush (and erase mistakes)
- Inpaint with a **local model (LaMa)**, **OpenCV**, or an **API** (OpenAI `gpt-image-1`)
- Only the masked area is changed; the rest of the image stays pixel-identical
- Results are saved to a `cleaned/` subfolder, so originals are never overwritten
- Iterate: send the result back into the editor to fix leftovers

<!-- Add a screenshot: docs/screenshot.png -->

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/itssatria/cleanslate.git
cd cleanslate
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[all]"
```

Install only what you need:

| Extra | Adds |
|---|---|
| *(none)* | UI + OpenCV backend |
| `.[lama]` | local LaMa model (`simple-lama-inpainting`, pulls in PyTorch) |
| `.[openai]` | OpenAI API backend |
| `.[all]` | everything |

For GPU speed with LaMa, install a CUDA build of PyTorch first (see pytorch.org), then the extras.
On Windows you can also just double-click `run.bat`.

## Run

```bash
wmremover                 # or: python -m wmremover
wmremover --port 8000 --no-browser
```

## Use

1. **Browse** (or paste a path) and press **Load**.
2. Click a thumbnail to open it.
3. Paint over the watermark. Slightly oversized strokes work better; the **Grow mask** slider helps too.
4. Choose a backend and press **Remove**.
5. **Save to /cleaned**, or **Continue editing** to mask what's left and run again.

## Backends

| Backend | Where | Best for |
|---|---|---|
| LaMa | local | textured backgrounds, large marks, no cost |
| OpenCV (Telea) | local | thin lines or tiny marks on simple backgrounds, instant |
| OpenAI API | cloud | complex scenes; needs an API key (paste in UI or set `OPENAI_API_KEY`) |

The API backend resizes to a supported size, edits, then pastes only the masked area back at full
resolution. Images are sent to the provider, so avoid it for private material.

### Add your own backend

Create `src/wmremover/backends/my_backend.py`:

```python
from .base import Backend

class MyBackend(Backend):
    name = "My model"
    def inpaint(self, image, mask, *, api_key=None, prompt=None):
        ...  # image: RGB PIL, mask: uint8 (h, w) with 255 = remove; return RGB PIL
```

Then add an instance to the tuple in `backends/__init__.py`. It appears in the UI automatically.
A ComfyUI, Replicate or Stability backend fits the same pattern.

## Development

```bash
pip install -e ".[dev]"
pytest
```

```
src/wmremover/
  app.py          Gradio UI + CLI
  imaging.py      folder scan, thumbnails, mask + compositing helpers
  backends/       one file per inpainting backend
tests/
```

## Responsible use

Only remove watermarks from images you own or have permission to edit. Watermarks often
indicate ownership, and removing them from others' work may violate copyright or licensing terms.

## License

MIT. See [LICENSE](LICENSE).
