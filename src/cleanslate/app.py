"""Gradio UI. Left: folder, thumbnail size, gallery. Right: Single image and Batch tabs."""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from . import __version__
from .backends import BACKENDS, DEFAULT_PROMPT, BackendError
from .batch import run_batch
from .imaging import (
    extract_mask,
    grow_mask,
    list_images,
    load_image,
    make_thumb,
    output_path,
    paste_masked,
    save_result,
)

DEFAULT_BACKEND = next(iter(BACKENDS))
DEFAULT_THUMB = 112
SYMBOL = {"running": "… ", "done": "✓ ", "skipped": "– ", "failed": "✗ "}


def columns_for(size: int) -> int:
    """Gallery columns for a thumbnail size in px (the sidebar is ~300 px wide)."""
    return max(1, min(6, round(300 / max(int(size), 1))))


def editor_value(img):
    return {"background": img, "layers": [], "composite": None}


# --------------------------------------------------------------------------- folder / gallery
def browse_folder():
    """Native folder picker. Works because the app runs on the user's own machine."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory()
        root.destroy()
        return path or gr.update()
    except Exception:
        return gr.update()


def load_folder(folder, progress=gr.Progress()):
    """Runs automatically after Browse or Enter. Builds thumbnails (cached) and fills the gallery."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        return gr.update(), gr.update(), gr.update(), gr.update()
    try:
        paths = list_images(folder)
    except ValueError as e:
        raise gr.Error(str(e))

    items, files, thumbs = [], [], []
    for i, p in enumerate(paths):
        if i % 10 == 0:
            progress(i / len(paths), desc=f"Creating thumbnails {i}/{len(paths)}")
        try:
            t = make_thumb(p)
        except Exception:
            continue  # unreadable file: skip it everywhere so gallery and file list stay aligned
        items.append((t, p.name))
        files.append(str(p))
        thumbs.append(t)
    skipped = len(paths) - len(files)
    msg = f"**{len(files)}** images loaded" + (f" ({skipped} unreadable skipped)" if skipped else "")
    return items, files, thumbs, msg


def resize_gallery(size):
    return gr.update(columns=columns_for(size))


def open_image(files, evt: gr.SelectData):
    path = files[evt.index]
    img = load_image(path)
    name = Path(path).name
    info = f"Opened **{name}** ({img.width}×{img.height})"
    ref = f"Reference: **{name}** ({img.width}×{img.height}). Paint over the watermark once."
    # outputs: single editor, path, result image, result state, single status, batch editor, batch note, tabs
    return editor_value(img), path, None, None, info, editor_value(img), ref, gr.Tabs(selected="mask")


# --------------------------------------------------------------------------- single image
def remove(editor, path, backend_name, grow_px, api_key, prompt, progress=gr.Progress()):
    if not path or not editor or editor.get("background") is None:
        raise gr.Error("Select an image first.")
    img = editor["background"].convert("RGB")
    backend = BACKENDS[backend_name]
    try:
        mask = grow_mask(extract_mask(editor.get("layers"), img.size), int(grow_px))
        # Checks the model file exists; downloads it automatically the first time.
        backend.ensure_ready(lambda frac, msg: progress(frac, desc=msg))
        progress(None, desc="Removing...")
        out = backend.inpaint(img, mask, api_key=api_key or None, prompt=prompt or None)
    except (ValueError, BackendError) as e:
        raise gr.Error(str(e))
    if backend.composite:
        out = paste_masked(img, out, mask, feather=backend.feather)
    return out, out, "Done. Save it, or **Continue editing** to fix leftovers.", gr.Tabs(selected="result")


def continue_editing(result):
    if result is None:
        raise gr.Error("Nothing to continue from yet.")
    return editor_value(result), gr.Tabs(selected="mask")


def save(result, path):
    if result is None or not path:
        raise gr.Error("Nothing to save yet.")
    return f"Saved to `{save_result(result, path)}`"


# --------------------------------------------------------------------------- batch
def _captions(thumbs, names, state):
    return [(t, SYMBOL.get(state.get(i), "") + n) for i, (t, n) in enumerate(zip(thumbs, names))]


def batch_run(editor, files, thumbs, backend_name, grow_px, mismatch, existing,
              api_key, prompt, progress=gr.Progress()):
    if not files:
        raise gr.Error("Load a folder first.")
    if not editor or editor.get("background") is None:
        raise gr.Error("Click a thumbnail to choose the reference image first.")
    img = editor["background"].convert("RGB")
    try:
        ref_mask = extract_mask(editor.get("layers"), img.size)
    except ValueError as e:
        raise gr.Error(str(e))

    backend = BACKENDS[backend_name]
    try:
        backend.ensure_ready(lambda frac, msg: progress(frac, desc=msg))
    except BackendError as e:
        raise gr.Error(str(e))

    names = [Path(f).name for f in files]
    total = len(files)
    state: dict[int, str] = {}
    counts = {"done": 0, "skipped": 0, "failed": 0}
    log: list[str] = []
    every = 1 if total <= 200 else 5  # fewer gallery refreshes on huge folders

    for ev in run_batch(files, ref_mask, backend, grow_px=int(grow_px), on_size_mismatch=mismatch,
                        on_existing=existing, api_key=api_key or None, prompt=prompt or None):
        state[ev.index] = ev.status
        progress(ev.index / total, desc=f"{ev.index + 1}/{total}  {names[ev.index]}")
        if ev.status == "running":
            yield gr.update(), "\n".join(log[-300:]), f"Cleaning **{names[ev.index]}**  ({ev.index + 1}/{total})"
            continue
        counts[ev.status] += 1
        tag = {"done": f"done in {ev.seconds:.1f}s", "skipped": f"skipped ({ev.message})",
               "failed": f"FAILED: {ev.message}"}[ev.status]
        log.append(f"[{ev.index + 1}/{total}] {names[ev.index]}: {tag}")
        gallery = _captions(thumbs, names, state) if (ev.index % every == 0) else gr.update()
        yield gallery, "\n".join(log[-300:]), (
            f"**{ev.index + 1} / {total}**  ·  {counts['done']} cleaned  ·  "
            f"{counts['skipped']} skipped  ·  {counts['failed']} failed")

    out_dir = output_path(files[0]).parent
    yield _captions(thumbs, names, state), "\n".join(log[-300:]), (
        f"**Finished.** {counts['done']} cleaned, {counts['skipped']} skipped, {counts['failed']} failed. "
        f"Saved to `{out_dir}`")


# --------------------------------------------------------------------------- UI
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Cleanslate") as demo:
        files_state, thumbs_state = gr.State([]), gr.State([])
        path_state, result_state = gr.State(None), gr.State(None)

        with gr.Row(equal_height=False):
            # ---------------- left: folder + gallery ----------------
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("## Cleanslate")
                with gr.Row():
                    folder = gr.Textbox(label="Image folder", placeholder=r"C:\Images", scale=4,
                                        info="Press Enter after pasting a path")
                    browse_btn = gr.Button("Browse…", scale=1, min_width=90)
                status = gr.Markdown("Pick a folder to load its images.")
                thumb_size = gr.Slider(64, 256, value=DEFAULT_THUMB, step=16, label="Thumbnail size (px)")
                gallery = gr.Gallery(show_label=False, columns=columns_for(DEFAULT_THUMB), height=560,
                                     object_fit="cover", allow_preview=False)
                with gr.Accordion("API settings (OpenAI backend)", open=False):
                    api_key = gr.Textbox(label="OpenAI API key (or set OPENAI_API_KEY)", type="password")
                    prompt = gr.Textbox(label="API prompt", value=DEFAULT_PROMPT, lines=2)

            # ---------------- right: work area ----------------
            with gr.Column(scale=3):
                with gr.Tabs():
                    # ---- Single image ----
                    with gr.Tab("Single image"):
                        with gr.Tabs() as view_tabs:
                            with gr.Tab("Mask", id="mask"):
                                editor = gr.ImageEditor(
                                    show_label=False, type="pil", sources=[], transforms=[],
                                    brush=gr.Brush(default_size=25, colors=["#ff0000"], color_mode="fixed"),
                                    eraser=gr.Eraser(default_size=25), layers=True, height=600)
                            with gr.Tab("Result", id="result"):
                                result_img = gr.Image(show_label=False, type="pil", interactive=False, height=600)
                                with gr.Row():
                                    cont_btn = gr.Button("Continue editing")
                                    save_btn = gr.Button("Save to /cleaned", variant="primary")
                        with gr.Row():
                            backend = gr.Radio(list(BACKENDS), value=DEFAULT_BACKEND, label="Backend", scale=3)
                            grow_px = gr.Slider(0, 30, value=6, step=1, label="Grow mask (px)", scale=2)
                        run_btn = gr.Button("Remove", variant="primary")
                        single_status = gr.Markdown()

                    # ---- Batch ----
                    with gr.Tab("Batch"):
                        gr.Markdown("### Batch clean\nPaint the watermark **once**. Cleanslate applies the same "
                                    "mask to every image in the folder and saves results to `cleaned/`.")
                        ref_note = gr.Markdown("Click a thumbnail to choose the reference image.")
                        batch_editor = gr.ImageEditor(
                            show_label=False, type="pil", sources=[], transforms=[],
                            brush=gr.Brush(default_size=25, colors=["#ff0000"], color_mode="fixed"),
                            eraser=gr.Eraser(default_size=25), layers=True, height=520)
                        with gr.Row():
                            b_backend = gr.Radio(list(BACKENDS), value=DEFAULT_BACKEND, label="Backend", scale=3)
                            b_grow = gr.Slider(0, 30, value=6, step=1, label="Grow mask (px)", scale=2)
                        with gr.Row():
                            b_mismatch = gr.Radio(
                                [("Scale mask", "scale"), ("Skip image", "skip")], value="scale",
                                label="If an image has a different size",
                                info="The mask is relative to the image, so it follows the watermark when sizes differ.")
                            b_existing = gr.Radio(
                                [("Skip it", "skip"), ("Overwrite", "overwrite")], value="skip",
                                label="If a cleaned file already exists",
                                info="Skip lets you stop and resume a long run.")
                        with gr.Row():
                            b_run = gr.Button("Clean all images", variant="primary", scale=3)
                            b_stop = gr.Button("Stop", scale=1)
                        b_status = gr.Markdown()
                        b_log = gr.Textbox(label="Log", lines=8, max_lines=8, interactive=False, autoscroll=True)

        # ---------------- events ----------------
        load_outputs = [gallery, files_state, thumbs_state, status]
        folder.submit(load_folder, folder, load_outputs)
        browse_btn.click(browse_folder, outputs=folder).then(load_folder, folder, load_outputs)  # auto-loads
        thumb_size.change(resize_gallery, thumb_size, gallery)
        gallery.select(open_image, files_state,
                       [editor, path_state, result_img, result_state, single_status,
                        batch_editor, ref_note, view_tabs])

        run_btn.click(remove, [editor, path_state, backend, grow_px, api_key, prompt],
                      [result_img, result_state, single_status, view_tabs])
        cont_btn.click(continue_editing, result_state, [editor, view_tabs])
        save_btn.click(save, [result_state, path_state], single_status)

        batch_evt = b_run.click(
            batch_run,
            [batch_editor, files_state, thumbs_state, b_backend, b_grow, b_mismatch, b_existing, api_key, prompt],
            [gallery, b_log, b_status])
        b_stop.click(None, None, None, cancels=[batch_evt])
    return demo


def main():
    ap = argparse.ArgumentParser(description="Cleanslate: paint over a watermark or object and erase it")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--download-models", action="store_true",
                    help="download local model files now instead of on first use, then exit")
    ap.add_argument("--version", action="version", version=f"cleanslate {__version__}")
    args = ap.parse_args()
    if args.download_models:
        for b in BACKENDS.values():
            b.ensure_ready(lambda frac, msg: print(f"\r{msg}", end="", flush=True))
        print("\nModels ready.")
        return
    build_ui().launch(server_name=args.host, server_port=args.port,
                      share=args.share, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
