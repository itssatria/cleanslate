"""Gradio UI: folder -> thumbnails -> full image -> paint mask -> remove -> save."""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from .backends import BACKENDS, DEFAULT_PROMPT, BackendError
from .imaging import (
    extract_mask,
    grow_mask,
    list_images,
    load_image,
    make_thumb,
    paste_masked,
    save_result,
)

DEFAULT_BACKEND = next(iter(BACKENDS))


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


def load_folder(folder):
    try:
        paths = list_images(folder)
    except ValueError as e:
        raise gr.Error(str(e))
    items = []
    for p in paths:
        try:
            items.append((make_thumb(p), p.name))
        except Exception:
            continue  # unreadable file: skip
    return items, [str(p) for p in paths], f"{len(items)} images loaded."


def open_image(files, evt: gr.SelectData):
    path = files[evt.index]
    img = load_image(path)
    editor = {"background": img, "layers": [], "composite": None}
    return editor, path, None, f"Opened **{Path(path).name}** ({img.width}x{img.height})"


def toggle_api_fields(name):
    return gr.update(visible=BACKENDS[name].uses_api)


def remove(editor, path, backend_name, grow_px, api_key, prompt):
    if not path or not editor or editor.get("background") is None:
        raise gr.Error("Select an image first.")
    img = editor["background"].convert("RGB")
    backend = BACKENDS[backend_name]
    try:
        mask = grow_mask(extract_mask(editor.get("layers"), img.size), int(grow_px))
        out = backend.inpaint(img, mask, api_key=api_key or None, prompt=prompt or None)
    except (ValueError, BackendError) as e:
        raise gr.Error(str(e))
    if backend.composite:
        out = paste_masked(img, out, mask, feather=backend.feather)
    return out, out, "Done. Save it, or **Continue editing** to fix leftovers."


def continue_editing(result):
    if result is None:
        raise gr.Error("Nothing to continue from yet.")
    return {"background": result, "layers": [], "composite": None}


def save(result, path):
    if result is None or not path:
        raise gr.Error("Nothing to save yet.")
    return f"Saved to `{save_result(result, path)}`"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Watermark Remover") as demo:
        gr.Markdown("## Watermark Remover")
        files_state, path_state, result_state = gr.State([]), gr.State(None), gr.State(None)

        with gr.Row():
            folder = gr.Textbox(label="Image folder", placeholder=r"C:\Images", scale=6)
            browse_btn = gr.Button("Browse...", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)

        gallery = gr.Gallery(label="Images (click one)", columns=8, height=220,
                             object_fit="cover", allow_preview=False)
        status = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=3):
                editor = gr.ImageEditor(
                    label="Paint over the watermark",
                    type="pil",
                    sources=[],
                    transforms=[],
                    brush=gr.Brush(default_size=25, colors=["#ff0000"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=25),
                    layers=True,
                    height=650,
                )
            with gr.Column(scale=2):
                backend = gr.Radio(list(BACKENDS), value=DEFAULT_BACKEND, label="Backend")
                grow_px = gr.Slider(0, 30, value=6, step=1, label="Grow mask (px)")
                api_key = gr.Textbox(label="OpenAI API key (or set OPENAI_API_KEY)",
                                     type="password", visible=False)
                prompt = gr.Textbox(label="API prompt", value=DEFAULT_PROMPT, lines=2, visible=False)
                run_btn = gr.Button("Remove", variant="primary")
                result_img = gr.Image(label="Result", type="pil", interactive=False)
                with gr.Row():
                    cont_btn = gr.Button("Continue editing")
                    save_btn = gr.Button("Save to /cleaned")

        browse_btn.click(browse_folder, outputs=folder)
        load_btn.click(load_folder, folder, [gallery, files_state, status])
        folder.submit(load_folder, folder, [gallery, files_state, status])
        gallery.select(open_image, files_state, [editor, path_state, result_img, status])
        backend.change(toggle_api_fields, backend, api_key).then(toggle_api_fields, backend, prompt)
        run_btn.click(remove, [editor, path_state, backend, grow_px, api_key, prompt],
                      [result_img, result_state, status])
        cont_btn.click(continue_editing, result_state, editor)
        save_btn.click(save, [result_state, path_state], status)
    return demo


def main():
    ap = argparse.ArgumentParser(description="Watermark Remover (Gradio)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port,
                      share=args.share, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
