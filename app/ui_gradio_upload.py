from __future__ import annotations

import io
import os
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr
from PIL import Image

from core.sequence import load_json, ensure_dir, seed_for_frame
from core.line_pass import LinePass
from core.paint_pass import PaintPass


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _pil_open(file_obj) -> Image.Image:
    # gr.File gives a temp file path, gr.Image can give PIL directly depending on config
    if isinstance(file_obj, Image.Image):
        return file_obj
    if hasattr(file_obj, "name"):
        return Image.open(file_obj.name)
    if isinstance(file_obj, str):
        return Image.open(file_obj)
    raise ValueError("Unsupported input type for image.")


def _list_images_in_dir(d: Path) -> List[Path]:
    files = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
    # sort by name (expects 0001.png etc.)
    return sorted(files, key=lambda p: p.name)


def _extract_zip(zip_path: Path, out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Some zips include nested folders; collect images recursively
    imgs = [p for p in out_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs, key=lambda p: p.name)


def _make_zip_from_dir(folder: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                arc = p.relative_to(folder)
                zf.write(p, arcname=str(arc))


class Pipelines:
    """
    Cache pipelines across runs.
    """
    def __init__(self):
        self.line_pass: Optional[LinePass] = None
        self.paint_pass: Optional[PaintPass] = None
        self.line_preset_path: Optional[str] = None
        self.paint_preset_path: Optional[str] = None
        self.device: str = "cuda"

    def build(self, line_preset_path: str, paint_preset_path: str, device: str) -> Tuple[dict, dict]:
        self.device = device
        line_preset = load_json(Path(line_preset_path))
        paint_preset = load_json(Path(paint_preset_path))

        # rebuild only if preset path changed or not built
        if (self.line_pass is None) or (self.line_preset_path != line_preset_path) or (device != self.device):
            self.line_pass = LinePass.build(
                model_id=line_preset["model_id"],
                controlnet_id=line_preset["controlnet_id"],
                device=device
            )
            self.line_preset_path = line_preset_path

        if (self.paint_pass is None) or (self.paint_preset_path != paint_preset_path) or (device != self.device):
            self.paint_pass = PaintPass.build(
                model_id=paint_preset["model_id"],
                controlnet_id=paint_preset["controlnet_id"],
                device=device
            )
            self.paint_preset_path = paint_preset_path

        return line_preset, paint_preset


PIPES = Pipelines()


def run_single(
    rough_img,
    preset_line_path: str,
    preset_paint_path: str,
    mode: str,
    base_seed: int,
    stride: int,
    device: str,
    # UI overrides
    width: int,
    height: int,
    line_strength: float,
    line_cn: float,
    paint_strength: float,
    paint_cn: float,
):
    if rough_img is None:
        return None, None, "画像をアップロードしてください。"

    line_preset, paint_preset = PIPES.build(preset_line_path, preset_paint_path, device)

    rough = _pil_open(rough_img).convert("RGB")

    seed = seed_for_frame(int(base_seed), 0, int(stride))

    line_out = None
    color_out = None

    # LINE
    if mode in ("LINE", "BOTH"):
        line_out = PIPES.line_pass.run(
            rough_img=rough,
            prompt=line_preset["prompt"],
            negative=line_preset["negative"],
            steps=int(line_preset["steps"]),
            cfg=float(line_preset["cfg"]),
            strength=float(line_strength),
            controlnet_scale=float(line_cn),
            width=int(width),
            height=int(height),
            seed=int(seed),
        )

    # PAINT
    if mode in ("PAINT", "BOTH"):
        if line_out is None:
            # if user selected PAINT only, create a line first implicitly to lock structure
            line_tmp = PIPES.line_pass.run(
                rough_img=rough,
                prompt=line_preset["prompt"],
                negative=line_preset["negative"],
                steps=int(line_preset["steps"]),
                cfg=float(line_preset["cfg"]),
                strength=float(line_strength),
                controlnet_scale=float(line_cn),
                width=int(width),
                height=int(height),
                seed=int(seed),
            )
            line_input = line_tmp
        else:
            line_input = line_out

        color_out = PIPES.paint_pass.run(
            line_img=line_input,
            prompt=paint_preset["prompt"],
            negative=paint_preset["negative"],
            steps=int(paint_preset["steps"]),
            cfg=float(paint_preset["cfg"]),
            strength=float(paint_strength),
            controlnet_scale=float(paint_cn),
            width=int(width),
            height=int(height),
            seed=int(seed),
        )

    log = f"Done. seed={seed}  size={width}x{height}  mode={mode}"
    return line_out, color_out, log


def run_zip_sequence(
    zip_file,
    preset_line_path: str,
    preset_paint_path: str,
    mode: str,
    base_seed: int,
    stride: int,
    device: str,
    # UI overrides
    width: int,
    height: int,
    line_strength: float,
    line_cn: float,
    paint_strength: float,
    paint_cn: float,
):
    if zip_file is None:
        return None, None, None, "ZIPをアップロードしてください。"

    line_preset, paint_preset = PIPES.build(preset_line_path, preset_paint_path, device)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_dir = td / "in"
        out_line = td / "out_line"
        out_color = td / "out_color"
        ensure_dir(in_dir)
        ensure_dir(out_line)
        ensure_dir(out_color)

        zip_path = Path(zip_file.name)
        frames = _extract_zip(zip_path, in_dir)
        if not frames:
            return None, None, None, "ZIP内に画像が見つかりませんでした。png/jpg/webp を入れてください。"

        # process
        for i, fp in enumerate(frames):
            seed = seed_for_frame(int(base_seed), i, int(stride))
            rough = Image.open(fp).convert("RGB")

            stem = fp.stem
            line_path = out_line / f"{stem}.png"
            color_path = out_color / f"{stem}.png"

            line_img = None

            if mode in ("LINE", "BOTH", "PAINT"):
                # Make line if needed (PAINT also needs line to lock structure)
                if mode in ("LINE", "BOTH", "PAINT"):
                    line_img = PIPES.line_pass.run(
                        rough_img=rough,
                        prompt=line_preset["prompt"],
                        negative=line_preset["negative"],
                        steps=int(line_preset["steps"]),
                        cfg=float(line_preset["cfg"]),
                        strength=float(line_strength),
                        controlnet_scale=float(line_cn),
                        width=int(width),
                        height=int(height),
                        seed=int(seed),
                    )
                    if mode in ("LINE", "BOTH"):
                        line_img.save(line_path)

            if mode in ("PAINT", "BOTH"):
                # Use line image as both init and control
                if line_img is None:
                    line_img = Image.open(line_path)
                color = PIPES.paint_pass.run(
                    line_img=line_img,
                    prompt=paint_preset["prompt"],
                    negative=paint_preset["negative"],
                    steps=int(paint_preset["steps"]),
                    cfg=float(paint_preset["cfg"]),
                    strength=float(paint_strength),
                    controlnet_scale=float(paint_cn),
                    width=int(width),
                    height=int(height),
                    seed=int(seed),
                )
                color.save(color_path)

        # pick preview frames (first/middle/last)
        idxs = sorted({0, len(frames)//2, len(frames)-1})
        preview_rough = []
        preview_line = []
        preview_color = []

        for j in idxs:
            fp = frames[j]
            stem = fp.stem
            preview_rough.append(Image.open(fp).convert("RGB").resize((width, height), Image.LANCZOS))

            if mode in ("LINE", "BOTH"):
                lp = out_line / f"{stem}.png"
                preview_line.append(Image.open(lp).convert("RGB"))
            else:
                preview_line.append(None)

            if mode in ("PAINT", "BOTH"):
                cp = out_color / f"{stem}.png"
                preview_color.append(Image.open(cp).convert("RGB"))
            else:
                preview_color.append(None)

        # make output zip
        out_zip = td / "result.zip"
        # include both folders if exist
        pack_dir = td / "pack"
        ensure_dir(pack_dir)
        if mode in ("LINE", "BOTH"):
            (pack_dir / "line").mkdir(parents=True, exist_ok=True)
            for p in out_line.glob("*.png"):
                (pack_dir / "line" / p.name).write_bytes(p.read_bytes())
        if mode in ("PAINT", "BOTH"):
            (pack_dir / "color").mkdir(parents=True, exist_ok=True)
            for p in out_color.glob("*.png"):
                (pack_dir / "color" / p.name).write_bytes(p.read_bytes())

        _make_zip_from_dir(pack_dir, out_zip)

        log = f"Done. frames={len(frames)} mode={mode} size={width}x{height}"
        # Gradio expects file path for download
        return preview_rough, preview_line, preview_color, str(out_zip), log


with gr.Blocks(title="NanoBanana-like LinePaint (Upload)") as demo:
    gr.Markdown("## ラフ → かっこいいアニメ線画 → 画材仕上げ（アップロードUI）")

    with gr.Accordion("設定（プリセット）", open=True):
        with gr.Row():
            preset_line = gr.Textbox(label="LINEプリセット", value="presets/line_anime.json")
            preset_paint = gr.Textbox(label="PAINTプリセット", value="presets/paint_copic.json")
        with gr.Row():
            device = gr.Dropdown(label="device", choices=["cuda", "cpu"], value="cuda")
            mode = gr.Dropdown(label="モード", choices=["BOTH", "LINE", "PAINT"], value="BOTH")
        with gr.Row():
            base_seed = gr.Number(label="base seed", value=12345, precision=0)
            stride = gr.Number(label="seed stride", value=17, precision=0)

    with gr.Accordion("画質/安定性（VRAM 8GB向け）", open=False):
        with gr.Row():
            width = gr.Slider(512, 1024, value=768, step=64, label="width")
            height = gr.Slider(512, 1024, value=768, step=64, label="height")
        with gr.Row():
            line_strength = gr.Slider(0.25, 0.85, value=0.55, step=0.01, label="LINE strength（線の整理度）")
            line_cn = gr.Slider(0.5, 1.6, value=1.05, step=0.05, label="LINE ControlNet scale（線拘束）")
        with gr.Row():
            paint_strength = gr.Slider(0.25, 0.90, value=0.62, step=0.01, label="PAINT strength（塗り変化量）")
            paint_cn = gr.Slider(0.6, 2.0, value=1.25, step=0.05, label="PAINT ControlNet scale（線固定）")

    gr.Markdown("### 単体画像（1枚）")
    with gr.Row():
        rough_img = gr.Image(type="pil", label="ラフ画像をアップロード", height=320)
        with gr.Column():
            run_btn = gr.Button("生成（単体）")
            log1 = gr.Textbox(label="ログ", lines=6)

    with gr.Row():
        out_line_img = gr.Image(label="LINE結果", height=320)
        out_color_img = gr.Image(label="PAINT結果", height=320)

    run_btn.click(
        fn=run_single,
        inputs=[rough_img, preset_line, preset_paint, mode, base_seed, stride, device,
                width, height, line_strength, line_cn, paint_strength, paint_cn],
        outputs=[out_line_img, out_color_img, log1],
    )

    gr.Markdown("---\n### 連番ZIP（まとめて）")
    gr.Markdown("ZIPの中に `0001.png, 0002.png ...` のような画像を入れてアップロードしてください。")
    with gr.Row():
        zip_in = gr.File(label="連番ZIPをアップロード（.zip）", file_types=[".zip"])
        run_zip_btn = gr.Button("生成（ZIP連番）")

    with gr.Row():
        prev_rough = gr.Gallery(label="Roughプレビュー（先頭/中間/末尾）", columns=3, rows=1, height=220)
        prev_line = gr.Gallery(label="Lineプレビュー（先頭/中間/末尾）", columns=3, rows=1, height=220)
        prev_color = gr.Gallery(label="Colorプレビュー（先頭/中間/末尾）", columns=3, rows=1, height=220)

    out_zip = gr.File(label="結果ZIP（line/ と color/ を同梱）")
    log2 = gr.Textbox(label="ログ", lines=6)

    def _zip_wrapper(*args):
        pr, pl, pc, zpath, log = run_zip_sequence(*args)
        # Galleries expect list of images or list of tuples; we pass list of PIL images
        return pr, pl, pc, zpath, log

    run_zip_btn.click(
        fn=_zip_wrapper,
        inputs=[zip_in, preset_line, preset_paint, mode, base_seed, stride, device,
                width, height, line_strength, line_cn, paint_strength, paint_cn],
        outputs=[prev_rough, prev_line, prev_color, out_zip, log2],
    )


if __name__ == "__main__":
    demo.launch()

