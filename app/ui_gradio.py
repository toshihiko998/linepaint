from __future__ import annotations
import gradio as gr
import subprocess
from pathlib import Path

def run_batch(rough_dir, out_dir, mode, preset_line, preset_paint, seed, stride, skip_existing):
    rough_dir = str(rough_dir)
    out_dir = str(out_dir)
    cmd = [
        "python", "run_batch.py",
        "--rough", rough_dir,
        "--out", out_dir,
        "--mode", mode,
        "--preset_line", preset_line,
        "--preset_paint", preset_paint,
        "--seed", str(int(seed)),
        "--stride", str(int(stride)),
    ]
    if not skip_existing:
        cmd.append("--no-skip")

    p = subprocess.run(cmd, capture_output=True, text=True)
    log = ""
    if p.stdout:
        log += p.stdout + "\n"
    if p.stderr:
        log += p.stderr + "\n"
    return log

with gr.Blocks(title="NanoBanana-like LinePaint (Batch)") as demo:
    gr.Markdown("## Line → Paint 連番バッチ（SD1.5 + ControlNet）")

    with gr.Row():
        rough_dir = gr.Textbox(label="入力 rough フォルダ", value="input/rough")
        out_dir = gr.Textbox(label="出力フォルダ", value="output")

    with gr.Row():
        mode = gr.Dropdown(label="モード", choices=["both", "line", "paint"], value="both")
        skip_existing = gr.Checkbox(label="既存出力をスキップ", value=True)

    with gr.Row():
        preset_line = gr.Textbox(label="LINEプリセット", value="presets/line_anime.json")
        preset_paint = gr.Textbox(label="PAINTプリセット", value="presets/paint_copic.json")

    with gr.Row():
        seed = gr.Number(label="base seed", value=12345, precision=0)
        stride = gr.Number(label="seed stride", value=17, precision=0)

    run_btn = gr.Button("実行")
    log = gr.Textbox(label="ログ", lines=18)

    run_btn.click(
        fn=run_batch,
        inputs=[rough_dir, out_dir, mode, preset_line, preset_paint, seed, stride, skip_existing],
        outputs=[log]
    )

if __name__ == "__main__":
    demo.launch()

