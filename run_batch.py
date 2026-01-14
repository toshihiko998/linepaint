from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from core.sequence import (
    BatchConfig, list_images_sorted, ensure_dir,
    load_json, save_json, seed_for_frame
)
from core.line_pass import LinePass
from core.paint_pass import PaintPass

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rough", type=str, required=True, help="input rough frames folder")
    ap.add_argument("--out", type=str, required=True, help="output folder")
    ap.add_argument("--mode", type=str, default="both", choices=["line", "paint", "both"])
    ap.add_argument("--preset_line", type=str, default="presets/line_anime.json")
    ap.add_argument("--preset_paint", type=str, default="presets/paint_copic.json")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stride", type=int, default=17)
    ap.add_argument("--no-skip", action="store_true", help="do not skip existing outputs")
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = BatchConfig(
        rough_dir=Path(args.rough),
        out_dir=Path(args.out),
        preset_line=Path(args.preset_line),
        preset_paint=Path(args.preset_paint),
        mode=args.mode,
        base_seed=args.seed,
        seed_stride=args.stride,
        skip_existing=(not args.no_skip),
    )

    ensure_dir(cfg.out_dir)
    out_line = cfg.out_dir / "line"
    out_color = cfg.out_dir / "color"
    ensure_dir(out_line)
    ensure_dir(out_color)

    preset_line = load_json(cfg.preset_line)
    preset_paint = load_json(cfg.preset_paint)

    # Build pipelines once (important for speed)
    line_pass = LinePass.build(
        model_id=preset_line["model_id"],
        controlnet_id=preset_line["controlnet_id"],
        device=args.device
    )
    paint_pass = PaintPass.build(
        model_id=preset_paint["model_id"],
        controlnet_id=preset_paint["controlnet_id"],
        device=args.device
    )

    frames = list_images_sorted(cfg.rough_dir)
    if not frames:
        raise SystemExit(f"No images found in: {cfg.rough_dir}")

    # Save batch params for reproducibility
    params_path = cfg.out_dir / "params.json"
    save_json(params_path, {
        "rough_dir": str(cfg.rough_dir),
        "out_dir": str(cfg.out_dir),
        "mode": cfg.mode,
        "base_seed": cfg.base_seed,
        "seed_stride": cfg.seed_stride,
        "preset_line": preset_line,
        "preset_paint": preset_paint,
    })

    for i, fp in enumerate(tqdm(frames, desc="Processing frames")):
        stem = fp.stem
        seed = seed_for_frame(cfg.base_seed, i, cfg.seed_stride)

        rough = Image.open(fp)

        line_path = out_line / f"{stem}.png"
        color_path = out_color / f"{stem}.png"

        # PASS1: LINE
        if cfg.mode in ("line", "both"):
            if (not cfg.skip_existing) or (not line_path.exists()):
                line = line_pass.run(
                    rough_img=rough,
                    prompt=preset_line["prompt"],
                    negative=preset_line["negative"],
                    steps=preset_line["steps"],
                    cfg=preset_line["cfg"],
                    strength=preset_line["strength"],
                    controlnet_scale=preset_line["controlnet_scale"],
                    width=preset_line["width"],
                    height=preset_line["height"],
                    seed=seed,
                )
                line.save(line_path)

        # PASS2: PAINT (needs line)
        if cfg.mode in ("paint", "both"):
            if (not cfg.skip_existing) or (not color_path.exists()):
                if line_path.exists():
                    line_img = Image.open(line_path)
                else:
                    # if mode=paint and line not present, create from rough on the fly
                    tmp_line = line_pass.run(
                        rough_img=rough,
                        prompt=preset_line["prompt"],
                        negative=preset_line["negative"],
                        steps=preset_line["steps"],
                        cfg=preset_line["cfg"],
                        strength=preset_line["strength"],
                        controlnet_scale=preset_line["controlnet_scale"],
                        width=preset_line["width"],
                        height=preset_line["height"],
                        seed=seed,
                    )
                    line_img = tmp_line

                color = paint_pass.run(
                    line_img=line_img,
                    prompt=preset_paint["prompt"],
                    negative=preset_paint["negative"],
                    steps=preset_paint["steps"],
                    cfg=preset_paint["cfg"],
                    strength=preset_paint["strength"],
                    controlnet_scale=preset_paint["controlnet_scale"],
                    width=preset_paint["width"],
                    height=preset_paint["height"],
                    seed=seed,
                )
                color.save(color_path)

    print("Done.")
    print(f"LINE : {out_line}")
    print(f"COLOR: {out_color}")

if __name__ == "__main__":
    main()

