import argparse
import sys
from pathlib import Path


MODEL_ID = "Qwen/Qwen-Image-2.1"

RATIO_TO_SIZE = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

RGBA_PREFIX = (
    "This is an RGBA image with transparency. {prompt} "
    "The image has alpha channel and the background is transparent."
)


def parse_args():
    p = argparse.ArgumentParser(description="Run Qwen-Image-2.1 locally (24GB VRAM friendly).")
    p.add_argument("--prompt", required=True, help="Text prompt / editing instruction.")
    p.add_argument("--image", action="append", default=None,
                   help="Input image path for editing. Repeat up to 10 times for multi-reference.")
    p.add_argument("--ratio", choices=sorted(RATIO_TO_SIZE), default=None,
                   help="Aspect ratio preset (2K). Ignored when --width/--height given.")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--steps", type=int, default=40, help="Inference steps (default 40).")
    p.add_argument("--resolution", type=int, default=2048,
                   help="Target side length for output/condition resize (default 2048). "
                        "Used when --width/--height omitted (e.g. editing follows input aspect).")
    p.add_argument("--cfg", type=float, default=1.0,
                   help="true_cfg_scale. 1.0 = no guidance (model default); >1 needs --negative-prompt.")
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="Starting seed (default 42). Incremented for each output.")
    p.add_argument("--count", type=int, default=1,
                   help="Number of images to generate sequentially (default 1).")
    p.add_argument("--output", default="output.png",
                   help="Output path; multiple images append their index and seed.")
    p.add_argument("--output-dir", default=None,
                   help="Save images in this folder, using the filename from --output.")
    p.add_argument("--transparent", action="store_true",
                   help="Wrap prompt for native RGBA (transparent) generation.")
    p.add_argument("--offload", choices=["model", "sequential", "none"], default="model",
                   help="VRAM strategy. 'model' recommended for 24GB; 'sequential' for <12GB.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--cpu", action="store_true", help="Force CPU (very slow, for debugging).")
    p.add_argument("--model", default=MODEL_ID, help="Model id or local path.")
    args = p.parse_args()
    if args.count < 1:
        p.error("--count must be at least 1")
    return args


def resolve_size(args):
    if args.width and args.height:
        return args.width, args.height
    if args.ratio:
        return RATIO_TO_SIZE[args.ratio]
    if args.image:
        return None, None
    return RATIO_TO_SIZE["1:1"]


def main():
    args = parse_args()

    output = Path(args.output)
    if args.output_dir is not None:
        output = Path(args.output_dir) / output.name
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import QwenImage21Pipeline
    from PIL import Image

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cpu" if args.cpu else "cuda"
    width, height = resolve_size(args)

    print(f"[load] {args.model} dtype={args.dtype} device={device} offload={args.offload}", flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=dtype)
    if args.image:
        # The pipeline already resizes references for both encoders; a second resize breaks token alignment.
        pipe.processor.image_processor.do_resize = False

    if device == "cpu":
        pipe.to("cpu")
    elif args.offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    if args.offload != "none" and device == "cuda" and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    prompt = args.prompt
    if args.transparent:
        prompt = RGBA_PREFIX.format(prompt=prompt)

    call = {
        "prompt": prompt,
        "num_inference_steps": args.steps,
        "output_resolution": args.resolution,
        "true_cfg_scale": args.cfg,
    }
    if width and height:
        call["width"] = width
        call["height"] = height
    if args.negative_prompt:
        call["negative_prompt"] = args.negative_prompt

    if args.image:
        images = [Image.open(p).convert("RGB") for p in args.image]
        call["image"] = images if len(images) > 1 else images[0]
        print(f"[edit] using {len(images)} reference image(s)", flush=True)

    size_desc = f"{width}x{height}" if (width and height) else f"auto (side={args.resolution})"
    if not args.image and args.cfg > 1.0 and not args.negative_prompt:
        print("[warn] --cfg > 1 without --negative-prompt; guidance disabled.", flush=True)
    for index in range(args.count):
        seed = args.seed + index
        call["generator"] = torch.Generator(device).manual_seed(seed)
        path = output
        if args.count > 1:
            path = output.with_name(f"{output.stem}_{index + 1:02d}_seed{seed}{output.suffix}")
        print(f"[run {index + 1}/{args.count}] size={size_desc} steps={args.steps} "
              f"cfg={args.cfg} seed={seed}", flush=True)
        result = pipe(**call).images[0]
        result.save(path)
        print(f"[done {index + 1}/{args.count}] saved -> {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
