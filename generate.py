import argparse
import functools
import sys
import time
from pathlib import Path


MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"

RGBA_PREFIX = (
    "This is an RGBA image with transparency. {prompt} "
    "The image has alpha channel and the background is transparent."
)


class PhaseTimer:
    def __init__(self, sync=None):
        self._sync = sync or (lambda: None)
        self.totals = {}
        self.counts = {}
        self._starts = {}

    def reset(self):
        self.totals.clear()
        self.counts.clear()
        self._starts.clear()

    def start(self, name):
        self._sync()
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        begin = self._starts.pop(name, None)
        if begin is None:
            return
        self._sync()
        self.add(name, time.perf_counter() - begin)

    def add(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, wall, steps, dit_gb=None):
        parts = []
        for name in ("text_encoder", "transformer", "vae.encode", "vae.decode"):
            total = self.totals.get(name)
            if not total:
                continue
            count = self.counts[name]
            if count > 1:
                parts.append(f"{name}={total:.1f}s (n={count}, avg={total / count:.2f}s)")
            else:
                parts.append(f"{name}={total:.1f}s")
        print(f"[timing] wall={wall:.1f}s steps={steps} | " + " | ".join(parts), flush=True)
        avg = self.totals.get("transformer", 0.0) / max(self.counts.get("transformer", 1), 1)
        if dit_gb and avg > 0:
            moved = 2 * dit_gb / avg
            print(f"[timing] DiT {dit_gb:.2f}GB moved up+down per step -> "
                  f"~{moved:.1f}GB/s effective PCIe", flush=True)


def _timed_method(label, fn, timer):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        timer.start(label)
        try:
            return fn(*args, **kwargs)
        finally:
            timer.stop(label)
    return wrapper


def install_timers(pipe, timer):
    for name in ("text_encoder", "transformer"):
        module = getattr(pipe, name, None)
        if module is None or not hasattr(module, "register_forward_pre_hook"):
            continue
        module.register_forward_pre_hook(lambda *_, name=name: timer.start(name), prepend=True)
        module.register_forward_hook(lambda *_, name=name: timer.stop(name))
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    for method_name in ("encode", "decode"):
        method = getattr(vae, method_name, None)
        if callable(method):
            setattr(vae, method_name, _timed_method(f"vae.{method_name}", method, timer))


def parse_args():
    p = argparse.ArgumentParser(description="Run Qwen-Image-2.1 locally (16GB/24GB VRAM friendly).")
    p.add_argument("--prompt", required=True, help="Text prompt / editing instruction.")
    p.add_argument("--image", action="append", default=None,
                   help="Input image path for editing. Repeat up to 10 times for multi-reference.")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--steps", type=int, default=40,
                   help="Inference steps (default 40). Ignored when the checkpoint ships a fixed "
                        "schedule (official Turbo = 8 steps).")
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
    p.add_argument("--batch", type=int, default=1,
                   help="Images per pipeline call (default 1). >1 batches denoising via "
                        "num_images_per_prompt: text encoding runs once and the DiT step cost "
                        "is shared, at the price of VRAM (~4-6GB extra per image at 1024).")
    p.add_argument("--output", default="output.png",
                   help="Output path; multiple images append their index and seed.")
    p.add_argument("--output-dir", default=None,
                   help="Save images in this folder, using the filename from --output.")
    p.add_argument("--transparent", action="store_true",
                   help="Wrap prompt for native RGBA (transparent) generation.")
    p.add_argument("--offload", choices=["auto", "model", "sequential", "none"], default="auto",
                   help="VRAM strategy. 'auto': >=20GB -> model (24GB), else sequential (16GB). "
                        "Ignored when --gguf is set (DiT stays resident in VRAM).")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--cpu", action="store_true", help="Force CPU (very slow, for debugging).")
    p.add_argument("--model", default=MODEL_ID,
                   help="Model id or local path (default: Qwen/Qwen-Image-2.1-Turbo).")
    p.add_argument("--gguf", default=None,
                   help="Path to a GGUF file used as the DiT (e.g. gguf\\Qwen-Image-2.1-Turbo-Q8_0.gguf). "
                        "Text encoder/VAE still load from --model; the DiT stays resident in VRAM and "
                        "--offload is ignored.")
    args = p.parse_args()
    if args.count < 1:
        p.error("--count must be at least 1")
    if args.batch < 1:
        p.error("--batch must be at least 1")
    return args


def resolve_size(args):
    if args.width and args.height:
        return args.width, args.height
    if args.image:
        return None, None
    return 2048, 2048


def patch_gguf_meta_offload():
    """accelerate rebuilds offloaded parameters via `param_cls(value, requires_grad=...)`,
    which drops GGUFParameter.quant_type and crashes its __new__ (KeyError: None).
    Fall back to the source tensor's quant_type when none is passed explicitly."""
    import torch
    from diffusers.quantizers.gguf.utils import GGUFParameter

    original_new = GGUFParameter.__new__
    if getattr(original_new, "_meta_offload_patched", False):
        return

    def __new__(cls, data, requires_grad=False, quant_type=None):
        if quant_type is None and data is not None:
            quant_type = getattr(data, "quant_type", None)
            if quant_type is None and getattr(data, "is_meta", False):
                obj = torch.Tensor._make_subclass(cls, data, requires_grad)
                obj.quant_type = None
                obj.quant_shape = None
                return obj
        return original_new(cls, data, requires_grad=requires_grad, quant_type=quant_type)

    __new__._meta_offload_patched = True
    GGUFParameter.__new__ = staticmethod(__new__)


def dequantize_gguf_non_weights(module, dtype):
    """GGUF files may quantize norm weights (and biases) too. Model code uses those
    directly, so restore them to `dtype`; quantized GGUFLinear weights stay compressed."""
    import torch
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor

    restored = 0
    for sub in module.modules():
        for param_name, param in list(sub._parameters.items()):
            if not isinstance(param, GGUFParameter):
                continue
            if isinstance(sub, GGUFLinear) and param_name == "weight":
                continue
            sub._parameters[param_name] = torch.nn.Parameter(
                dequantize_gguf_tensor(param).to(dtype), requires_grad=False
            )
            restored += 1
    return restored


def _dit_on_cuda(transformer):
    param = next(transformer.parameters(), None)
    return getattr(param, "is_cuda", False) is True


def swap_dit_to_cpu(transformer):
    """Park the resident DiT back on CPU (called before VAE decode frees up its VRAM)."""
    import torch

    if not _dit_on_cuda(transformer):
        return False
    torch.cuda.synchronize()
    start = time.perf_counter()
    transformer.to("cpu")
    torch.cuda.empty_cache()
    print(f"[gguf] DiT moved to CPU before VAE decode ({time.perf_counter() - start:.1f}s)", flush=True)
    return True


def ensure_dit_on_gpu(transformer):
    """Bring the resident DiT back to GPU after a decode-phase swap."""
    import torch

    if _dit_on_cuda(transformer):
        return False
    torch.cuda.synchronize()
    start = time.perf_counter()
    transformer.to("cuda")
    torch.cuda.synchronize()
    print(f"[gguf] DiT back on GPU ({time.perf_counter() - start:.1f}s)", flush=True)
    return True


def install_decode_swap(pipe):
    """Wrap vae.decode so a resident DiT is moved off the GPU while decoding."""
    vae = getattr(pipe, "vae", None)
    transformer = getattr(pipe, "transformer", None)
    if vae is None or transformer is None:
        return
    original = vae.decode

    @functools.wraps(original)
    def decode(*args, **kwargs):
        swap_dit_to_cpu(transformer)
        return original(*args, **kwargs)

    vae.decode = decode


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

    resident = device == "cuda" and bool(args.gguf) and args.batch == 1
    offload = args.offload
    if device == "cuda" and args.gguf:
        if not resident:
            print(f"[gguf] --batch {args.batch} > 1: DiT cannot stay resident in VRAM "
                  f"(batch activations + ~7.6GB weights exceed the GPU); using sequential offload", flush=True)
        elif offload not in ("auto", "sequential"):
            print(f"[gguf] --offload {offload} ignored: DiT stays resident on GPU, "
                  f"text encoder/VAE use sequential offload", flush=True)
        offload = "sequential"
    elif offload == "auto":
        if device == "cuda":
            free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            offload = "model" if free_gb >= 20 else "sequential"
            print(f"[auto] free VRAM {free_gb:.1f}GB -> offload={offload}", flush=True)
        else:
            offload = "none"

    print(f"[load] {args.model} dtype={args.dtype} device={device} offload={offload}", flush=True)
    load_start = time.perf_counter()
    pipeline_kwargs = {}
    if args.gguf:
        from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel

        print(f"[gguf] loading DiT from {args.gguf} (quantized weights kept in RAM)", flush=True)
        pipeline_kwargs["transformer"] = QwenImage21Transformer2DModel.from_single_file(
            args.gguf,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        restored = dequantize_gguf_non_weights(pipeline_kwargs["transformer"], dtype)
        if restored:
            print(f"[gguf] restored {restored} non-linear tensors (norms) to {args.dtype}", flush=True)
        patch_gguf_meta_offload()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=dtype, **pipeline_kwargs)
    sample_sigmas = getattr(pipe.config, "sample_sigmas", None)
    turbo = bool(sample_sigmas)
    if turbo:
        print(f"[turbo] checkpoint schedule: {len(sample_sigmas)} steps, cfg=1 "
              f"(--steps {args.steps} ignored)", flush=True)
    if args.image:
        # The pipeline already resizes references for both encoders; a second resize breaks token alignment.
        pipe.processor.image_processor.do_resize = False

    if device == "cpu":
        pipe.to("cpu")
    elif resident:
        pipe._exclude_from_cpu_offload = [*pipe._exclude_from_cpu_offload, "transformer"]
        pipe.enable_sequential_cpu_offload()
        resident_gb = sum(p.numel() * p.element_size() for p in pipe.transformer.parameters()) / 1024 ** 3
        print(f"[gguf] DiT resident on GPU ({resident_gb:.2f}GB), no per-step transfer", flush=True)
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    if offload != "none" and device == "cuda":
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()

    print(f"[timing] load+setup={time.perf_counter() - load_start:.1f}s", flush=True)
    sync = torch.cuda.synchronize if device == "cuda" else None
    timer = PhaseTimer(sync)
    install_timers(pipe, timer)
    if resident:
        install_decode_swap(pipe)
    dit_gb = None
    if offload == "sequential" and not resident:
        dit_gb = sum(p.numel() * p.element_size() for p in pipe.transformer.parameters()) / 1024 ** 3

    prompt = args.prompt
    if args.transparent:
        prompt = RGBA_PREFIX.format(prompt=prompt)

    call = {
        "prompt": prompt,
        "output_resolution": args.resolution,
        "true_cfg_scale": args.cfg,
    }
    steps = args.steps
    if turbo:
        steps = len(sample_sigmas)
    else:
        call["num_inference_steps"] = args.steps
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
    batch = min(args.batch, args.count)
    if batch != args.batch:
        print(f"[batch] clamped to {batch} (== --count)", flush=True)
    elif batch > 1:
        print(f"[batch] {batch} images per call via num_images_per_prompt", flush=True)
    for start in range(0, args.count, batch):
        chunk = min(batch, args.count - start)
        seeds = [args.seed + start + offset for offset in range(chunk)]
        if chunk > 1:
            call["num_images_per_prompt"] = chunk
            call["generator"] = [torch.Generator(device).manual_seed(seed) for seed in seeds]
        else:
            call.pop("num_images_per_prompt", None)
            call["generator"] = torch.Generator(device).manual_seed(seeds[0])
        label = str(start + 1) if chunk == 1 else f"{start + 1}-{start + chunk}"
        print(f"[run {label}/{args.count}] size={size_desc} steps={steps} "
              f"cfg={args.cfg} seeds={','.join(str(seed) for seed in seeds)}", flush=True)
        timer.reset()
        if resident:
            ensure_dit_on_gpu(pipe.transformer)
        if sync:
            sync()
        run_start = time.perf_counter()
        results = pipe(**call).images
        if sync:
            sync()
        wall = time.perf_counter() - run_start
        timer.report(wall, steps, dit_gb)
        for offset, (seed, result) in enumerate(zip(seeds, results)):
            index = start + offset
            path = output
            if args.count > 1:
                path = output.with_name(f"{output.stem}_{index + 1:02d}_seed{seed}{output.suffix}")
            result.save(path)
            print(f"[done {index + 1}/{args.count}] saved -> {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
