"""Local-only, worker-owned Qwen Image Turbo inference. No HF imports at import time."""

import functools
import gc
import json
import math
import os
import struct
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from generate import RGBA_PREFIX, dequantize_gguf_non_weights, patch_gguf_meta_offload


class GenerationCancelled(Exception):
    """Cooperative cancellation; already emitted outputs remain on disk."""


def _integer(name, value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    model: str = "turbo"
    encoder_strategy: str = "sequential"
    dit_strategy: str = "resident"
    batch: int = 1
    count: int = 1
    seed: int = 42
    width: int | None = None
    height: int | None = None
    resolution: int = 1024
    transparent: bool = False
    steps: int = 8
    mode: str | None = None
    reference_names: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        object.__setattr__(self, "prompt", self.prompt.strip())
        for name, choices in (("model", ("turbo", "q8")),
                              ("encoder_strategy", ("resident", "sequential")),
                              ("dit_strategy", ("resident", "sequential"))):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of: {', '.join(choices)}")
        for name, low, high in (("batch", 1, 8), ("count", 1, 64), ("seed", 0, 2147483647),
                                ("resolution", 32, 4096), ("steps", 8, 8)):
            _integer(name, getattr(self, name), low, high)
        if self.seed + self.count - 1 > 2147483647:
            raise ValueError("seed + count - 1 must not exceed 2147483647")
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be provided together, or both null")
        if self.width is not None:
            _integer("width", self.width, 32, 4096)
            _integer("height", self.height, 32, 4096)
        if type(self.transparent) is not bool:
            raise ValueError("transparent must be a boolean")
        if self.mode is not None and not isinstance(self.mode, str):
            raise ValueError("mode annotation must be a string or null")
        if not isinstance(self.reference_names, (list, tuple)) or not all(
            isinstance(name, str) for name in self.reference_names
        ):
            raise ValueError("reference_names must be a list of strings")
        object.__setattr__(self, "reference_names", tuple(self.reference_names))

    @classmethod
    def from_dict(cls, config: dict):
        if not isinstance(config, dict):
            raise ValueError("generation config must be an object")
        unknown = set(config) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unsupported generation fields: " + ", ".join(sorted(map(str, unknown))))
        if "prompt" not in config:
            raise ValueError("prompt must be a non-empty string")
        return cls(**config)

    def to_dict(self):
        result = asdict(self)
        result["reference_names"] = list(self.reference_names)
        return result


def _read_json(path):
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _turbo_schedule(config):
    sigmas = config.get("sample_sigmas")
    if not isinstance(sigmas, (list, tuple)) or len(sigmas) != 8 or not all(
        type(s) in (int, float) and math.isfinite(s) and s > 0 for s in sigmas
    ):
        raise ValueError("official Turbo sample_sigmas schedule (8 positive sigmas) is required")


def _present(path):
    return path.is_file() and path.stat().st_size > 0


def _weights_present(folder):
    indices = list(folder.glob("*.safetensors.index.json")) + list(folder.glob("*.bin.index.json"))
    if indices:
        for index in indices:
            mapping = _read_json(index).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                continue
            shards = set(mapping.values())
            if all(isinstance(s, str) and Path(s).name == s and _present(folder / s) for s in shards):
                return True
        return False
    return any(_present(folder / name) for name in (
        "model.safetensors", "diffusion_pytorch_model.safetensors",
        "pytorch_model.bin", "diffusion_pytorch_model.bin",
    ))


def _validate_source(source, model):
    index = _read_json(source / "model_index.json")
    if index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("model_index.json must describe QwenImage21Pipeline")
    _turbo_schedule(index)
    for name in ("text_encoder", "vae", "transformer", "scheduler", "processor"):
        component = index.get(name)
        if not isinstance(component, list) or len(component) != 2 or not all(
            isinstance(item, str) and item for item in component
        ):
            raise ValueError(f"missing {name} component in model_index.json")
    for name in ("text_encoder", "vae", "transformer"):
        folder = source / name
        _read_json(folder / "config.json")
        if (name != "transformer" or model == "turbo") and not _weights_present(folder):
            raise ValueError(f"missing or incomplete {name} weights/shards")
    _read_json(source / "scheduler" / "scheduler_config.json")
    processor = source / "processor"
    _read_json(processor / "tokenizer_config.json")
    if (processor / "processor_config.json").exists():
        config = _read_json(processor / "processor_config.json")
        if config.get("processor_class") != "Qwen3VLProcessor" or not isinstance(
            config.get("image_processor"), dict
        ) or not config["image_processor"]:
            raise ValueError("processor_config.json must describe Qwen3VLProcessor with an image_processor config")
    else:
        _read_json(processor / "preprocessor_config.json")
    if not (_present(processor / "tokenizer.json") or
            (_present(processor / "vocab.json") and _present(processor / "merges.txt"))):
        raise ValueError("missing processor tokenizer files")


class _PromptCacheMixin:
    # Inherit the upstream constructor signature for Diffusers component loading.
    @property
    def _execution_device(self):
        import torch
        return torch.device(self._engine_device)

    def encode_prompt(self, prompt, image=None, device=None, num_images_per_prompt=1,
                      prompt_embeds=None, prompt_embeds_mask=None, image_pad_mask=None):
        engine = self._inference_engine
        engine._check_cancel()
        if engine._prompt_cache is None:
            engine._emit("encoding", detail="Encoding prompt and reference vision context once per job")
            notice = engine._activate("text_encoder")
            if notice:
                engine._emit("encoding", detail=notice)
            try:
                import torch
                with torch.no_grad():
                    with engine._timed("encoding"):
                        encoded = super().encode_prompt(prompt, image, device, 1)
                    engine._prompt_cache = tuple(t.detach().cpu() if t is not None else None for t in encoded)
                del encoded
            finally:
                engine._park("text_encoder")
            engine._check_cancel()
        embeds, mask, image_mask = (
            t.to(device or self._execution_device) if t is not None else None for t in engine._prompt_cache
        )
        return super().encode_prompt(
            prompt, image, device, num_images_per_prompt,
            prompt_embeds=embeds, prompt_embeds_mask=mask, image_pad_mask=image_mask,
        )


class InferenceEngine:
    """One reusable pipeline. generate/close serialize mutation; capabilities never touches it.

    Resident weights occupy CUDA only in their active phase. Sequential components use
    per-layer Accelerate hooks. VAE uses whole-component CPU offload and slicing, no tiling.
    Upstream KV-cache tensors remain alive through decode; parking DiT does not free them.
    """

    def __init__(self, root: Path, model_path: Path | None = None,
                 gguf_path: Path | None = None, cpu: bool = False):
        self.root = Path(root).resolve()
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.gguf_path = (Path(gguf_path).resolve() if gguf_path is not None else
                          self.root / "gguf" / "Qwen-Image-2.1-Turbo-Q8_0.gguf")
        self.cpu = cpu
        os.environ["HF_HOME"] = str(self.root / ".hf")
        self._lock = threading.Lock()
        self._pipe = None
        self._model = None
        self._strategies = {}
        self._resident_active = set()
        self._prompt_cache = None
        self._job = None

    def _device_info(self):
        if self.cpu:
            return "cpu", None
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return "cuda", {"name": torch.cuda.get_device_name(),
                                "free_bytes": int(free), "total_bytes": int(total)}
        except (ImportError, RuntimeError, OSError):
            pass
        return "unavailable", None

    def _source(self, model):
        if self.model_path is not None:
            candidates = [self.model_path]
        else:
            snapshots = self.root / ".hf" / "hub" / "models--Qwen--Qwen-Image-2.1-Turbo" / "snapshots"
            candidates = sorted(snapshots.glob("*"), reverse=True)
        reason = "official Turbo checkpoint is not cached"
        for source in candidates:
            try:
                _validate_source(source, model)
                if model == "q8":
                    if not _present(self.gguf_path):
                        raise ValueError("Q8 GGUF weights are missing")
                    with self.gguf_path.open("rb") as data:
                        header = data.read(24)
                    if len(header) != 24:
                        raise ValueError("Q8 GGUF header is missing or truncated")
                    magic, version, tensors, _ = struct.unpack("<4sIQQ", header)
                    if magic != b"GGUF" or version not in (2, 3) or tensors == 0 or self.gguf_path.stat().st_size <= 24:
                        raise ValueError("Q8 weights have an invalid or incomplete GGUF header")
                return source, ""
            except (OSError, ValueError, TypeError) as error:
                reason = str(error)
        download = "download_turbo.bat" if model == "turbo" else "download_turbo.bat and download_gguf.bat"
        return None, f"{reason}. Run {download} to install local weights. No automatic downloads."

    def capabilities(self):
        device, gpu = self._device_info()
        models = []
        for model, name in (("turbo", "Official Turbo (BF16)"), ("q8", "Official Turbo (Q8 GGUF)")):
            source, reason = self._source(model)
            models.append({"id": model, "name": name, "available": source is not None, "reason": reason})
        return {"models": models, "device": device, "gpu": gpu, "steps": 8,
                "limits": {"batch": 8, "count": 64, "references": 10,
                           "width": 4096, "height": 4096, "resolution": 4096}, "residency": "phase"}

    def _load_pipeline(self, source, model, device):
        # Optional GGUF CUDA kernels can download executable code; this engine is local-only.
        os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "false"
        import torch
        from diffusers import QwenImage21Pipeline

        class CachedQwenImage21Pipeline(_PromptCacheMixin, QwenImage21Pipeline):
            pass

        kwargs = {"torch_dtype": torch.float32 if device == "cpu" else torch.bfloat16,
                  "local_files_only": True}
        if model == "q8":
            from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel
            patch_gguf_meta_offload()
            kwargs["transformer"] = QwenImage21Transformer2DModel.from_single_file(
                str(self.gguf_path), config=str(source), subfolder="transformer",
                local_files_only=True, torch_dtype=kwargs["torch_dtype"],
                quantization_config=GGUFQuantizationConfig(compute_dtype=kwargs["torch_dtype"]),
            )
            dequantize_gguf_non_weights(kwargs["transformer"], kwargs["torch_dtype"])
            self._check_cancel()
        return CachedQwenImage21Pipeline.from_pretrained(str(source), **kwargs)

    def _install_wrappers(self):
        pipe = self._pipe
        pipe._inference_engine = self
        pipe._engine_device = self._device
        pipe.processor.image_processor.do_resize = False
        pipe.vae.enable_slicing()
        pipe.vae.disable_tiling()
        pipe.set_progress_bar_config(disable=True)
        original_forward = pipe.transformer.forward

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            self._check_cancel()
            if not self._job["denoising_started"]:
                self._emit("denoising", step=0, step_started=None)
                notice = self._activate("transformer")
                if notice:
                    self._emit("denoising", detail=notice)
                self._job["denoising_started"] = True
                self._job["step_clock"] = time.perf_counter()
            self._job["step_started"] = time.time()
            self._emit("denoising", step_started=self._job["step_started"])
            with self._timed("denoising"):
                result = original_forward(*args, **kwargs)
            self._check_cancel()
            return result

        pipe.transformer.forward = forward
        for method_name, phase in (("encode", "reference_encoding"), ("decode", "decoding")):
            original = getattr(pipe.vae, method_name)

            def wrap(original=original, phase=phase):
                @functools.wraps(original)
                def method(*args, **kwargs):
                    self._check_cancel()
                    if phase == "decoding":
                        self._park("transformer")
                    if phase == "reference_encoding":
                        self._job["reference_index"] += 1
                        self._emit(phase, reference_index=self._job["reference_index"],
                                   reference_total=self._job["reference_count"])
                    else:
                        self._emit(phase)
                    try:
                        with self._timed(phase):
                            result = original(*args, **kwargs)
                    finally:
                        # Diffusers' encode/decode decorator calls pre_forward, not post_forward.
                        hook = getattr(pipe.vae, "_hf_hook", None)
                        if hook is not None:
                            hook.post_forward(pipe.vae, None)
                    self._check_cancel()
                    return result
                return method

            setattr(pipe.vae, method_name, wrap())

    def _configure(self, request):
        from accelerate import cpu_offload
        from accelerate.hooks import remove_hook_from_module
        import torch

        desired = {"text_encoder": request.encoder_strategy, "transformer": request.dit_strategy,
                   "vae": "sequential"}
        for name, strategy in desired.items():
            if self._strategies.get(name) == strategy:
                continue
            module = getattr(self._pipe, name)
            if name in self._strategies:
                remove_hook_from_module(module, recurse=True)
            module.to("cpu")
            if self._device == "cuda" and strategy == "sequential":
                cpu_offload(module, execution_device=torch.device("cuda"), offload_buffers=True,
                            preload_module_classes=[type(module).__name__] if name == "vae" else None)
            self._strategies[name] = strategy

    def _activate(self, name):
        self._check_cancel()
        if self._device != "cuda" or self._strategies[name] != "resident" or name in self._resident_active:
            return None
        import torch
        module = getattr(self._pipe, name)
        weight_bytes = sum(t.numel() * t.element_size() for t in module.parameters())
        weight_bytes += sum(t.numel() * t.element_size() for t in module.buffers())
        # Admission is a lower-bound guard, not a promise that all activations will fit.
        gib = 1024 ** 3
        headroom = 2 * gib if name == "text_encoder" else (
            2 * gib + 4 * gib * self._job["effective"]["batch"] * self._job["area"] / 1024 ** 2
        )
        # Driver free memory excludes reusable blocks held by the CUDA allocator.
        self._empty_cache()
        free, _ = torch.cuda.mem_get_info()
        notice = None
        if free < weight_bytes + headroom:
            strategy = "encoder_strategy" if name == "text_encoder" else "dit_strategy"
            notice = (f"Insufficient VRAM for resident {name}: requires at least "
                      f"{(weight_bytes + headroom) / gib:.2f} GiB, free {free / gib:.2f} GiB. "
                      f"Continuing anyway; if this fails, select {strategy}='sequential' or reduce "
                      f"batch/resolution.")
        with self._timed(f"{'encoder' if name == 'text_encoder' else 'dit'}_load"):
            module.to("cuda")
        self._resident_active.add(name)
        self._check_cancel()
        return notice

    def _park(self, name):
        if name not in self._resident_active:
            return
        with self._timed(f"{'encoder' if name == 'text_encoder' else 'dit'}_park"):
            getattr(self._pipe, name).to("cpu")
        self._resident_active.remove(name)
        self._empty_cache()

    def _empty_cache(self):
        if getattr(self, "_device", None) == "cuda":
            import torch
            torch.cuda.empty_cache()

    def _discard(self):
        pipe, self._pipe = self._pipe, None
        self._model = None
        self._strategies.clear()
        self._resident_active.clear()
        self._prompt_cache = None
        if pipe is not None:
            from accelerate.hooks import remove_hook_from_module
            for name in ("text_encoder", "transformer", "vae"):
                module = getattr(pipe, name)
                try:
                    remove_hook_from_module(module, recurse=True)
                    if not any(p.is_meta for p in module.parameters()):
                        module.to("cpu")
                except Exception:
                    # A failed Accelerate forward may leave partially materialized parameters.
                    # Never reuse this pipeline or move a whole meta module.
                    pass
            pipe._inference_engine = None
            del module
        del pipe
        gc.collect()
        self._empty_cache()

    def close(self):
        with self._lock:
            self._discard()

    def _check_cancel(self):
        if self._job is not None and self._job["cancel"].is_set():
            raise GenerationCancelled("Generation cancelled; completed outputs have been preserved")

    def _emit(self, phase, **values):
        job = self._job
        job["on_event"]({"phase": phase, "batch_index": job["batch_index"], "batches": job["batches"],
                         "steps": 8, "count": job["count"], "completed": job["completed"],
                         "timings": dict(job["timings"]), "effective": dict(job["effective"]), **values})
        self._check_cancel()

    @contextmanager
    def _timed(self, name):
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            totals = self._job["timings"]
            totals[name] = totals.get(name, 0.0) + time.perf_counter() - start

    def _sync(self):
        if self._device == "cuda":
            import torch
            torch.cuda.synchronize()

    def generate(self, request: GenerationRequest, references: list, output_dir: Path, *,
                 cancel_event: threading.Event, on_event) -> list[dict]:
        if not isinstance(request, GenerationRequest):
            raise ValueError("request must be a validated GenerationRequest")
        from PIL import Image
        if not isinstance(references, list) or len(references) > 10 or not all(
            isinstance(image, Image.Image) and image.width > 0 and image.height > 0 for image in references
        ):
            raise ValueError("references must be a list of at most 10 non-empty PIL images")
        # Match upstream rounding only to validate it; upstream still owns reference resizing.
        for image in references:
            ratio = image.width / image.height
            w = round(math.sqrt(request.resolution ** 2 * ratio) / 32) * 32
            h = round(math.sqrt(request.resolution ** 2 / ratio) / 32) * 32
            if not 32 <= w <= 4096 or not 32 <= h <= 4096:
                raise ValueError("reference aspect/resolution produces dimensions outside 32..4096; "
                                 "reduce resolution or use a less extreme aspect ratio")
        width, height = request.width, request.height
        if width is None and not references:
            width = height = 2048
        if width is None:
            ratio = references[-1].width / references[-1].height
            area = (round(math.sqrt(request.resolution ** 2 * ratio) / 32) * 32 *
                    round(math.sqrt(request.resolution ** 2 / ratio) / 32) * 32)
        else:
            area = (width // 32 * 32) * (height // 32 * 32)
        batch = min(request.batch, request.count)
        effective = {"model": request.model, "encoder_strategy": request.encoder_strategy,
                     "dit_strategy": request.dit_strategy, "batch": batch, "steps": 8,
                     "vae_policy": "CPU offload on CUDA; slicing enabled; tiling disabled", "use_kv_cache": True}
        outputs = []
        with self._lock:
            self._job = {"cancel": cancel_event, "on_event": on_event, "batch_index": 1,
                         "batches": (request.count + batch - 1) // batch, "count": request.count,
                         "completed": 0, "timings": {}, "effective": effective, "area": area,
                         "denoising_started": False, "reference_count": len(references),
                         "reference_index": 0, "step_times": [], "step_clock": None,
                         "step_started": None}
            pipe = results = None
            failure = None
            try:
                self._check_cancel()
                self._device, _ = self._device_info()
                if self._device == "unavailable":
                    raise RuntimeError("CUDA is unavailable. Enable CPU explicitly for debugging, or install a CUDA PyTorch build.")
                source, reason = self._source(request.model)
                if source is None:
                    raise ValueError(reason)
                self._emit("loading", detail="Local weights only; loading is coarse-grained and cannot interrupt a loader or kernel")
                if self._model != request.model:
                    self._discard()
                    with self._timed("loading"):
                        self._pipe = self._load_pipeline(source, request.model, self._device)
                    _turbo_schedule(dict(self._pipe.config))
                    self._model = request.model
                    self._install_wrappers()
                self._check_cancel()
                self._configure(request)
                pipe = self._pipe
                self._prompt_cache = None
                import torch
                from PIL.PngImagePlugin import PngInfo
                directory = Path(output_dir).resolve()
                directory.mkdir(parents=True, exist_ok=True)

                def step_callback(callback_pipe, index, timestep, kwargs):
                    self._check_cancel()
                    now = time.perf_counter()
                    started = self._job["step_clock"] or now
                    self._job["step_clock"] = now
                    self._job["step_times"].append(now - started)
                    self._emit("denoising", step=index + 1, step_time=now - started,
                               step_times=list(self._job["step_times"]))
                    self._check_cancel()
                    return {}

                for start in range(0, request.count, batch):
                    self._check_cancel()
                    self._job["batch_index"] = start // batch + 1
                    self._job["denoising_started"] = False
                    self._job["reference_index"] = 0
                    self._job["step_times"] = []
                    self._job["step_clock"] = None
                    chunk = min(batch, request.count - start)
                    seeds = [request.seed + start + offset for offset in range(chunk)]
                    generators = [torch.Generator(self._device).manual_seed(seed) for seed in seeds]
                    call = {"prompt": RGBA_PREFIX.format(prompt=request.prompt) if request.transparent else request.prompt,
                            "output_resolution": request.resolution, "true_cfg_scale": 1.0,
                            "num_images_per_prompt": chunk,
                            "generator": generators if chunk > 1 else generators[0],
                            "callback_on_step_end": step_callback, "callback_on_step_end_tensor_inputs": [],
                            "use_kv_cache": True}
                    if references:
                        call["image"] = references
                    if width is not None:
                        call.update(width=width, height=height)
                    with torch.no_grad():
                        results = pipe(**call).images
                    self._check_cancel()
                    if len(results) != chunk:
                        raise RuntimeError(f"Pipeline returned {len(results)} images; expected {chunk}")
                    self._emit("saving")
                    for offset, image in enumerate(results):
                        self._check_cancel()
                        index = start + offset + 1
                        filename = f"{index:04d}_seed{seeds[offset]}.png"
                        path = directory / filename
                        has_alpha = any(band.lower() == "a" for band in image.getbands()) or (
                            image.mode == "P" and "transparency" in image.info
                        )
                        record = {"id": f"{index:04d}", "filename": filename, "path": str(path),
                                  "seed": seeds[offset], "width": image.width, "height": image.height,
                                  "mode": "RGBA" if has_alpha else "RGB", "model": request.model,
                                  "effective": dict(effective)}
                        provenance = {"output": {key: value for key, value in record.items() if key != "path"},
                                      "config": request.to_dict(),
                                      "reference_count": len(references), "timings": dict(self._job["timings"])}
                        metadata = PngInfo()
                        metadata.add_text("qwen_generation", json.dumps(provenance))
                        with self._timed("saving"):
                            # Exclusive creation prevents accidental replacement if a job directory is reused.
                            with path.open("xb") as target:
                                try:
                                    image.convert(record["mode"]).save(target, format="PNG", pnginfo=metadata)
                                except Exception:
                                    target.close()
                                    path.unlink(missing_ok=True)
                                    raise
                        outputs.append(record)
                        self._job["completed"] = len(outputs)
                        on_event({"output": record})
                        self._check_cancel()
                    results = None
                    self._empty_cache()
                    self._emit("saving", detail="Batch saved")
                for name in tuple(self._resident_active):
                    self._park(name)
            except GenerationCancelled as error:
                failure = GenerationCancelled(str(error))
                self._discard()
            except Exception as error:
                import torch
                if isinstance(error, torch.OutOfMemoryError):
                    failure = RuntimeError("Out of memory. Select sequential encoder/DiT placement or reduce batch/resolution. "
                                           "The failed pipeline was evicted; no placement strategy was silently changed.")
                elif isinstance(error, ValueError):
                    failure = ValueError(str(error))
                else:
                    failure = RuntimeError(str(error))
                self._discard()
            finally:
                self._prompt_cache = None
                self._job = None
                pipe = results = None
                if self._pipe is None:
                    gc.collect()
                    self._empty_cache()
            if failure is not None:
                raise failure from None
        return outputs
