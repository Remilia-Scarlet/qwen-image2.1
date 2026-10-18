import contextlib
import dataclasses
import inspect
import json
import os
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

import generate
import inference
from inference import GenerationCancelled, GenerationRequest, InferenceEngine


def local_source(root, *, transformer=True, sigmas=None):
    source = root / "checkpoint"
    source.mkdir(exist_ok=True)
    index = {"_class_name": "QwenImage21Pipeline", "sample_sigmas": [1 - i / 9 for i in range(8)]
             if sigmas is None else sigmas}
    for name in ("text_encoder", "vae", "transformer", "scheduler", "processor"):
        index[name] = ["diffusers", "Component"]
        folder = source / name
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        if name in ("text_encoder", "vae") or name == "transformer" and transformer:
            (folder / "model.safetensors").write_bytes(b"small fixture, never loaded")
    (source / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
    (source / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    image_config = {"image_processor_type": "Qwen2VLImageProcessorFast", "patch_size": 16, "merge_size": 2}
    processor_config = {"processor_class": "Qwen3VLProcessor", "image_processor": image_config,
                        "video_processor": {"video_processor_type": "Qwen3VLVideoProcessor"}}
    (source / "processor" / "processor_config.json").write_text(json.dumps(processor_config), encoding="utf-8")
    (source / "processor" / "preprocessor_config.json").write_text(json.dumps(image_config), encoding="utf-8")
    for name in ("tokenizer_config.json", "tokenizer.json"):
        (source / "processor" / name).write_text("{}", encoding="utf-8")
    return source


class RequestTests(unittest.TestCase):
    def test_round_trip_and_frozen_annotations(self):
        request = GenerationRequest.from_dict({"prompt": "  cat  ", "reference_names": ["a.png"], "mode": "edit"})
        self.assertEqual(request.prompt, "cat")
        self.assertIsNone(request.width)
        self.assertEqual(request.resolution, 1024)
        self.assertEqual(request.reference_names, ("a.png",))
        self.assertEqual(GenerationRequest.from_dict(json.loads(json.dumps(request.to_dict()))), request)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            request.count = 2

    def test_strict_invalid_fields(self):
        cases = {"prompt": [None, "", "  ", 1, False], "model": ["base", None, [], True],
                 "encoder_strategy": ["auto", None, 1], "dit_strategy": ["model", None, []],
                 "batch": [0, 9, True, 1.0, "2"], "count": [0, 65, False, 2.0],
                 "seed": [-1, 2147483648, True, 3.0], "resolution": [0, 31, 4097, 1e30, True, "32"],
                 "transparent": [0, 1, "true", None], "steps": [7, 9, True, 8.0],
                 "mode": [0, []], "reference_names": [None, "a", [1]]}
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    GenerationRequest.from_dict({"prompt": "test", field: value})
        for config in (None, [], {}, {"prompt": "test", "ratio": "1:1"},
                       {"prompt": "test", "output_dir": "../escape"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                GenerationRequest.from_dict(config)

    def test_paired_dimensions_integer_ranges_and_seed_overflow(self):
        for value in (True, 32.0, "32", 31, 4097, -1):
            for field in ("width", "height"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    GenerationRequest.from_dict({"prompt": "test", "width": 32, "height": 32, field: value})
        with self.assertRaisesRegex(ValueError, "together"):
            GenerationRequest("test", width=32)
        with self.assertRaisesRegex(ValueError, r"seed \+ count"):
            GenerationRequest("test", seed=2147483647, count=2)
        for value in (32, 33, 2049, 4096):
            self.assertEqual(GenerationRequest("test", width=value, height=value).width, value)
        GenerationRequest("test", seed=2147483584, count=64)


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def test_missing_cache_never_loads_or_downloads(self):
        engine = InferenceEngine(self.root, cpu=True)
        with patch.object(engine, "_load_pipeline", side_effect=AssertionError("must not load")):
            capabilities = engine.capabilities()
            self.assertEqual(capabilities["device"], "cpu")
            self.assertIsNone(capabilities["gpu"])
            self.assertEqual(capabilities["residency"], "phase")
            self.assertFalse(any(model["available"] for model in capabilities["models"]))
            self.assertIn("download_turbo.bat", capabilities["models"][0]["reason"])
            self.assertIn("download_gguf.bat", capabilities["models"][1]["reason"])
            with self.assertRaisesRegex(ValueError, "No automatic downloads"):
                engine.generate(GenerationRequest("cat"), [], self.root / "out",
                                cancel_event=threading.Event(), on_event=lambda _: None)
        self.assertEqual(os.environ["HF_HOME"], str(self.root / ".hf"))

    def test_q8_without_bf16_transformer_and_broken_shards(self):
        source = local_source(self.root, transformer=False)
        gguf = self.root / "test.gguf"
        gguf.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + b"fixture")
        engine = InferenceEngine(self.root, source, gguf, cpu=True)
        models = engine.capabilities()["models"]
        self.assertFalse(models[0]["available"])
        self.assertTrue(models[1]["available"])
        encoder = source / "text_encoder"
        (encoder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}), encoding="utf-8")
        (encoder / "part1.safetensors").write_bytes(b"x")
        self.assertFalse(engine.capabilities()["models"][1]["available"])
        (encoder / "part2.safetensors").write_bytes(b"x")
        self.assertTrue(engine.capabilities()["models"][1]["available"])
        (encoder / "part2.safetensors").write_bytes(b"")
        self.assertFalse(engine.capabilities()["models"][1]["available"])

    def test_truncated_gguf_rejected_without_reading_weight_tensors(self):
        source = local_source(self.root)
        gguf = self.root / "test.gguf"
        engine = InferenceEngine(self.root, source, gguf, cpu=True)
        for contents in (b"GGUF", b"x" * 100, struct.pack("<4sIQQ", b"GGUF", 3, 0, 1) + b"x"):
            gguf.write_bytes(contents)
            self.assertFalse(engine.capabilities()["models"][1]["available"])

    def test_base_schedule_and_bad_component_metadata_rejected(self):
        source = local_source(self.root)
        engine = InferenceEngine(self.root, source, cpu=True)
        self.assertTrue(engine.capabilities()["models"][0]["available"])
        for sigmas in (None, [], [1] * 7, [1] * 9, [True] * 8, [float("nan")] * 8, [0] * 8):
            index = json.loads((source / "model_index.json").read_text(encoding="utf-8"))
            index["sample_sigmas"] = sigmas
            (source / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
            self.assertFalse(engine.capabilities()["models"][0]["available"])
        source = local_source(self.root)
        (source / "processor" / "tokenizer.json").unlink()
        self.assertFalse(engine.capabilities()["models"][0]["available"])

    def test_processor_config_is_optional_in_checkpoint_layout(self):
        source = local_source(self.root)
        (source / "processor" / "processor_config.json").unlink()
        engine = InferenceEngine(self.root, source, cpu=True)
        self.assertTrue(engine.capabilities()["models"][0]["available"])

    def test_official_turbo_consolidated_processor_layout_available_for_both_models(self):
        source = local_source(self.root)
        processor = source / "processor"
        (processor / "preprocessor_config.json").unlink()
        (processor / "config.json").unlink()
        (processor / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        self.assertEqual({path.name for path in processor.iterdir()}, {
            "chat_template.jinja", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
        })
        gguf = self.root / "test.gguf"
        gguf.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + b"fixture")
        engine = InferenceEngine(self.root, source, gguf, cpu=True)
        with patch.object(engine, "_load_pipeline", side_effect=AssertionError("must not load weights")):
            for model in engine.capabilities()["models"]:
                self.assertTrue(model["available"], model["reason"])
                self.assertEqual(model["reason"], "")
            (processor / "tokenizer_config.json").unlink()
            self.assertFalse(any(model["available"] for model in engine.capabilities()["models"]))

    def test_invalid_consolidated_processor_does_not_fall_back_to_legacy(self):
        source = local_source(self.root)
        engine = InferenceEngine(self.root, source, cpu=True)
        for config in ({}, {"processor_class": "Qwen3VLProcessor"},
                       {"processor_class": "Qwen3VLProcessor", "image_processor": {}},
                       {"processor_class": "Qwen3VLProcessor", "image_processor": []},
                       {"processor_class": "OtherProcessor", "image_processor": {"patch_size": 16}}):
            with self.subTest(config=config):
                (source / "processor" / "processor_config.json").write_text(json.dumps(config), encoding="utf-8")
                model = engine.capabilities()["models"][0]
                self.assertFalse(model["available"])
                self.assertIn("processor_config.json", model["reason"])

    def test_only_official_turbo_snapshot_is_resolved(self):
        source = local_source(self.root)
        base = self.root / ".hf" / "hub" / "models--Qwen--Qwen-Image-2.1" / "snapshots" / "base"
        base.parent.mkdir(parents=True)
        source.rename(base)
        engine = InferenceEngine(self.root, cpu=True)
        self.assertFalse(engine.capabilities()["models"][0]["available"])
        turbo = self.root / ".hf" / "hub" / "models--Qwen--Qwen-Image-2.1-Turbo" / "snapshots" / "official"
        turbo.parent.mkdir(parents=True)
        base.rename(turbo)
        self.assertTrue(engine.capabilities()["models"][0]["available"])

    def test_no_implicit_cpu_fallback(self):
        engine = InferenceEngine(self.root, cpu=False)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(engine.capabilities()["device"], "unavailable")
            with self.assertRaisesRegex(RuntimeError, "CPU explicitly"):
                engine.generate(GenerationRequest("cat"), [], self.root / "out",
                                cancel_event=threading.Event(), on_event=lambda _: None)


class FakeModule(torch.nn.Module):
    def __init__(self, name, trace):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.name, self.trace = name, trace
        self.placement = "cpu"
        self.sequential = False

    def to(self, device):
        if any(param.is_meta for param in self.parameters()):
            raise AssertionError("whole meta module must not be moved")
        self.trace.append((self.name, "to", str(device)))
        self.placement = str(device)
        return self

    def forward(self, *args, **kwargs):
        self.trace.append((self.name, "forward", self.placement, self.sequential))
        if self.name == "transformer":
            assert self.engine._device == "cpu" or self.placement == "cuda" or self.sequential
        return torch.ones(1, 3, 2)


class FakeVAE(FakeModule):
    def enable_slicing(self):
        self.slicing = True

    def disable_tiling(self):
        self.tiling = False

    def encode(self, image):
        self.trace.append(("vae", "encode", image.mode))
        if hasattr(self, "_hf_hook"):
            self._hf_hook.pre_forward(self)
        return torch.ones(1)

    def decode(self, *args, **kwargs):
        self.trace.append(("vae", "decode"))
        if hasattr(self, "_hf_hook"):
            self._hf_hook.pre_forward(self)
        return torch.ones(1)


class FakeBasePipeline:
    def encode_prompt(self, prompt, image=None, device=None, num_images_per_prompt=1,
                      prompt_embeds=None, prompt_embeds_mask=None, image_pad_mask=None):
        from diffusers import QwenImage21Pipeline
        # Use the exact installed expansion/mask logic, with only device transfer mocked.
        return QwenImage21Pipeline.encode_prompt(
            self, prompt, image, torch.device("cpu"), num_images_per_prompt,
            prompt_embeds, prompt_embeds_mask, image_pad_mask,
        )

    def _get_qwen_prompt_embeds(self, prompt, image, device):
        self.encode_count += 1
        self.trace.append(("prompt", "encode", tuple(i.mode for i in image or [])))
        assert not torch.is_grad_enabled()
        self.text_encoder()
        return torch.arange(6.0).reshape(1, 3, 2), torch.tensor([[1, 1, 0]]), torch.tensor([[False, bool(image), False]])

    def set_progress_bar_config(self, **kwargs):
        assert kwargs == {"disable": True}

    def __call__(self, *, prompt, output_resolution, true_cfg_scale, num_images_per_prompt,
                 generator, callback_on_step_end, callback_on_step_end_tensor_inputs, use_kv_cache,
                 image=None, width=None, height=None):
        assert true_cfg_scale == 1.0
        assert callback_on_step_end_tensor_inputs == []
        assert use_kv_cache is True
        self.calls.append({"batch": num_images_per_prompt, "generator": generator, "width": width, "height": height,
                           "resolution": output_resolution, "prompt": prompt, "image": image})
        # Deliberately resizes here, not in the engine, as the upstream call does.
        refs = [i.resize((32, 32)) for i in image] if image else None
        triple = self.encode_prompt(prompt, refs, torch.device("cpu"), num_images_per_prompt)
        self.triples.append(triple)
        assert triple[0].shape[0] == num_images_per_prompt
        assert triple[2].shape == (1, 3)
        if image:
            assert triple[2][0, 1].item()
        for ref in refs or []:
            self.vae.encode(ref)
        for index in range(8):
            self.transformer()
            result = callback_on_step_end(self, index, torch.tensor(index), {})
            assert result == {}
            self.callback_returns.append(result)
        self.vae.decode(torch.ones(1), return_dict=False)
        w, h = (width or output_resolution) // 32 * 32, (height or output_resolution) // 32 * 32
        if self.output_image is not None:
            return SimpleNamespace(images=[self.output_image.copy() for _ in range(num_images_per_prompt)])
        return SimpleNamespace(images=[Image.new("RGBA", (w, h), (10, 20, 30, 80))
                                       for _ in range(num_images_per_prompt)])


class FakePipeline(inference._PromptCacheMixin, FakeBasePipeline):
    def __init__(self, trace, sigmas=None):
        self.trace = trace
        self.config = {"sample_sigmas": [1 - i / 9 for i in range(8)] if sigmas is None else sigmas}
        self.text_encoder = FakeModule("text_encoder", trace)
        self.transformer = FakeModule("transformer", trace)
        self.vae = FakeVAE("vae", trace)
        self.processor = SimpleNamespace(image_processor=SimpleNamespace(do_resize=True))
        self.encode_count = 0
        self.output_image = None
        self.calls, self.triples, self.callback_returns = [], [], []


class EngineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.source = local_source(self.root)
        self.gguf = self.root / "fixture.gguf"
        self.gguf.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + b"fixture")
        self.engine = InferenceEngine(self.root, self.source, self.gguf, cpu=True)
        self.trace, self.pipes, self.events = [], [], []
        self.cancel = threading.Event()
        self.load = patch.object(self.engine, "_load_pipeline", side_effect=self.load_fake).start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self.engine.close)
        self.offload = patch("accelerate.cpu_offload", side_effect=self.offload_fake).start()
        self.remove = patch("accelerate.hooks.remove_hook_from_module", side_effect=self.remove_fake).start()
        patch.object(self.engine, "_sync").start()
        patch.object(self.engine, "_empty_cache").start()
        patch("torch.cuda.mem_get_info", return_value=(100 * 1024 ** 3, 100 * 1024 ** 3)).start()

    def load_fake(self, source, model, device):
        self.assertEqual(source, self.source)
        pipe = FakePipeline(self.trace)
        pipe.transformer.engine = self.engine
        self.pipes.append(pipe)
        return pipe

    def offload_fake(self, module, *, execution_device, offload_buffers, preload_module_classes):
        self.assertEqual(str(execution_device), "cuda")
        self.assertTrue(offload_buffers)
        self.assertEqual(preload_module_classes, ["FakeVAE"] if module.name == "vae" else None)
        module.sequential = True
        module._hf_hook = Mock()
        self.trace.append((module.name, "offload"))
        return module

    def remove_fake(self, module, *, recurse):
        self.assertTrue(recurse)
        module.sequential = False
        if hasattr(module, "_hf_hook"):
            del module._hf_hook
        return module

    def run_job(self, request=None, refs=None, directory="out", on_event=None):
        return self.engine.generate(request or GenerationRequest("cat", width=32, height=32), refs or [],
                                    self.root / directory, cancel_event=self.cancel,
                                    on_event=on_event or self.events.append)

    def use_cuda(self):
        patch.object(self.engine, "_device_info", return_value=("cuda", None)).start()
        # Only generators are CPU-backed; no CUDA tensor allocation or model loading.
        original_generator = torch.Generator
        patch("torch.Generator", side_effect=lambda _: original_generator("cpu")).start()

    def test_prompt_triple_cached_once_across_remainder_and_reset_per_job(self):
        request = GenerationRequest("cat", batch=2, count=3, width=33, height=65, transparent=True)
        ref = Image.new("RGBA", (64, 64), (1, 2, 3, 12))
        outputs = self.run_job(request, [ref])
        pipe = self.pipes[0]
        self.assertEqual(pipe.encode_count, 1)
        self.assertEqual([call["batch"] for call in pipe.calls], [2, 1])
        self.assertEqual([triple[0].shape for triple in pipe.triples], [(2, 3, 2), (1, 3, 2)])
        self.assertEqual(pipe.triples[1][2].tolist(), [[False, True, False]])
        self.assertIsNone(self.engine._prompt_cache)
        self.assertEqual(len(pipe.callback_returns), 16)
        self.assertFalse(pipe.processor.image_processor.do_resize)
        self.assertTrue(pipe.vae.slicing)
        self.assertFalse(pipe.vae.tiling)
        emitted = [event["output"] for event in self.events if "output" in event]
        self.assertEqual(emitted, outputs)
        self.assertEqual([r["seed"] for r in outputs], [42, 43, 44])
        for index, record in enumerate(outputs):
            self.assertEqual(record["id"], f"{index + 1:04d}")
            self.assertTrue(Path(record["path"]).is_absolute())
            with Image.open(record["path"]) as image:
                self.assertEqual(image.size, (32, 64))
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getpixel((0, 0))[3], 80)
                metadata = json.loads(image.info["qwen_generation"])
                self.assertEqual(metadata["config"], request.to_dict())
                self.assertEqual(metadata["reference_count"], 1)
                self.assertNotIn("path", metadata["output"])
        json.dumps(self.events)
        self.run_job(request, [ref], directory="second")
        self.assertEqual(self.load.call_count, 1)
        self.assertEqual(pipe.encode_count, 2)
        self.assertEqual(len(pipe.calls), 4)

    def test_native_rgba_without_transparent_request_preserves_alpha_and_reports_rgba(self):
        request = GenerationRequest("cat", width=32, height=32, transparent=False)
        records = self.run_job(request)
        self.assertEqual(self.pipes[0].calls[0]["prompt"], "cat")
        self.assertEqual(records[0]["mode"], "RGBA")
        self.assertEqual([event["output"] for event in self.events if "output" in event], records)
        with Image.open(records[0]["path"]) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getpixel((0, 0)), (10, 20, 30, 80))
            metadata = json.loads(image.info["qwen_generation"])
            self.assertEqual(metadata["output"]["mode"], "RGBA")
            self.assertFalse(metadata["config"]["transparent"])

    def test_native_rgb_with_transparent_request_stays_rgb_and_prompt_is_wrapped(self):
        pipe = self.load_fake(self.source, "turbo", "cpu")
        pipe.output_image = Image.new("RGB", (32, 32), (10, 20, 30))
        self.load.side_effect = None
        self.load.return_value = pipe
        request = GenerationRequest("cat", width=32, height=32, transparent=True)
        records = self.run_job(request)
        self.assertEqual(pipe.calls[0]["prompt"], generate.RGBA_PREFIX.format(prompt="cat"))
        self.assertEqual(records[0]["mode"], "RGB")
        self.assertEqual([event["output"] for event in self.events if "output" in event], records)
        with Image.open(records[0]["path"]) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
            self.assertNotIn("A", image.getbands())
            metadata = json.loads(image.info["qwen_generation"])
            self.assertEqual(metadata["output"]["mode"], "RGB")
            self.assertTrue(metadata["config"]["transparent"])

    def test_palette_transparency_is_preserved_when_normalizing_png(self):
        pipe = self.load_fake(self.source, "turbo", "cpu")
        self.load.side_effect = None
        self.load.return_value = pipe
        palette = Image.new("P", (32, 32))
        palette.putpalette([10, 20, 30] * 256)
        for transparency, alpha in ((0, 0), (bytes([80] + [255] * 255), 80), (None, None)):
            with self.subTest(transparency=transparency):
                pipe.output_image = palette.copy()
                if transparency is not None:
                    pipe.output_image.info["transparency"] = transparency
                records = self.run_job(directory=f"palette-{alpha}")
                mode = "RGBA" if alpha is not None else "RGB"
                self.assertEqual(records[0]["mode"], mode)
                with Image.open(records[0]["path"]) as image:
                    self.assertEqual(image.mode, mode)
                    self.assertEqual(image.getpixel((0, 0)), (10, 20, 30, alpha) if alpha is not None else (10, 20, 30))

    def test_all_independent_strategies_phase_order_and_q8_batch(self):
        self.use_cuda()
        for model in ("turbo", "q8"):
            for encoder in ("resident", "sequential"):
                for dit in ("resident", "sequential"):
                    with self.subTest(model=model, encoder=encoder, dit=dit):
                        self.trace.clear()
                        request = GenerationRequest("cat", model=model, encoder_strategy=encoder,
                                                    dit_strategy=dit, batch=2, count=3, width=32, height=32)
                        self.run_job(request, [Image.new("RGB", (32, 32))], f"{model}-{encoder}-{dit}")
                        pipe = self.engine._pipe
                        self.assertEqual(pipe.text_encoder.sequential, encoder == "sequential")
                        self.assertEqual(pipe.transformer.sequential, dit == "sequential")
                        self.assertEqual(pipe.transformer.placement, "cpu")
                        self.assertEqual(pipe.text_encoder.placement, "cpu")
                        trace = self.trace
                        vae_indices = [i for i, item in enumerate(trace) if item[:2] == ("vae", "encode")]
                        decode_indices = [i for i, item in enumerate(trace) if item[:2] == ("vae", "decode")]
                        if encoder == "resident":
                            encoder_park = trace.index(("text_encoder", "to", "cpu"),
                                                       trace.index(("text_encoder", "to", "cuda")))
                            self.assertLess(encoder_park, vae_indices[0])
                        if dit == "resident":
                            moves = [i for i, item in enumerate(trace) if item == ("transformer", "to", "cuda")]
                            self.assertEqual(len(moves), 2)
                            for ref_index, move, decode in zip(vae_indices, moves, decode_indices):
                                self.assertLess(ref_index, move)
                                park = trace.index(("transformer", "to", "cpu"), move)
                                self.assertLess(park, decode)
                        self.assertEqual(pipe.vae._hf_hook.post_forward.call_count,
                                         pipe.vae._hf_hook.pre_forward.call_count)
        self.assertEqual(self.load.call_count, 2)

    def test_batch_clamped_before_admission_and_no_silent_fallback(self):
        self.use_cuda()
        # A batch of 8 at 1024 would fail this guard; clamped batch 1 succeeds.
        patch("torch.cuda.mem_get_info", return_value=(7 * 1024 ** 3, 24 * 1024 ** 3)).start()
        result = self.run_job(GenerationRequest("cat", batch=8, count=1, width=1024, height=1024))
        self.assertEqual(result[0]["effective"]["batch"], 1)
        self.trace.clear()
        result = self.run_job(GenerationRequest("cat", model="q8", batch=2, count=2, width=1024, height=1024),
                              directory="low-memory")
        self.assertEqual(len(result), 2)
        self.assertIn(("transformer", "to", "cuda"), self.trace)
        self.assertFalse(self.pipes[-1].transformer.sequential)
        self.assertEqual(self.engine._strategies["transformer"], "resident")
        self.assertEqual(result[0]["effective"]["dit_strategy"], "resident")
        notices = [event for event in self.events
                   if event.get("phase") == "denoising" and "Insufficient VRAM" in event.get("detail", "")]
        self.assertTrue(notices)

    def test_resident_admission_reclaims_cached_but_not_live_memory(self):
        self.use_cuda()
        self.engine._device = "cuda"
        self.engine._pipe = self.load_fake(self.source, "q8", "cuda")
        self.engine._strategies = {"transformer": "resident"}
        self.engine._job = {"cancel": self.cancel, "effective": {"batch": 2}, "area": 1024 ** 2}
        free = [int(14.43 * 1024 ** 3)]
        patch("torch.cuda.mem_get_info", side_effect=lambda: (free[0], 24 * 1024 ** 3)).start()
        tensor = SimpleNamespace(numel=lambda: int(7.07 * 1024 ** 3), element_size=lambda: 1, is_meta=False)
        self.engine._pipe.transformer.parameters = Mock(return_value=[tensor])
        self.engine._empty_cache.side_effect = lambda: free.__setitem__(0, int(21.5 * 1024 ** 3))
        self.engine._activate("transformer")
        self.assertIn("transformer", self.engine._resident_active)
        self.engine._park("transformer")
        free[0] = int(14.43 * 1024 ** 3)
        self.engine._empty_cache.side_effect = None
        notice = self.engine._activate("transformer")
        self.assertIn("Insufficient VRAM", notice)
        self.assertIn("transformer", self.engine._resident_active)

    def test_resident_two_batches_release_decode_cache_and_keep_all_outputs(self):
        self.use_cuda()
        gib = 1024 ** 3
        weights = int(7.07 * gib)
        memory = {"free": int(21.5 * gib), "cached": 0}
        pipe = self.load_fake(self.source, "q8", "cuda")
        tensor = SimpleNamespace(numel=lambda: weights, element_size=lambda: 1, is_meta=False)
        pipe.transformer.parameters = Mock(return_value=[tensor])
        original_to, original_decode = pipe.transformer.to, pipe.vae.decode

        def move(device):
            previous = pipe.transformer.placement
            result = original_to(device)
            if previous != str(device):
                memory["free"] += -weights if str(device) == "cuda" else weights
            return result

        def decode(*args, **kwargs):
            result = original_decode(*args, **kwargs)
            memory["free"] -= weights
            memory["cached"] += weights
            return result

        def empty_cache():
            memory["free"] += memory["cached"]
            memory["cached"] = 0

        pipe.transformer.to = move
        pipe.vae.decode = decode
        self.load.side_effect = None
        self.load.return_value = pipe
        self.engine._empty_cache.side_effect = empty_cache
        patch("torch.cuda.mem_get_info", side_effect=lambda: (memory["free"], 24 * gib)).start()
        records = self.run_job(GenerationRequest("cat", model="q8", batch=2, count=4,
                                               width=1024, height=1024))
        self.assertEqual([record["seed"] for record in records], [42, 43, 44, 45])
        self.assertEqual([call["batch"] for call in pipe.calls], [2, 2])
        self.assertEqual(pipe.encode_count, 1)
        self.assertEqual(memory["cached"], 0)
        self.assertEqual(memory["free"], int(21.5 * gib))
        self.assertEqual(pipe.transformer.placement, "cpu")

    def test_oom_evicts_and_next_job_reloads(self):
        self.run_job()
        self.pipes[0].transformer.forward = Mock(side_effect=torch.OutOfMemoryError("synthetic OOM"))
        with self.assertRaisesRegex(RuntimeError, "no placement strategy was silently changed"):
            self.run_job(directory="oom")
        self.assertIsNone(self.engine._pipe)
        self.run_job(directory="retry")
        self.assertEqual(self.load.call_count, 2)

    def test_failed_forward_detaches_hooks_without_moving_meta_module(self):
        self.use_cuda()
        self.run_job(GenerationRequest("cat", dit_strategy="sequential", width=32, height=32))
        pipe = self.engine._pipe
        pipe.transformer.weight = torch.nn.Parameter(torch.ones(2, device="meta"), requires_grad=False)
        pipe.transformer.forward = Mock(side_effect=RuntimeError("synthetic forward failure"))
        moves_before = len([t for t in self.trace if t[:2] == ("transformer", "to")])
        with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
            self.run_job(GenerationRequest("cat", dit_strategy="sequential", width=32, height=32), directory="failed")
        self.assertIsNone(self.engine._pipe)
        self.assertFalse(hasattr(pipe.transformer, "_hf_hook"))
        self.assertEqual(len([t for t in self.trace if t[:2] == ("transformer", "to")]), moves_before)
        self.run_job(directory="retry")
        self.assertEqual(self.load.call_count, 2)

    def test_cancel_exceptions_do_not_retain_pipeline_traceback_frames(self):
        def on_event(event):
            if event.get("phase") == "denoising" and event.get("step") == 8:
                self.cancel.set()
        try:
            self.run_job(on_event=on_event)
        except GenerationCancelled as error:
            traceback = error.__traceback__
            names = []
            while traceback is not None:
                names.append(traceback.tb_frame.f_code.co_name)
                traceback = traceback.tb_next
            self.assertNotIn("__call__", names)
            self.assertNotIn("forward", names)
            self.assertNotIn("step_callback", names)
            self.assertIsNone(error.__context__)
        else:
            self.fail("expected cancellation")

    def test_missing_runtime_schedule_fault_evicts(self):
        self.load.side_effect = lambda *args: FakePipeline(self.trace, sigmas=[])
        with self.assertRaisesRegex(ValueError, "8 positive sigmas"):
            self.run_job()
        self.assertIsNone(self.engine._pipe)

    def test_cancel_last_step_prevents_decode_and_saving(self):
        def on_event(event):
            self.events.append(event)
            if event.get("phase") == "denoising" and event.get("step") == 8:
                self.cancel.set()
        with self.assertRaises(GenerationCancelled):
            self.run_job(on_event=on_event)
        self.assertFalse(any(item[:2] == ("vae", "decode") for item in self.trace))
        self.assertFalse(list((self.root / "out").glob("*.png")))
        self.assertIsNone(self.engine._pipe)

    def test_cancel_preserves_completed_outputs_and_reloads(self):
        def on_event(event):
            self.events.append(event)
            if "output" in event:
                self.cancel.set()
        with self.assertRaises(GenerationCancelled):
            self.run_job(GenerationRequest("cat", count=3, width=32, height=32), on_event=on_event)
        records = [event["output"] for event in self.events if "output" in event]
        self.assertEqual(len(records), 1)
        self.assertTrue(Path(records[0]["path"]).is_file())
        self.assertEqual(len(self.pipes[0].calls), 1)
        self.cancel.clear()
        self.run_job(directory="retry")
        self.assertEqual(self.load.call_count, 2)

    def test_cancel_before_load_during_encoding_reference_and_decoding(self):
        self.cancel.set()
        with self.assertRaises(GenerationCancelled):
            self.run_job()
        self.assertEqual(self.load.call_count, 0)
        for phase in ("loading", "encoding", "reference_encoding", "decoding", "saving"):
            self.cancel.clear()
            def on_event(event):
                if event.get("phase") == phase:
                    self.cancel.set()
            with self.subTest(phase=phase), self.assertRaises(GenerationCancelled):
                self.run_job(refs=[Image.new("RGBA", (32, 32))], directory=phase, on_event=on_event)
            self.assertIsNone(self.engine._pipe)

    def test_dimensions_annotations_and_reference_validation(self):
        self.run_job(GenerationRequest("cat", mode="edit"), directory="text")
        self.assertEqual((self.pipes[0].calls[0]["width"], self.pipes[0].calls[0]["height"]), (2048, 2048))
        self.run_job(GenerationRequest("cat", mode="text"), [Image.new("RGB", (64, 32))], directory="edit")
        self.assertIsNone(self.pipes[0].calls[1]["width"])
        for refs in (["bad"], [Image.new("RGB", (1, 10000))], [Image.new("RGB", (32, 32))] * 11):
            with self.subTest(refs=len(refs)), self.assertRaisesRegex(ValueError, "reference"):
                self.run_job(refs=refs, directory="invalid")

    def test_output_collision_does_not_overwrite(self):
        records = self.run_job()
        path = Path(records[0]["path"])
        original = path.read_bytes()
        with self.assertRaises(RuntimeError):
            self.run_job()
        self.assertEqual(path.read_bytes(), original)

    def test_capabilities_during_job_never_mutates_pipeline(self):
        def on_event(event):
            with contextlib.ExitStack() as stack:
                for name in ("_load_pipeline", "_configure", "_park", "_discard"):
                    stack.enter_context(patch.object(self.engine, name, side_effect=AssertionError(name)))
                values = []
                thread = threading.Thread(target=lambda: values.append(self.engine.capabilities()))
                thread.start()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                self.assertTrue(values[0]["models"][0]["available"])
        self.run_job(on_event=on_event)

    def test_close_waits_for_idle_and_releases(self):
        self.run_job()
        self.engine._lock.acquire()
        closed = threading.Event()
        thread = threading.Thread(target=lambda: (self.engine.close(), closed.set()))
        thread.start()
        self.assertFalse(closed.wait(0.05))
        self.engine._lock.release()
        thread.join(timeout=5)
        self.assertTrue(closed.is_set())
        self.assertIsNone(self.engine._pipe)
        self.engine.close()


class RealSmallTests(unittest.TestCase):
    def test_actual_upstream_call_with_cpu_fake_components_and_real_hooks(self):
        from accelerate import cpu_offload
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
        from diffusers.utils.accelerate_utils import apply_forward_hook

        trace = []

        class TinyVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
                self.config = SimpleNamespace(z_dim=4, latents_mean=[0] * 4, latents_std=[1] * 4)

            @property
            def dtype(self):
                return self.weight.dtype

            def enable_slicing(self):
                pass

            def disable_tiling(self):
                pass

            @apply_forward_hook
            def encode(self, image):
                self.assert_materialized()
                trace.append(("vae.encode", image.shape[1], image.shape[-2:]))
                latents = torch.zeros((1, 4, 1, image.shape[-2] // 16, image.shape[-1] // 16))
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latents))

            @apply_forward_hook
            def decode(self, latents, return_dict):
                self.assert_materialized()
                trace.append(("vae.decode",))
                return (torch.zeros((latents.shape[0], 4, 1, latents.shape[-2] * 16, latents.shape[-1] * 16)),)

            def assert_materialized(self):
                assert not self.weight.is_meta

        class TinyTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
                self.transformer_blocks = torch.nn.ModuleList()
                self.config = SimpleNamespace(in_channels=4, causal_condition=True)

            def cache_context(self, name):
                return contextlib.nullcontext()

            def forward(self, *, hidden_states, timestep, encoder_hidden_states,
                        encoder_hidden_states_mask, img_shapes, img_mask, attention_kwargs,
                        kv_cache, kv_cache_mode, return_dict):
                assert not self.weight.is_meta
                assert not torch.is_grad_enabled()
                assert encoder_hidden_states.shape[0] == hidden_states.shape[0]
                assert img_mask[0, 1].item()
                trace.append(("transformer", hidden_states.shape[0], kv_cache_mode))
                return (torch.zeros_like(hidden_states),)

        class TinyPipeline(inference._PromptCacheMixin, QwenImage21Pipeline):
            def _get_qwen_prompt_embeds(self, prompt, image, device):
                assert image[0].mode == "RGBA"
                assert image[0].size == (32, 32)
                assert image[0].getpixel((0, 0))[3] == 77
                trace.append(("prompt",))
                self.text_encoder(torch.ones((1, 1)))
                return torch.zeros((1, 3, 2)), torch.tensor([[1, 1, 0]]), torch.tensor([[False, True, False]])

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        source = local_source(root)
        engine = InferenceEngine(root, source, cpu=True)
        self.addCleanup(engine.close)
        class TinyProcessor(SimpleNamespace):
            pass

        processor = TinyProcessor(image_processor=SimpleNamespace(do_resize=True),
                                  apply_chat_template=lambda *args, **kwargs: [[1]],
                                  tokenizer=SimpleNamespace(encode=lambda _: [7]))
        pipe = TinyPipeline(FlowMatchEulerDiscreteScheduler(), TinyVAE(), torch.nn.Linear(1, 1),
                            processor, TinyTransformer(), sample_sigmas=[1 - i / 9 for i in range(8)])
        original_configure = engine._configure

        def configure(request):
            original_configure(request)
            # Exercise real Accelerate meta/CPU behavior, without ever allocating CUDA tensors.
            for module in (pipe.text_encoder, pipe.transformer):
                if not hasattr(module, "_hf_hook"):
                    cpu_offload(module, execution_device=torch.device("cpu"), offload_buffers=True)
            if not hasattr(pipe.vae, "_hf_hook"):
                cpu_offload(pipe.vae, execution_device=torch.device("cpu"), offload_buffers=True,
                            preload_module_classes=["TinyVAE"])

        events = []
        request = GenerationRequest("cat", batch=2, count=3, width=33, height=65,
                                    resolution=32, transparent=True, dit_strategy="sequential")
        with patch.object(engine, "_load_pipeline", return_value=pipe), patch.object(engine, "_configure", side_effect=configure):
            records = engine.generate(request, [Image.new("RGBA", (64, 64), (0, 0, 0, 77))], root / "out",
                                      cancel_event=threading.Event(), on_event=events.append)
        self.assertEqual(len(records), 3)
        self.assertEqual([r["width"] for r in records], [32] * 3)
        self.assertEqual([r["height"] for r in records], [64] * 3)
        self.assertEqual(trace.count(("prompt",)), 1)
        self.assertEqual(len([t for t in trace if t[0] == "transformer"]), 16)
        self.assertEqual([t[1] for t in trace if t[0] == "transformer"], [2] * 8 + [1] * 8)
        self.assertEqual([t[2] for t in trace if t[0] == "transformer"], ["extract"] + ["cached"] * 7 +
                         ["extract"] + ["cached"] * 7)
        self.assertTrue(pipe.vae.weight.is_meta, "VAE decorator cleanup must offload after decode")
        self.assertTrue(pipe.transformer.weight.is_meta)
        self.assertTrue(next(pipe.text_encoder.parameters()).is_meta)
        for record in records:
            with Image.open(record["path"]) as image:
                self.assertEqual(image.mode, "RGBA")
        self.assertEqual([e["step"] for e in events if e.get("phase") == "denoising" and e.get("step", 0)],
                         list(range(1, 9)) * 2)
        decode_count = trace.count(("vae.decode",))
        cancelled = threading.Event()

        def cancel_at_last_step(event):
            if event.get("phase") == "denoising" and event.get("step") == 8:
                cancelled.set()

        with patch.object(engine, "_configure", side_effect=configure), self.assertRaises(GenerationCancelled):
            engine.generate(request, [Image.new("RGBA", (64, 64), (0, 0, 0, 77))], root / "cancelled",
                            cancel_event=cancelled, on_event=cancel_at_last_step)
        self.assertEqual(trace.count(("vae.decode",)), decode_count)
        self.assertFalse(pipe.vae.weight.is_meta)
        self.assertFalse(pipe.transformer.weight.is_meta)
        self.assertFalse(next(pipe.text_encoder.parameters()).is_meta)
        self.assertIsNone(engine._pipe)

    def test_gguf_patch_idempotent_and_real_accelerate_cpu_offload_round_trip(self):
        import gguf
        from accelerate import cpu_offload
        from accelerate.hooks import remove_hook_from_module
        from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter
        generate.patch_gguf_meta_offload()
        constructor = GGUFParameter.__new__
        generate.patch_gguf_meta_offload()
        self.assertIs(GGUFParameter.__new__, constructor)
        linear = GGUFLinear(32, 2, bias=False, compute_dtype=torch.float32)
        # Q8 blocks: a half-precision scale followed by 32 signed quantized bytes.
        data = torch.zeros((2, 34), dtype=torch.uint8)
        data[:, 1] = 60
        data[:, 2:] = 1
        linear.weight = GGUFParameter(data, quant_type=gguf.GGMLQuantizationType.Q8_0)
        baseline = linear(torch.ones((1, 32)))
        cpu_offload(linear, execution_device=torch.device("cpu"))
        for _ in range(2):
            torch.testing.assert_close(linear(torch.ones((1, 32))), baseline)
            self.assertTrue(linear.weight.is_meta)
        remove_hook_from_module(linear, recurse=True)
        self.assertFalse(linear.weight.is_meta)
        self.assertEqual(linear.weight.quant_type, gguf.GGMLQuantizationType.Q8_0)
        torch.testing.assert_close(linear(torch.ones((1, 32))), baseline)

    def test_adapter_constructor_signature_preserves_component_loader(self):
        from diffusers import QwenImage21Pipeline
        class Cached(inference._PromptCacheMixin, QwenImage21Pipeline):
            pass
        self.assertEqual(inspect.signature(Cached.__init__), inspect.signature(QwenImage21Pipeline.__init__))

    def test_loader_uses_only_trusted_local_config_and_local_files(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        engine = InferenceEngine(Path(temp.name), cpu=True)
        engine._job = {"cancel": threading.Event()}
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
        sentinel = Mock()
        with patch.object(QwenImage21Pipeline, "from_pretrained", return_value=sentinel) as pretrained, \
             patch.object(QwenImage21Transformer2DModel, "from_single_file", return_value=sentinel) as single, \
             patch.object(inference, "dequantize_gguf_non_weights"):
            source = Path(temp.name) / "local"
            self.assertIs(engine._load_pipeline(source, "q8", "cpu"), sentinel)
            self.assertEqual(single.call_args.kwargs["config"], str(source))
            self.assertEqual(single.call_args.kwargs["subfolder"], "transformer")
            self.assertIs(single.call_args.kwargs["local_files_only"], True)
            self.assertIs(pretrained.call_args.kwargs["local_files_only"], True)
            self.assertEqual(pretrained.call_args.args, (str(source),))
            self.assertIs(pretrained.call_args.kwargs["transformer"], sentinel)


if __name__ == "__main__":
    unittest.main()
