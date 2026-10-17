import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import generate


class GenerateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.torch = MagicMock()
        self.torch.Generator.return_value.manual_seed.side_effect = lambda seed: seed
        self.torch.cuda.mem_get_info.return_value = (24 * 1024 ** 3, 48 * 1024 ** 3)
        self.diffusers = MagicMock()
        self.pipe = self.diffusers.QwenImage21Pipeline.from_pretrained.return_value
        self.pipe.config.sample_sigmas = None
        self.result = MagicMock()
        self.pipe.return_value.images = [self.result]
        self.pil = MagicMock()
        gguf_utils = MagicMock()
        gguf_utils.GGUFParameter = type("GGUFParameter", (), {})
        modules = patch.dict(sys.modules, {
            "torch": self.torch, "diffusers": self.diffusers, "PIL": self.pil,
            "diffusers.quantizers": MagicMock(), "diffusers.quantizers.gguf": MagicMock(),
            "diffusers.quantizers.gguf.utils": gguf_utils,
        })
        modules.start()
        self.addCleanup(modules.stop)

    def run_main(self, *args):
        with patch.object(sys, "argv", ["generate.py", "--prompt", "test", *args]):
            with contextlib.redirect_stdout(io.StringIO()):
                generate.main()

    def test_single_image_keeps_output_path(self):
        output = self.root / "nested" / "result.png"
        self.run_main("--output", str(output))
        self.assertTrue(output.parent.is_dir())
        self.result.save.assert_called_once_with(output)
        self.pipe.assert_called_once()
        self.torch.Generator.assert_called_once_with("cuda")
        self.assertEqual(self.pipe.call_args.kwargs["generator"], 42)
        self.assertIsNot(self.pipe.processor.image_processor.do_resize, False)

    def test_ten_images_reuse_model_and_references(self):
        folder = self.root / "candidates"
        self.run_main("--image", "a.png", "--image", "b.png", "--count", "10",
                      "--output-dir", str(folder), "--width", "1024", "--height", "1024",
                      "--resolution", "1024", "--negative-prompt", "blur", "--cfg", "3")
        self.diffusers.QwenImage21Pipeline.from_pretrained.assert_called_once()
        self.pipe.enable_model_cpu_offload.assert_called_once()
        self.assertIs(self.pipe.processor.image_processor.do_resize, False)
        self.assertEqual(self.pil.Image.open.call_args_list, [call("a.png"), call("b.png")])
        self.assertTrue(folder.is_dir())
        self.assertEqual(self.result.save.call_args_list, [
            call(folder / f"output_{i + 1:02d}_seed{42 + i}.png") for i in range(10)
        ])
        self.assertEqual(self.torch.Generator.call_count, 10)
        for i, invocation in enumerate(self.pipe.call_args_list):
            kwargs = invocation.kwargs
            self.assertEqual(kwargs["generator"], 42 + i)
            self.assertEqual((kwargs["width"], kwargs["height"]), (1024, 1024))
            self.assertEqual(kwargs["output_resolution"], 1024)
            self.assertEqual(kwargs["num_inference_steps"], 40)
            self.assertEqual(kwargs["negative_prompt"], "blur")
            self.assertEqual(kwargs["true_cfg_scale"], 3)
            self.assertEqual(len(kwargs["image"]), 2)
            self.assertIs(kwargs["image"], self.pipe.call_args_list[0].kwargs["image"])
            self.assertNotIn("num_images_per_prompt", kwargs)

    def test_batch_groups_images_per_call(self):
        results = [MagicMock(), MagicMock()]
        self.pipe.return_value.images = results
        self.run_main("--count", "4", "--batch", "2", "--output", str(self.root / "b.png"))
        self.assertEqual(self.pipe.call_count, 2)
        first, second = self.pipe.call_args_list
        self.assertEqual(first.kwargs["num_images_per_prompt"], 2)
        self.assertEqual(first.kwargs["generator"], [42, 43])
        self.assertEqual(second.kwargs["num_images_per_prompt"], 2)
        self.assertEqual(second.kwargs["generator"], [44, 45])
        self.assertEqual(self.torch.Generator.call_count, 4)
        self.assertEqual(results[0].save.call_args_list, [
            call(self.root / "b_01_seed42.png"), call(self.root / "b_03_seed44.png"),
        ])
        self.assertEqual(results[1].save.call_args_list, [
            call(self.root / "b_02_seed43.png"), call(self.root / "b_04_seed45.png"),
        ])

    def test_batch_remainder_is_single_call(self):
        results = [MagicMock(), MagicMock()]
        self.pipe.return_value.images = results
        self.run_main("--count", "3", "--batch", "2", "--output", str(self.root / "c.png"))
        self.assertEqual(self.pipe.call_count, 2)
        first, second = self.pipe.call_args_list
        self.assertEqual(first.kwargs["num_images_per_prompt"], 2)
        self.assertNotIn("num_images_per_prompt", second.kwargs)
        self.assertEqual(second.kwargs["generator"], 44)
        self.assertEqual(results[0].save.call_args_list, [
            call(self.root / "c_01_seed42.png"), call(self.root / "c_03_seed44.png"),
        ])
        self.assertEqual(results[1].save.call_args_list, [call(self.root / "c_02_seed43.png")])

    def test_batch_clamped_to_count(self):
        self.run_main("--count", "1", "--batch", "4", "--output", str(self.root / "d.png"))
        self.assertEqual(self.pipe.call_count, 1)
        self.assertNotIn("num_images_per_prompt", self.pipe.call_args.kwargs)
        self.result.save.assert_called_once_with(self.root / "d.png")

    def test_small_edit_preserves_requested_output_and_reference_resolution(self):
        self.run_main("--image", "reference.jpg", "--width", "128", "--height", "128",
                      "--resolution", "128", "--count", "2", "--output-dir", str(self.root))
        self.assertIs(self.pipe.processor.image_processor.do_resize, False)
        self.assertEqual(self.pipe.call_count, 2)
        for invocation in self.pipe.call_args_list:
            kwargs = invocation.kwargs
            self.assertEqual((kwargs["width"], kwargs["height"]), (128, 128))
            self.assertEqual(kwargs["output_resolution"], 128)
            self.assertIs(kwargs["image"], self.pil.Image.open.return_value.convert.return_value)

    def test_auto_uses_model_offload_above_20gb(self):
        self.run_main("--output", str(self.root / "m.png"))
        self.pipe.enable_model_cpu_offload.assert_called_once()
        self.pipe.enable_sequential_cpu_offload.assert_not_called()

    def test_auto_uses_sequential_at_16gb(self):
        self.torch.cuda.mem_get_info.return_value = (16 * 1024 ** 3, 48 * 1024 ** 3)
        self.run_main("--output", str(self.root / "g.png"))
        self.pipe.enable_sequential_cpu_offload.assert_called_once()
        self.pipe.enable_model_cpu_offload.assert_not_called()

    def test_auto_falls_back_to_sequential_on_low_vram(self):
        self.torch.cuda.mem_get_info.return_value = (8 * 1024 ** 3, 48 * 1024 ** 3)
        self.run_main("--output", str(self.root / "s.png"))
        self.pipe.enable_sequential_cpu_offload.assert_called_once()
        self.pipe.enable_model_cpu_offload.assert_not_called()

    def test_default_model_is_official_turbo(self):
        self.run_main("--output", str(self.root / "t.png"))
        self.diffusers.QwenImage21Pipeline.from_pretrained.assert_called_once_with(
            "Qwen/Qwen-Image-2.1-Turbo", torch_dtype=self.torch.bfloat16,
        )

    def test_gguf_loads_dit_from_single_file(self):
        self.run_main("--output", str(self.root / "q.png"), "--gguf", "gguf/q8.gguf")
        self.diffusers.GGUFQuantizationConfig.assert_called_once_with(compute_dtype=self.torch.bfloat16)
        self.diffusers.QwenImage21Transformer2DModel.from_single_file.assert_called_once_with(
            "gguf/q8.gguf",
            quantization_config=self.diffusers.GGUFQuantizationConfig.return_value,
            torch_dtype=self.torch.bfloat16,
        )
        kwargs = self.diffusers.QwenImage21Pipeline.from_pretrained.call_args.kwargs
        self.assertIs(
            kwargs["transformer"],
            self.diffusers.QwenImage21Transformer2DModel.from_single_file.return_value,
        )
        self.assertEqual(kwargs["torch_dtype"], self.torch.bfloat16)

    def test_no_gguf_keeps_pipeline_call_unchanged(self):
        self.run_main("--output", str(self.root / "n.png"))
        self.diffusers.QwenImage21Transformer2DModel.from_single_file.assert_not_called()
        kwargs = self.diffusers.QwenImage21Pipeline.from_pretrained.call_args.kwargs
        self.assertNotIn("transformer", kwargs)

    def test_gguf_resident_takes_priority_over_auto_offload(self):
        self.run_main("--output", str(self.root / "a.png"), "--gguf", "gguf/q8.gguf")
        self.torch.cuda.mem_get_info.assert_not_called()
        self.pipe.enable_sequential_cpu_offload.assert_called_once()
        self.pipe.enable_model_cpu_offload.assert_not_called()

    def test_gguf_keeps_dit_resident_and_ignores_offload(self):
        seen = {}
        self.pipe.enable_sequential_cpu_offload.side_effect = (
            lambda: seen.update(excluded=list(self.pipe._exclude_from_cpu_offload))
        )
        for offload in ("none", "model", "sequential"):
            with self.subTest(offload=offload):
                self.run_main("--output", str(self.root / "r.png"),
                              "--gguf", "gguf/q8.gguf", "--offload", offload)
                self.assertIn("transformer", seen["excluded"])
        self.pipe.enable_model_cpu_offload.assert_not_called()
        self.pipe.to.assert_not_called()
        self.pipe.vae.enable_slicing.assert_called()

    def test_gguf_batch_above_one_falls_back_to_offloaded_dit(self):
        self.run_main("--output", str(self.root / "b.png"), "--gguf", "gguf/q8.gguf",
                      "--count", "2", "--batch", "2")
        self.pipe.enable_sequential_cpu_offload.assert_called_once()
        self.pipe.enable_model_cpu_offload.assert_not_called()
        self.assertNotIn("transformer", self.pipe._exclude_from_cpu_offload)
        self.pipe.to.assert_not_called()

    def test_base_2_1_fallback_still_uses_num_inference_steps(self):
        self.run_main("--model", "Qwen/Qwen-Image-2.1", "--output", str(self.root / "b.png"))
        self.diffusers.QwenImage21Pipeline.from_pretrained.assert_called_once_with(
            "Qwen/Qwen-Image-2.1", torch_dtype=self.torch.bfloat16,
        )
        kwargs = self.pipe.call_args.kwargs
        self.assertEqual(kwargs["num_inference_steps"], 40)

    def test_checkpoint_schedule_overrides_steps(self):
        self.pipe.config.sample_sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        self.run_main("--steps", "40", "--output", str(self.root / "s.png"))
        kwargs = self.pipe.call_args.kwargs
        self.assertNotIn("num_inference_steps", kwargs)

    def test_vae_slicing_enabled_for_gpu_offload(self):
        self.run_main("--output", str(self.root / "v.png"))
        self.pipe.vae.enable_slicing.assert_called_once()

    def test_vae_slicing_skipped_on_cpu(self):
        self.run_main("--output", str(self.root / "v.png"), "--cpu")
        self.pipe.vae.enable_slicing.assert_not_called()

    def test_multiple_images_use_output_parent_stem_suffix_and_starting_seed(self):
        output = self.root / "nested" / "pick.jpg"
        self.run_main("--output", str(output), "--count", "2", "--seed", "100", "--cpu")
        self.assertEqual(self.result.save.call_args_list, [
            call(output.with_name("pick_01_seed100.jpg")),
            call(output.with_name("pick_02_seed101.jpg")),
        ])
        self.assertEqual(self.torch.Generator.call_args_list, [call("cpu"), call("cpu")])
        self.pipe.to.assert_called_once_with("cpu")

    def test_output_dir_overrides_parent_but_preserves_single_filename(self):
        folder = self.root / "chosen"
        self.run_main("--output", "ignored/pick.png", "--output-dir", str(folder))
        self.result.save.assert_called_once_with(folder / "pick.png")

    def test_invalid_count_fails_before_model_loading(self):
        for count in ("0", "-1", "invalid"):
            with self.subTest(count=count), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_main("--count", count)
                self.assertEqual(error.exception.code, 2)
        self.diffusers.QwenImage21Pipeline.from_pretrained.assert_not_called()


class GgufMetaOffloadPatchTests(unittest.TestCase):
    def test_quant_type_survives_meta_recreation(self):
        import gguf
        import torch
        from diffusers.quantizers.gguf.utils import GGUFParameter

        generate.patch_gguf_meta_offload()
        param = GGUFParameter(
            torch.zeros((2, 34), dtype=torch.uint8),
            quant_type=gguf.GGMLQuantizationType.Q8_0,
        )
        meta = param.to("meta")
        self.assertIsInstance(meta, GGUFParameter)
        self.assertEqual(meta.quant_type, gguf.GGMLQuantizationType.Q8_0)
        recreated = GGUFParameter(meta, requires_grad=False)
        self.assertEqual(recreated.quant_type, gguf.GGMLQuantizationType.Q8_0)
        self.assertEqual(recreated.quant_shape, meta.quant_shape)


class _FakeParam:
    def __init__(self, is_cuda):
        self.is_cuda = is_cuda


class _FakeDiT:
    def __init__(self, is_cuda):
        self._params = [_FakeParam(is_cuda)]
        self.moves = []

    def parameters(self):
        return iter(self._params)

    def to(self, device):
        self.moves.append(device)
        for param in self._params:
            param.is_cuda = device == "cuda"
        return self


class GgufResidentTests(unittest.TestCase):
    def test_ensure_dit_on_gpu_moves_only_once(self):
        dit = _FakeDiT(is_cuda=False)
        with patch("torch.cuda.synchronize"):
            self.assertTrue(generate.ensure_dit_on_gpu(dit))
            self.assertFalse(generate.ensure_dit_on_gpu(dit))
        self.assertEqual(dit.moves, ["cuda"])

    def test_swap_dit_to_cpu_only_when_resident(self):
        dit = _FakeDiT(is_cuda=True)
        with patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
            self.assertTrue(generate.swap_dit_to_cpu(dit))
            self.assertFalse(generate.swap_dit_to_cpu(dit))
        self.assertEqual(dit.moves, ["cpu"])

    def test_decode_swap_wraps_vae_decode(self):
        dit = _FakeDiT(is_cuda=True)
        calls = []

        class FakeVAE:
            def decode(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "decoded"

        class FakePipe:
            vae = FakeVAE()
            transformer = dit

        pipe = FakePipe()
        generate.install_decode_swap(pipe)
        with patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
            self.assertEqual(pipe.vae.decode(1, return_dict=False), "decoded")
        self.assertEqual(calls, [((1,), {"return_dict": False})])
        self.assertEqual(dit.moves, ["cpu"])


class ReferenceResizeTests(unittest.TestCase):
    def test_vision_and_vae_reference_tokens_stay_aligned(self):
        from PIL import Image
        from transformers import Qwen2VLImageProcessor
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        processor = Qwen2VLImageProcessor(
            patch_size=16, merge_size=2,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )
        for resolution in (128, 256, 1024):
            with self.subTest(resolution=resolution):
                width, height, _ = calculate_dimensions(resolution ** 2, 981 / 1225)
                image = Image.new("RGB", (width, height))
                vae_tokens = width * height // (16 ** 2)
                processor.do_resize = True
                original = processor(images=image, return_tensors="pt")
                original_tokens = int(original.image_grid_thw.prod())
                if resolution < 1024:
                    self.assertNotEqual(original_tokens, vae_tokens)
                else:
                    self.assertEqual(original_tokens, vae_tokens)
                processor.do_resize = False
                aligned = processor(images=image, return_tensors="pt")
                self.assertEqual(int(aligned.image_grid_thw.prod()), vae_tokens)


if __name__ == "__main__":
    unittest.main()
