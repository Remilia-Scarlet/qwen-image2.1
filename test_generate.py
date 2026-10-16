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
        self.diffusers = MagicMock()
        self.pipe = self.diffusers.QwenImage21Pipeline.from_pretrained.return_value
        self.result = MagicMock()
        self.pipe.return_value.images = [self.result]
        self.pil = MagicMock()
        modules = patch.dict(sys.modules, {
            "torch": self.torch, "diffusers": self.diffusers, "PIL": self.pil,
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
