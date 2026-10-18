"""Real loopback HTTP tests with tiny PIL fixtures, never model inference or downloads."""

import base64
import http.client
import io
import json
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import server
from inference import GenerationCancelled


ROOT = Path(__file__).resolve().parent


def image_bytes(mode="RGB", size=(64, 32), *, orientation=None, transparent_palette=False):
    with Image.new(mode, size) as image:
        buffer = io.BytesIO()
        if orientation is not None:
            exif = Image.Exif()
            exif[274] = orientation
            image.save(buffer, format="JPEG", exif=exif)
        else:
            image.save(buffer, format="PNG", **({"transparency": 0} if transparent_palette else {}))
        return buffer.getvalue()


def reference(name="reference.png", data=None):
    return {"name": name, "data": base64.b64encode(image_bytes() if data is None else data).decode("ascii")}


def pixel_header(width, height):
    contents = bytearray(image_bytes())
    contents[16:24] = struct.pack(">II", width, height)
    contents[29:33] = struct.pack(">I", zlib.crc32(contents[12:29]) & 0xffffffff)
    return bytes(contents)


class FakeEngine:
    def __init__(self):
        self.available = True
        self.device = "cuda"
        self.capability_calls = 0
        self.calls = []
        self.generated_bytes = {}
        self.started = threading.Event()
        self.first_output = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.output_gate = threading.Event()
        self.output_gate.set()
        self.fail_next = False
        self.fail_after_output = False
        self.finish_despite_cancel = False
        self.record_transform = None
        self.close_threads = []
        self.concurrent = 0
        self.max_concurrent = 0

    def capabilities(self):
        self.capability_calls += 1
        return {"models": [{"id": model, "name": model, "available": self.available,
                            "reason": "" if self.available else "Local weights are missing; no automatic downloads"}
                           for model in ("turbo", "q8")], "device": self.device, "gpu": None, "steps": 8,
                "limits": {"batch": 8, "count": 64, "references": 10, "width": 4096, "height": 4096,
                           "resolution": 4096}, "residency": "phase"}

    def generate(self, request, references, output_dir, *, cancel_event, on_event):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.calls.append({"request": request, "references": list(references),
                           "modes": [image.mode for image in references],
                           "sizes": [image.size for image in references], "thread": threading.get_ident(),
                           "directory": output_dir})
        self.started.set()
        try:
            if not self.gate.wait(5):
                raise AssertionError("Test gate was not released")
            if cancel_event.is_set():
                raise GenerationCancelled("Stopped at safe test boundary")
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError(f"Failed local load in {output_dir}; test failure\nnot a browser traceback")
            effective = {"model": request.model, "encoder_strategy": request.encoder_strategy,
                         "dit_strategy": request.dit_strategy, "batch": min(request.batch, request.count), "steps": 8}
            timings = {"encoding": 0.01, "denoising": 0.02}
            batches = (request.count + effective["batch"] - 1) // effective["batch"]
            records = []
            for phase in ("loading", "encoding", "reference_encoding", "denoising", "decoding", "saving"):
                event = {"phase": phase, "detail": phase, "batch_index": 1, "batches": batches,
                         "steps": 8, "step": 8 if phase == "denoising" else 0, "count": request.count,
                         "timings": timings, "effective": effective}
                if phase == "denoising":
                    event.update(step_time=0.08, step_times=[0.1, 0.09, 0.08], step_started=123.5)
                elif phase == "reference_encoding":
                    event.update(reference_index=len(references), reference_total=len(references))
                on_event(event)
            if batches > 1:
                on_event({"phase": "denoising", "detail": "second batch", "batch_index": 2, "batches": batches,
                          "steps": 8, "step": 1, "count": request.count, "timings": timings,
                          "effective": effective, "step_time": 0.4, "step_times": [0.4], "step_started": 124.0})
            for index in range(request.count):
                output_id = f"{index + 1:04d}"
                seed = request.seed + index
                filename = f"{output_id}_seed{seed}.png"
                path = output_dir / filename
                mode = "RGBA" if request.transparent else "RGB"
                metadata = PngInfo()
                metadata.add_text("test_config", json.dumps(request.to_dict()))
                with Image.new(mode, (32, 32)) as image, path.open("xb") as destination:
                    image.save(destination, format="PNG", pnginfo=metadata)
                self.generated_bytes[output_id] = path.read_bytes()
                record = {"id": output_id, "filename": filename, "path": str(path), "seed": seed,
                          "width": 32, "height": 32, "mode": mode, "model": request.model,
                          "effective": dict(effective)}
                if self.record_transform is not None:
                    record = self.record_transform(record)
                records.append(record)
                on_event({"output": record})
                if index == 0:
                    self.first_output.set()
                    if not self.output_gate.wait(5):
                        raise AssertionError("Test output gate was not released")
                if self.fail_after_output:
                    self.fail_after_output = False
                    raise RuntimeError("Failed after saving a partial result")
                if cancel_event.is_set() and not self.finish_despite_cancel:
                    raise GenerationCancelled("Stopped after saving")
            # Mutating event dictionaries afterwards must not mutate stored HTTP snapshots.
            effective["steps"] = 99
            timings["encoding"] = 99
            return records
        finally:
            self.concurrent -= 1

    def close(self):
        if self.concurrent:
            raise AssertionError("Engine closed during generation")
        self.close_threads.append(threading.get_ident())


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name) / "outputs"
        self.engine = FakeEngine()
        self.app = server.LocalHTTPServer(ROOT, 0, engine=self.engine, output_dir=self.output_dir,
                                          history_limit=2, request_timeout=0.3)
        self.thread = threading.Thread(target=self.app.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.engine.gate.set()
        self.engine.output_gate.set()
        self.app.shutdown()
        self.app.server_close()
        self.thread.join(2)

    def request(self, method="GET", path="/api/status", payload=None, *, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.app.server_port, timeout=3)
        try:
            body = json.dumps(payload).encode("utf-8") if payload is not None else None
            supplied = {"Content-Type": "application/json"} if payload is not None else {}
            supplied.update(headers or {})
            connection.request(method, path, body=body, headers=supplied)
            response = connection.getresponse()
            content = response.read()
            result = json.loads(content) if "application/json" in response.getheader("Content-Type", "") else content
            return response.status, result, dict(response.getheaders())
        finally:
            connection.close()

    def raw(self, method="POST", path="/api/jobs", *, headers=(), body=b"{}", host=True, eof=False):
        with socket.create_connection(("127.0.0.1", self.app.server_port), timeout=3) as connection:
            lines = [f"{method} {path} HTTP/1.1"]
            if host:
                lines.append(f"Host: 127.0.0.1:{self.app.server_port}")
            lines.extend(f"{name}: {value}" for name, value in headers)
            connection.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body)
            if eof:
                connection.shutdown(socket.SHUT_WR)
            response = http.client.HTTPResponse(connection)
            response.begin()
            return response.status, json.loads(response.read()), dict(response.getheaders())

    def submit(self, *, config=None, references=None):
        return self.request("POST", "/api/jobs", {"config": {"prompt": "test", **(config or {})},
                                                   "references": references or []})

    def wait(self, job_id):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            status, snapshot, _ = self.request(path=f"/api/jobs/{job_id}")
            self.assertEqual(status, 200)
            if snapshot["status"] in server.TERMINAL:
                return snapshot
            time.sleep(0.01)
        self.fail("Job did not reach a terminal state")

    def assert_error(self, response, expected):
        status, payload, headers = response
        self.assertEqual(status, expected, payload)
        self.assertIsInstance(payload["error"], str)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_status_and_static_whitelist(self):
        status, capability, headers = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(capability["limits"]["max_image_bytes"], 10485760)
        self.assertEqual(capability["limits"]["max_total_image_bytes"], 41943040)
        self.assertEqual(capability["residency"], "phase")
        self.assertIsNone(capability["active_job"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        for route, (name, mime) in server.STATIC.items():
            with self.subTest(route=route):
                status, contents, headers = self.request(path=route)
                self.assertEqual(status, 200)
                self.assertEqual(contents, (ROOT / "design" / name).read_bytes())
                self.assertEqual(headers["Content-Type"], mime)
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(self.request(path="/favicon.ico")[0], 204)

    def test_repository_and_traversal_never_served(self):
        paths = ["/server.py", "/README.md", "/requirements.txt", "/.hf/token", "/.venv/pyvenv.cfg",
                 "/design/index.html", "/desktop-preview.png", "/design/mobile-preview.png", "/outputs/",
                 "/../server.py", "/%2e%2e/server.py", "/%2e%2e%5cserver.py", "/app.js/../server.py",
                 "/api/jobs/../../server.py"]
        for path in paths:
            with self.subTest(path=path):
                self.assert_error(self.request(path=path), 404)
        self.assert_error(self.raw(method="GET", path="http://127.0.0.1/api/status", body=b""), 404)

    def test_host_checks_all_routes(self):
        for host in ("evil.example", "localhost", "127.0.0.1", f"evil.example:{self.app.server_port}",
                     f"127.0.0.1:{self.app.server_port + 1}", f"127.0.0.1:{self.app.server_port}.evil"):
            for path in ("/api/status", "/", "/unknown"):
                with self.subTest(host=host, path=path):
                    self.assert_error(self.request(path=path, headers={"Host": host}), 403)
        self.assert_error(self.raw(method="GET", body=b"", host=False), 403)
        self.assert_error(self.raw(method="GET", headers=[("Host", "evil.example")], body=b""), 403)
        self.assertEqual(self.request(headers={"Host": f"localhost:{self.app.server_port}"})[0], 200)

    def test_mutating_origin_and_fetch_site_checks(self):
        for headers in ({"Origin": "https://evil.example"}, {"Origin": "null"},
                        {"Origin": f"http://localhost:{self.app.server_port}"},
                        {"Origin": f"http://127.0.0.1:{self.app.server_port + 1}"},
                        {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}):
            with self.subTest(headers=headers):
                self.assert_error(self.request("POST", "/api/jobs", {"config": {"prompt": "test"}}, headers=headers), 403)
        self.assertEqual(self.engine.capability_calls, 0)
        headers = {"Origin": f"http://127.0.0.1:{self.app.server_port}", "Sec-Fetch-Site": "same-origin"}
        status, snapshot, _ = self.request("POST", "/api/jobs", {"config": {"prompt": "test"}}, headers=headers)
        self.assertEqual(status, 202)
        self.wait(snapshot["id"])
        self.assert_error(self.request("POST", f"/api/jobs/{snapshot['id']}/cancel", {},
                                       headers={"Origin": "https://evil.example"}), 403)

    def test_framing_and_media_errors_without_engine_calls(self):
        cases = [((), b"{}", 411),
                 ((("Content-Length", "-1"), ("Content-Type", "application/json")), b"", 400),
                 ((("Content-Length", "+2"), ("Content-Type", "application/json")), b"{}", 400),
                 ((("Content-Length", "1.5"), ("Content-Type", "application/json")), b"{}", 400),
                 ((("Content-Length", "2, 2"), ("Content-Type", "application/json")), b"{}", 400),
                 ((("Content-Length", "2"), ("Content-Length", "2")), b"{}", 400),
                 ((("Content-Length", str(server.MAX_BODY_BYTES + 1)),), b"", 413),
                 ((("Content-Length", "2"),), b"{}", 415),
                 ((("Content-Length", "2"), ("Content-Type", "text/plain")), b"{}", 415),
                 ((("Content-Length", "2"), ("Content-Type", "multipart/form-data")), b"{}", 415),
                 ((("Content-Length", "2"), ("Transfer-Encoding", "chunked")), b"{}", 400),
                 ((("Content-Length", "0"), ("Content-Type", "application/json")), b"", 400)]
        for headers, body, expected in cases:
            with self.subTest(headers=headers):
                self.assert_error(self.raw(headers=headers, body=body), expected)
        self.assertEqual(self.engine.capability_calls, 0)
        self.assertEqual(self.engine.calls, [])

    def test_truncated_body_and_socket_timeout(self):
        headers = [("Content-Length", "20"), ("Content-Type", "application/json")]
        self.assert_error(self.raw(headers=headers, body=b"{}", eof=True), 400)
        self.assert_error(self.raw(headers=headers, body=b"{}"), 408)
        self.assertEqual(self.engine.capability_calls, 0)

    def test_invalid_json_and_unknown_body_fields(self):
        for body in (b"{", b"\xff", b'{"config":{"prompt":"x","seed":NaN}}', b"[" * 2000,
                     b'{"config":{"prompt":"x","prompt":"y"}}'):
            with self.subTest(body=body[:60]):
                self.assert_error(self.raw(headers=[("Content-Length", str(len(body))),
                                                    ("Content-Type", "application/json")], body=body), 400)
        for payload in ([], None, 1, {}, {"config": {"prompt": "test"}, "output_dir": "../escape"},
                        {"config": [], "references": []}, {"config": {"prompt": "test"}, "references": "bad"}):
            with self.subTest(payload=payload):
                body = json.dumps(payload).encode()
                self.assert_error(self.raw(headers=[("Content-Length", str(len(body))),
                                                    ("Content-Type", "application/json")], body=body), 400)
        self.assertEqual(self.engine.capability_calls, 0)

    def test_generation_validation_and_seed_increment_bound(self):
        cases = [{"prompt": " "}, {"model": "base"}, {"encoder_strategy": "auto"}, {"dit_strategy": "offload"},
                 {"batch": 9}, {"count": 65}, {"batch": True}, {"count": 1.0}, {"seed": -1},
                 {"seed": 2147483647, "count": 2}, {"width": 32}, {"width": 31, "height": 32},
                 {"width": 4097, "height": 32}, {"resolution": 31}, {"resolution": 4097},
                 {"transparent": "true"}, {"steps": 9}, {"ratio": "1:1"}, {"output_dir": "../escape"}]
        for config in cases:
            with self.subTest(config=config):
                self.assert_error(self.submit(config=config), 400)
        self.assertEqual(self.engine.capability_calls, 0)
        status, queued, _ = self.submit(config={"seed": 2147483646, "count": 2, "width": 33, "height": 4096})
        self.assertEqual(status, 202)
        snapshot = self.wait(queued["id"])
        self.assertEqual([output["seed"] for output in snapshot["outputs"]], [2147483646, 2147483647])
        self.assertEqual(snapshot["config"]["width"], 33)

    def test_reference_validation_without_engine_calls(self):
        bad_references = [[reference()] * 11, [None], [{"name": "x", "data": "AA==", "path": "../escape"}],
                          [reference(name="\x00evil.png")], [reference(name="../")], [reference(name="..")],
                          [{"name": "x", "data": ""}], [{"name": "x", "data": "%%%"}],
                          [{"name": "x", "data": "data:image/png;base64,AAAA"}],
                          [{"name": "x", "data": "\u00e9\u00e9\u00e9\u00e9"}], [reference(data=b"not an image")],
                          [reference(data=image_bytes(size=(1, 1000)))]]
        for references in bad_references:
            with self.subTest(references=str(references)[:80]):
                self.assert_error(self.submit(references=references), 400)
        self.assertEqual(self.engine.capability_calls, 0)
        self.assertFalse(self.output_dir.exists())

    def test_byte_limits_are_checked_before_decode_and_load(self):
        encoded = reference()
        with patch.object(server, "MAX_IMAGE_BYTES", 8), patch.object(server.base64, "b64decode") as decode:
            self.assert_error(self.submit(references=[encoded]), 413)
            decode.assert_not_called()
        size = len(base64.b64decode(encoded["data"]))
        with patch.object(server, "MAX_TOTAL_IMAGE_BYTES", size + 1), patch("PIL.Image.open") as image_open:
            self.assert_error(self.submit(references=[encoded, encoded]), 413)
            image_open.assert_not_called()
        self.assertEqual(self.engine.capability_calls, 0)

    def test_pixel_limits_checked_before_any_image_load(self):
        for references in ([reference(data=pixel_header(5000, 4001))],
                           [reference(data=pixel_header(50000, 50000))],
                           [reference(data=pixel_header(5000, 3000))] * 3):
            with self.subTest(count=len(references)), patch("PIL.Image.Image.load", side_effect=AssertionError("must not load")):
                self.assert_error(self.submit(references=references), 413)
        self.assertEqual(self.engine.capability_calls, 0)

    def test_animated_reference_rejected(self):
        buffer = io.BytesIO()
        with Image.new("RGB", (32, 32), "red") as first, Image.new("RGB", (32, 32), "blue") as second:
            first.save(buffer, format="GIF", save_all=True, append_images=[second])
        self.assert_error(self.submit(references=[reference(data=buffer.getvalue())]), 400)
        self.assertEqual(self.engine.capability_calls, 0)

    def test_reference_order_conversion_exif_and_authoritative_annotations(self):
        references = [reference(name="C:\\private\\first.png", data=image_bytes("L")),
                      reference(name="../../second.png", data=image_bytes("RGBA")),
                      reference(name="third.jpg", data=image_bytes(orientation=6)),
                      reference(name="palette.png", data=image_bytes("P", transparent_palette=True)),
                      reference(name="x" * 140 + ".png", data=image_bytes())]
        status, queued, _ = self.submit(config={"mode": "text", "reference_names": ["forged"], "resolution": 64},
                                         references=references)
        self.assertEqual(status, 202)
        snapshot = self.wait(queued["id"])
        self.assertEqual(snapshot["status"], "completed")
        call = self.engine.calls[0]
        self.assertEqual(call["modes"], ["RGB", "RGBA", "RGB", "RGBA", "RGB"])
        self.assertEqual(call["sizes"][2], (32, 64))
        self.assertEqual(snapshot["config"]["mode"], "edit")
        self.assertEqual(snapshot["config"]["reference_names"][:4], ["first.png", "second.png", "third.jpg", "palette.png"])
        self.assertEqual(len(snapshot["config"]["reference_names"][4]), 120)
        for image in call["references"]:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))

    def test_text_auto_defaults_and_strategies_preserved(self):
        status, queued, _ = self.submit(config={"mode": "edit", "reference_names": ["forged"], "model": "q8",
                                                "encoder_strategy": "resident", "dit_strategy": "sequential"})
        self.assertEqual(status, 202)
        snapshot = self.wait(queued["id"])
        self.assertEqual(snapshot["config"]["mode"], "text")
        self.assertEqual(snapshot["config"]["reference_names"], [])
        self.assertIsNone(snapshot["config"]["width"])
        self.assertIsNone(snapshot["config"]["height"])
        self.assertEqual(snapshot["config"]["resolution"], 1024)
        self.assertEqual(snapshot["effective"]["encoder_strategy"], "resident")
        self.assertEqual(snapshot["effective"]["dit_strategy"], "sequential")
        self.assertEqual(snapshot["effective"]["model"], "q8")
        self.assertEqual(snapshot["effective"]["steps"], 8)
        self.assertEqual(snapshot["timings"]["encoding"], 0.01)

    def test_step_times_and_reference_annotations_passthrough_and_batch_reset(self):
        status, queued, _ = self.submit(config={"batch": 1, "count": 2},
                                        references=[reference(), reference()])
        self.assertEqual(status, 202)
        snapshot = self.wait(queued["id"])
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["batches"], 2)
        self.assertEqual(snapshot["batch_index"], 2)
        self.assertEqual(snapshot["step"], 8)
        self.assertEqual(snapshot["step_time"], 0.4)
        self.assertEqual(snapshot["step_times"], [0.4])
        self.assertEqual(snapshot["step_started"], 124.0)
        self.assertEqual(snapshot["reference_index"], 2)
        self.assertEqual(snapshot["reference_total"], 2)

    def test_missing_model_or_device_rejected_without_generation(self):
        self.engine.available = False
        self.assert_error(self.submit(references=[reference()]), 503)
        self.engine.available = True
        self.engine.device = "unavailable"
        self.assert_error(self.submit(), 503)
        self.assertEqual(self.engine.calls, [])
        self.assertFalse(self.output_dir.exists())
        self.assertIsNone(self.request()[1]["active_job"])

    def test_accepted_snapshot_poll_and_original_png_download(self):
        status, queued, _ = self.submit(config={"transparent": True, "batch": 8, "count": 2, "seed": 99})
        self.assertEqual(status, 202)
        self.assertRegex(queued["id"], r"^[0-9a-f]{32}$")
        self.assertEqual(queued["status"], "queued")
        expected = {"id", "status", "phase", "detail", "batch_index", "batches", "step", "steps", "step_time",
                    "step_times", "step_started", "reference_index", "reference_total", "completed", "count",
                    "elapsed_seconds", "timings", "outputs", "config", "effective", "error", "cancel_requested"}
        self.assertEqual(set(queued), expected)
        snapshot = self.wait(queued["id"])
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["completed"], 2)
        self.assertEqual(snapshot["steps"], 8)
        self.assertEqual(snapshot["batches"], 1)
        self.assertEqual(snapshot["effective"]["batch"], 2)
        self.assertIsNone(snapshot["error"])
        for output in snapshot["outputs"]:
            self.assertNotIn("path", output)
            self.assertNotIn(str(self.output_dir), json.dumps(output))
            status, contents, headers = self.request(path=output["url"])
            self.assertEqual(status, 200)
            self.assertEqual(contents, self.engine.generated_bytes[output["id"]])
            self.assertEqual(headers["Content-Type"], "image/png")
            self.assertNotIn("Content-Disposition", headers)
            status, download, headers = self.request(path=output["download_url"])
            self.assertEqual(download, contents)
            self.assertEqual(headers["Content-Disposition"], f'attachment; filename="{output["filename"]}"')
            with Image.open(io.BytesIO(download)) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(json.loads(image.info["test_config"])["seed"], 99)

    def test_single_owner_busy_conflict_and_nonblocking_poll(self):
        self.engine.gate.clear()
        status, queued, _ = self.submit()
        self.assertEqual(status, 202)
        self.assertTrue(self.engine.started.wait(2))
        responses = []
        threads = [threading.Thread(target=lambda: responses.append(self.submit())) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(responses), 6)
        for response in responses:
            self.assert_error(response, 409)
            self.assertEqual(response[1]["active_job"], queued["id"])
        status, running, _ = self.request(path=f"/api/jobs/{queued['id']}")
        self.assertEqual(running["status"], "running")
        self.assertEqual(self.request()[1]["active_job"], queued["id"])
        self.engine.gate.set()
        self.wait(queued["id"])
        next_job = self.submit()[1]
        self.wait(next_job["id"])
        self.assertEqual(self.engine.max_concurrent, 1)
        self.assertEqual(len({call["thread"] for call in self.engine.calls}), 1)
        self.assertNotEqual(self.engine.calls[0]["thread"], threading.get_ident())

    def test_cancel_is_request_until_safe_boundary_and_preserves_partial_png(self):
        self.engine.output_gate.clear()
        queued = self.submit(config={"count": 2})[1]
        self.assertTrue(self.engine.first_output.wait(2))
        status, requested, _ = self.request("POST", f"/api/jobs/{queued['id']}/cancel", {})
        self.assertEqual(status, 200)
        self.assertEqual(requested["status"], "running")
        self.assertTrue(requested["cancel_requested"])
        self.assertEqual(requested["completed"], 1)
        self.assertEqual(self.request(path=requested["outputs"][0]["url"])[0], 200)
        self.assert_error(self.submit(), 409)
        self.engine.output_gate.set()
        final = self.wait(queued["id"])
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(final["completed"], 1)
        self.assertIsNone(final["error"])
        self.assertEqual(self.request(path=final["outputs"][0]["url"])[1], self.engine.generated_bytes["0001"])
        self.assertEqual(self.request("POST", f"/api/jobs/{queued['id']}/cancel", {})[1], final)

    def test_queued_cancel_never_calls_generate(self):
        # Hold admission's reentrant lock so the worker cannot claim the newly queued job.
        from inference import GenerationRequest
        with self.app.manager._condition:
            snapshot = self.app.manager.submit(GenerationRequest("test"), [])
            requested = self.app.manager.cancel(snapshot["id"])
            self.assertEqual(requested["status"], "queued")
            self.assertTrue(requested["cancel_requested"])
        final = self.wait(snapshot["id"])
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(self.engine.calls, [])

    def test_cancel_terminal_race_does_not_override_completed(self):
        self.engine.output_gate.clear()
        self.engine.finish_despite_cancel = True
        queued = self.submit()[1]
        self.assertTrue(self.engine.first_output.wait(2))
        requested = self.request("POST", f"/api/jobs/{queued['id']}/cancel", {})[1]
        self.assertEqual(requested["status"], "running")
        self.engine.output_gate.set()
        final = self.wait(queued["id"])
        self.assertEqual(final["status"], "completed")
        self.assertTrue(final["cancel_requested"])
        self.assertEqual(self.request("POST", f"/api/jobs/{queued['id']}/cancel", {})[1], final)

    def test_failure_cleanup_next_job_and_safe_error(self):
        self.engine.fail_next = True
        first = self.submit(references=[reference()])[1]
        final = self.wait(first["id"])
        self.assertEqual(final["status"], "failed")
        self.assertIn("Failed local load", final["error"])
        self.assertNotIn(str(self.output_dir), final["error"])
        self.assertNotIn("\n", final["error"])
        with self.assertRaises(ValueError):
            self.engine.calls[0]["references"][0].getpixel((0, 0))
        self.assertIsNone(self.request()[1]["active_job"])
        second = self.submit()[1]
        self.assertEqual(self.wait(second["id"])["status"], "completed")
        self.assertEqual(self.engine.close_threads, [])

    def test_failure_after_partial_output_preserves_file(self):
        self.engine.fail_after_output = True
        queued = self.submit(config={"count": 2})[1]
        final = self.wait(queued["id"])
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["completed"], 1)
        self.assertEqual(self.request(path=final["outputs"][0]["url"])[0], 200)

    def test_output_path_injection_and_unrecorded_files_blocked(self):
        outside = Path(self.temp.name) / "0001_seed42.png"
        outside.write_bytes(image_bytes())
        self.engine.record_transform = lambda record: {**record, "path": str(outside)}
        queued = self.submit()[1]
        final = self.wait(queued["id"])
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["outputs"], [])
        self.assert_error(self.request(path=f"/api/jobs/{queued['id']}/outputs/0001"), 404)
        for suffix in ("../server.py", "%2e%2e/server.py", "0001_seed42.png", "0000", "9999"):
            self.assert_error(self.request(path=f"/api/jobs/{queued['id']}/outputs/{suffix}"), 404)

    def test_symlink_output_rechecked_at_read_time(self):
        queued = self.submit()[1]
        final = self.wait(queued["id"])
        output = final["outputs"][0]
        path = self.output_dir / queued["id"] / output["filename"]
        target = Path(self.temp.name) / "outside.png"
        target.write_bytes(b"private contents")
        try:
            link = Path(self.temp.name) / "test-link.png"
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(f"Symlink creation not permitted: {error}")
        path.unlink()
        link.rename(path)
        self.assert_error(self.request(path=output["url"]), 404)

    def test_unknown_jobs_cancel_body_and_methods(self):
        missing = "0" * 32
        self.assert_error(self.request(path=f"/api/jobs/{missing}"), 404)
        self.assert_error(self.request("POST", f"/api/jobs/{missing}/cancel", {}), 404)
        self.assert_error(self.request("POST", f"/api/jobs/{missing}/cancel", {"force": True}), 400)
        for method in ("PUT", "DELETE", "PATCH", "OPTIONS"):
            self.assert_error(self.request(method, "/api/jobs", {}), 404)
        self.assert_error(self.raw(method="BREW", body=b""), 404)
        self.assert_error(self.raw(method="BREW", headers=[("Host", "evil.example")], body=b""), 403)
        self.assert_error(self.request("GET", "/api/jobs"), 404)
        self.assert_error(self.request("POST", "/api/status", {}), 404)

    def test_history_eviction_preserves_disk_and_snapshots_are_deep_copies(self):
        ids = []
        for _ in range(3):
            queued = self.submit()[1]
            ids.append(queued["id"])
            self.wait(queued["id"])
        self.assert_error(self.request(path=f"/api/jobs/{ids[0]}"), 404)
        self.assertTrue((self.output_dir / ids[0] / "0001_seed42.png").is_file())
        self.assertEqual(self.request(path=f"/api/jobs/{ids[1]}")[0], 200)
        detached = self.app.manager.snapshot(ids[2])
        detached["outputs"][0]["effective"]["steps"] = 100
        detached["config"]["reference_names"].append("bad")
        original = self.app.manager.snapshot(ids[2])
        self.assertEqual(original["outputs"][0]["effective"]["steps"], 8)
        self.assertEqual(original["config"]["reference_names"], [])

    def test_shutdown_closes_only_on_owner_thread_after_cancel_boundary(self):
        self.engine.gate.clear()
        queued = self.submit()[1]
        self.assertTrue(self.engine.started.wait(2))
        self.app.manager.request_shutdown()
        requested = self.app.manager.snapshot(queued["id"])
        self.assertEqual(requested["status"], "running")
        self.assertTrue(requested["cancel_requested"])
        self.assertEqual(self.engine.close_threads, [])
        self.engine.gate.set()
        self.app.manager.close()
        self.assertEqual(self.app.manager.snapshot(queued["id"])["status"], "cancelled")
        self.assertEqual(self.engine.close_threads, [self.engine.calls[0]["thread"]])
        self.assert_error(self.submit(), 503)

    def test_factory_is_lazy_and_invalid_upload_does_not_construct_engine(self):
        calls = []
        app = server.LocalHTTPServer(ROOT, 0, engine_factory=lambda: calls.append(1) or FakeEngine(),
                                      output_dir=self.output_dir)
        try:
            self.assertEqual(calls, [])
            with self.assertRaises(server.APIError):
                server.validate_payload({"config": {"prompt": "test"}, "references": [reference(data=b"bad")]})
            self.assertEqual(calls, [])
            self.assertEqual(app.manager.capabilities()["steps"], 8)
            self.assertEqual(calls, [1])
        finally:
            app.server_close()
        self.assertEqual(len(app.manager._engine.close_threads), 1)

    def test_output_root_symlink_blocked_before_writing(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        try:
            self.output_dir.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlink creation not permitted: {error}")
        queued = self.submit()[1]
        self.assertEqual(self.wait(queued["id"])["status"], "failed")
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(list(outside.iterdir()), [])

    def test_shutdown_waits_for_in_flight_factory_and_closes_on_worker(self):
        entered, release = threading.Event(), threading.Event()
        engine = FakeEngine()

        def factory():
            entered.set()
            if not release.wait(3):
                raise AssertionError("Factory gate was not released")
            return engine

        manager = server.JobManager(ROOT, engine_factory=factory, output_dir=self.output_dir)
        results = []
        thread = threading.Thread(target=lambda: results.append(manager.capabilities()))
        try:
            thread.start()
            self.assertTrue(entered.wait(2))
            manager.request_shutdown()
            self.assertEqual(engine.close_threads, [])
            release.set()
            thread.join(2)
            manager.close()
            self.assertEqual(len(results), 1)
            self.assertEqual(engine.close_threads, [manager._worker.ident])
        finally:
            release.set()
            thread.join(2)
            manager.close()


class StartupTests(unittest.TestCase):
    def test_import_has_no_heavy_imports(self):
        code = "import sys, server; assert 'torch' not in sys.modules; assert 'PIL' not in sys.modules; assert 'inference' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_browser_ephemeral_port_startup_without_inference(self):
        process = subprocess.Popen([sys.executable, "-B", str(ROOT / "server.py"), "--port", "0", "--no-browser"],
                                   cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            lines = []
            reader = threading.Thread(target=lambda: lines.append(process.stdout.readline()), daemon=True)
            reader.start()
            reader.join(5)
            self.assertFalse(reader.is_alive(), "Startup did not print its selected port")
            self.assertTrue(lines[0].startswith("Qwen Image console: http://127.0.0.1:"), lines)
            port = int(lines[0].strip().rsplit(":", 1)[-1])
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                connection.request("GET", "/")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), (ROOT / "design" / "index.html").read_bytes())
            finally:
                connection.close()
        finally:
            process.terminate()
            process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
