"""Loopback-only web console. Importing this module never imports Torch or loads weights."""

import argparse
import base64
import binascii
import copy
import io
import json
import math
import os
import re
import socket
import stat
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 40 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_TOTAL_IMAGE_PIXELS = 40_000_000
MAX_REFERENCES = 10
TERMINAL = {"completed", "cancelled", "failed"}
JOB_ROUTE = re.compile(r"/api/jobs/([0-9a-f]{32})(?:/(cancel)|/outputs/([0-9]{4}))?\Z")
OUTPUT_FILENAME = re.compile(r"[0-9]{4}_seed[0-9]+\.png\Z")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/preview.svg": ("preview.svg", "image/svg+xml")}


class APIError(Exception):
    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status = status
        self.payload = {"error": message, **extra}


def validate_payload(payload):
    """Decode bounded images without creating an engine or touching model weights."""
    from inference import GenerationRequest

    if not isinstance(payload, dict) or set(payload) - {"config", "references"}:
        raise APIError(400, "Body must contain only config and references")
    try:
        request = GenerationRequest.from_dict(payload.get("config"))
    except (ValueError, TypeError) as error:
        raise APIError(400, str(error)) from None
    references = payload.get("references", [])
    if not isinstance(references, list) or len(references) > MAX_REFERENCES:
        raise APIError(400, "references must be a list of at most 10 images")

    names, blobs = [], []
    total_bytes = 0
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"name", "data"}:
            raise APIError(400, "Each reference must contain only name and data")
        name, encoded = reference["name"], reference["data"]
        if not isinstance(name, str) or "\x00" in name:
            raise APIError(400, "Reference name must be a non-empty string without NUL")
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
        name = "".join(char if char.isprintable() else "_" for char in name)[:120]
        if not name.strip() or name in (".", ".."):
            raise APIError(400, "Reference name must have a display basename")
        if not isinstance(encoded, str) or not encoded:
            raise APIError(400, "Reference data must be non-empty base64 without a URL prefix")
        if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise APIError(413, "Each reference must be at most 10 MiB")
        # Account for padding before allocating the decoded buffer.
        estimated = len(encoded) // 4 * 3 - (len(encoded) - len(encoded.rstrip("=")))
        if estimated > MAX_IMAGE_BYTES or total_bytes + estimated > MAX_TOTAL_IMAGE_BYTES:
            raise APIError(413, "References exceed the 10 MiB per-image or 40 MiB total limit")
        try:
            blob = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise APIError(400, "Reference data is not valid base64") from None
        total_bytes += len(blob)
        if len(blob) > MAX_IMAGE_BYTES or total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise APIError(413, "References exceed the 10 MiB per-image or 40 MiB total limit")
        names.append(name)
        blobs.append(blob)

    from PIL import Image, ImageOps, UnidentifiedImageError

    sources, images = [], []
    valid = False
    try:
        total_pixels = 0
        # Inspect every header before loading any pixels, including the combined budget.
        for blob in blobs:
            source = Image.open(io.BytesIO(blob), formats=("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"))
            sources.append(source)
            pixels = source.width * source.height
            total_pixels += pixels
            if pixels <= 0 or pixels > MAX_IMAGE_PIXELS or total_pixels > MAX_TOTAL_IMAGE_PIXELS:
                raise APIError(413, "References exceed the 20 million per-image or 40 million total pixel limit")
            if getattr(source, "n_frames", 1) != 1:
                raise APIError(400, "References must be single-frame images")
        for source in sources:
            oriented = ImageOps.exif_transpose(source)
            try:
                mode = "RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB"
                image = oriented.convert(mode)
            finally:
                oriented.close()
            images.append(image)
            ratio = image.width / image.height
            width = round(math.sqrt(request.resolution ** 2 * ratio) / 32) * 32
            height = round(math.sqrt(request.resolution ** 2 / ratio) / 32) * 32
            if not 32 <= width <= 4096 or not 32 <= height <= 4096:
                raise APIError(400, "Reference aspect/resolution produces dimensions outside 32..4096")
        request = replace(request, mode="edit" if images else "text", reference_names=tuple(names))
        valid = True
        return request, images
    except Image.DecompressionBombError:
        raise APIError(413, "Reference exceeds the input pixel limit") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError):
        raise APIError(400, "Reference is not a supported, valid image") from None
    finally:
        for source in sources:
            source.close()
        if not valid:
            for image in images:
                image.close()


@dataclass
class _Job:
    request: object
    references: list
    directory: Path
    cancel: threading.Event
    state: dict
    paths: dict
    created: float
    finished: float | None = None


class JobManager:
    def __init__(self, root, *, engine=None, engine_factory=None, output_dir=None, history_limit=64,
                 model_path=None, gguf_path=None, cpu=False):
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.root = Path(root).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else self.root / "outputs"
        self.history_limit = history_limit
        self.model_path, self.gguf_path, self.cpu = model_path, gguf_path, cpu
        self._engine = engine
        self._engine_factory = engine_factory
        self._engine_lock = threading.Lock()
        self._condition = threading.Condition()
        self._jobs = {}
        self._active = None
        self._stopping = False
        self._worker = threading.Thread(target=self._run, name="qwen-inference", daemon=True)
        self._worker.start()

    def _get_engine(self):
        with self._engine_lock:
            if self._engine is None:
                with self._condition:
                    if self._stopping:
                        raise APIError(503, "Server is shutting down")
                if self._engine_factory is not None:
                    self._engine = self._engine_factory()
                else:
                    from inference import InferenceEngine
                    self._engine = InferenceEngine(self.root, model_path=self.model_path,
                                                   gguf_path=self.gguf_path, cpu=self.cpu)
            return self._engine

    def capabilities(self):
        with self._condition:
            if self._stopping:
                raise APIError(503, "Server is shutting down")
        result = copy.deepcopy(self._get_engine().capabilities())
        for model in result["models"]:
            if model.get("reason"):
                model["reason"] = self._error_text(model["reason"])
        result.setdefault("limits", {}).update(max_image_bytes=MAX_IMAGE_BYTES,
                                               max_total_image_bytes=MAX_TOTAL_IMAGE_BYTES,
                                               max_image_pixels=MAX_IMAGE_PIXELS,
                                               max_total_image_pixels=MAX_TOTAL_IMAGE_PIXELS)
        with self._condition:
            result["active_job"] = self._active
        return result

    def _snapshot(self, job):
        result = copy.deepcopy(job.state)
        result["elapsed_seconds"] = max(0.0, (job.finished or time.monotonic()) - job.created)
        return result

    def snapshot(self, job_id):
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise APIError(404, "Unknown job; history is session-only and bounded")
            return self._snapshot(job)

    def submit(self, request, references):
        accepted = False
        try:
            with self._condition:
                self._check_admission()
            capabilities = self.capabilities()
            model = next((model for model in capabilities["models"] if model["id"] == request.model), None)
            if model is None or not model["available"]:
                reason = model.get("reason") if model else None
                raise APIError(503, reason or "Selected local model is unavailable")
            if capabilities["device"] == "unavailable":
                raise APIError(503, "CUDA is unavailable; CPU debugging must be explicitly enabled at startup")
            # Reserve only under this lock; capabilities and CUDA loading never hold it.
            with self._condition:
                self._check_admission()
                job_id = uuid.uuid4().hex
                batch = min(request.batch, request.count)
                state = {"id": job_id, "status": "queued", "phase": "queued", "detail": "Waiting for worker",
                         "batch_index": 0, "batches": (request.count + batch - 1) // batch,
                         "step": 0, "steps": 8, "step_time": None, "step_times": [], "step_started": None,
                         "reference_index": 0, "reference_total": 0,
                         "completed": 0, "count": request.count,
                         "timings": {}, "outputs": [], "config": request.to_dict(), "effective": {},
                         "error": None, "cancel_requested": False}
                job = _Job(request, references, self.output_dir / job_id, threading.Event(), state, {}, time.monotonic())
                self._jobs[job_id] = job
                self._active = job_id
                accepted = True
                snapshot = self._snapshot(job)
                self._condition.notify()
                return snapshot
        finally:
            if not accepted:
                for image in references:
                    image.close()

    def _check_admission(self):
        if self._stopping:
            raise APIError(503, "Server is shutting down")
        if self._active is not None:
            raise APIError(409, "Another generation job is active", active_job=self._active)

    def cancel(self, job_id):
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise APIError(404, "Unknown job")
            if job.state["status"] not in TERMINAL:
                job.state["cancel_requested"] = True
                job.cancel.set()
            return self._snapshot(job)

    def _checked_output_path(self, job, path):
        # The only allowed files are direct, non-symlink children of this generated job directory.
        if self.output_dir.is_symlink() or self.output_dir.resolve() != self.output_dir:
            raise ValueError("Unsafe output root")
        if job.directory.is_symlink() or job.directory.resolve().parent != self.output_dir:
            raise ValueError("Unsafe output directory")
        path = Path(path)
        if path.is_symlink() or path.parent.resolve() != job.directory.resolve():
            raise ValueError("Unsafe output path")
        if not OUTPUT_FILENAME.fullmatch(path.name) or not path.is_file():
            raise ValueError("Invalid saved output")
        return path

    def _on_event(self, job, event):
        public = path = None
        if "output" in event:
            record = event["output"]
            output_id = record["id"]
            if not isinstance(output_id, str) or not re.fullmatch(r"[0-9]{4}", output_id) or not 1 <= int(output_id) <= job.request.count:
                raise ValueError("Invalid output identifier")
            path = self._checked_output_path(job, record["path"])
            if record["filename"] != path.name or not path.name.startswith(output_id + "_seed"):
                raise ValueError("Invalid output filename")
            public = {key: copy.deepcopy(record[key]) for key in
                      ("id", "filename", "seed", "width", "height", "mode", "model", "effective", "config") if key in record}
            url = f"/api/jobs/{job.state['id']}/outputs/{output_id}"
            public.update(url=url, download_url=url + "?download=1")
        with self._condition:
            state = job.state
            if "phase" in event and event["phase"] != state["phase"] and "detail" not in event:
                state["detail"] = ""
            if "batch_index" in event and event["batch_index"] != state["batch_index"]:
                state["step"] = 0
                state["step_time"] = None
                state["step_times"] = []
                state["step_started"] = None
            for key in ("phase", "detail", "batch_index", "batches", "step", "steps", "count", "timings",
                        "effective", "step_time", "step_times", "step_started", "reference_index", "reference_total"):
                if key in event:
                    state[key] = copy.deepcopy(event[key])
            if public is not None and public["id"] not in job.paths:
                job.paths[public["id"]] = path
                state["outputs"].append(public)
            state["completed"] = len(state["outputs"])

    def open_output(self, job_id, output_id):
        with self._condition:
            job = self._jobs.get(job_id)
            path = job.paths.get(output_id) if job is not None else None
        if path is None:
            raise APIError(404, "Unknown output")
        try:
            path = self._checked_output_path(job, path)
            source = path.open("rb")
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                source.close()
                raise ValueError("Invalid output file")
            return source, path.name
        except (OSError, ValueError):
            raise APIError(404, "Saved output is no longer available") from None

    def _error_text(self, error):
        text = str(error).splitlines()[0] if str(error).strip() else type(error).__name__
        for path in (self.root, self.output_dir, self.model_path, self.gguf_path):
            if path is not None:
                text = text.replace(str(path), "[local path]").replace(Path(path).as_posix(), "[local path]")
        return text[:1200]

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._active is not None)
                    if self._active is None:
                        return
                    job = self._jobs[self._active]
                    job.state.update(status="running", detail="Starting generation")
                status, error = "completed", None
                try:
                    from inference import GenerationCancelled
                    if job.cancel.is_set():
                        raise GenerationCancelled("Generation cancelled before loading")
                    if self.output_dir.is_symlink() or self.output_dir.resolve() != self.output_dir:
                        raise ValueError("Unsafe output root")
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    job.directory.mkdir(exist_ok=False)
                    if job.directory.is_symlink() or job.directory.resolve().parent != self.output_dir:
                        raise ValueError("Unsafe output directory")
                    records = self._get_engine().generate(job.request, job.references, job.directory,
                                                          cancel_event=job.cancel,
                                                          on_event=lambda event: self._on_event(job, event))
                    for record in records:
                        self._on_event(job, {"output": record})
                except Exception as failure:
                    from inference import GenerationCancelled
                    status = "cancelled" if isinstance(failure, GenerationCancelled) else "failed"
                    error = None if status == "cancelled" else self._error_text(failure)
                finally:
                    for image in job.references:
                        image.close()
                    job.references.clear()
                    with self._condition:
                        job.finished = time.monotonic()
                        job.state.update(status=status, phase=status, error=error,
                                         detail={"completed": "Generation completed", "cancelled": "Generation stopped; saved outputs preserved",
                                                 "failed": "Generation failed; saved outputs preserved"}[status])
                        if status == "completed":
                            job.state.update(step=8, batch_index=job.state["batches"])
                        self._active = None
                        terminal_ids = [key for key, value in self._jobs.items() if value.state["status"] in TERMINAL]
                        for key in terminal_ids[:-self.history_limit]:
                            del self._jobs[key]
                        self._condition.notify_all()
        finally:
            # Only the inference owner closes a pipeline, after the active call has unwound.
            with self._engine_lock:
                if self._engine is not None:
                    self._engine.close()

    def request_shutdown(self):
        with self._condition:
            self._stopping = True
            if self._active is not None:
                job = self._jobs[self._active]
                job.state["cancel_requested"] = True
                job.cancel.set()
            self._condition.notify_all()

    def close(self):
        self.request_shutdown()
        self._worker.join()


class LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, root, port=7860, *, request_timeout=15, **manager_options):
        self.root = Path(root).resolve()
        self.request_timeout = request_timeout
        super().__init__(("127.0.0.1", port), RequestHandler)
        self.manager = JobManager(self.root, **manager_options)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(self.request_timeout)
        return connection, address

    def server_close(self):
        if hasattr(self, "manager"):
            self.manager.request_shutdown()
        super().server_close()
        if hasattr(self, "manager"):
            self.manager.close()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "QwenLocal"
    sys_version = ""

    def log_message(self, format, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        if code == 501:
            try:
                self._check_origin(False)
            except APIError as error:
                return self._json(error.status, error.payload)
            return self._json(404, {"error": "Unknown endpoint"})
        self._json(code, {"error": "Malformed HTTP request" if code == 400 else "Unsupported HTTP request"})

    def _headers(self, status, content_type, length, *, filename=None):
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        if filename is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        if getattr(self, "command", None) != "HEAD":
            self.wfile.write(body)

    def _check_origin(self, mutating):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        supplied = self.headers.get_all("Host", [])
        if len(supplied) != 1 or supplied[0] not in hosts:
            raise APIError(403, "Host must match this loopback server")
        if mutating:
            origins = self.headers.get_all("Origin", [])
            if origins and (len(origins) != 1 or origins[0] != f"http://{supplied[0]}"):
                raise APIError(403, "Cross-origin requests are not allowed")
            sites = self.headers.get_all("Sec-Fetch-Site", [])
            if sites and (len(sites) != 1 or sites[0] not in {"same-origin", "none"}):
                raise APIError(403, "Cross-site requests are not allowed")

    def _body(self):
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise APIError(411, "Content-Length is required")
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]) or len(lengths[0]) > 20:
            raise APIError(400, "Content-Length must be one nonnegative decimal integer")
        if self.headers.get_all("Transfer-Encoding", []):
            raise APIError(400, "Transfer-Encoding is not supported")
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            raise APIError(413, "JSON body exceeds 64 MiB")
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
            raise APIError(415, "Content-Type must be application/json")
        try:
            body = self.rfile.read(length)
        except (socket.timeout, OSError):
            raise APIError(408, "Request body timed out") from None
        if len(body) != length:
            raise APIError(400, "Request body is shorter than Content-Length")
        try:
            def reject_constant(value):
                raise ValueError("Non-finite JSON number")

            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate JSON field")
                    result[key] = value
                return result

            return json.loads(body.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)
        except (UnicodeError, ValueError, RecursionError):
            raise APIError(400, "Body must be valid UTF-8 JSON") from None

    def _file(self, source, content_type, *, filename=None):
        with source:
            length = os.fstat(source.fileno()).st_size
            self._headers(200, content_type, length, filename=filename)
            remaining = length
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _dispatch(self, method):
        try:
            self._check_origin(method == "POST")
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or parsed.fragment:
                raise APIError(404, "Unknown endpoint")
            path = parsed.path
            match = JOB_ROUTE.fullmatch(path)
            if method == "GET":
                if path == "/api/status":
                    return self._json(200, self.server.manager.capabilities())
                if match:
                    job_id, cancel, output_id = match.groups()
                    if output_id is not None:
                        source, filename = self.server.manager.open_output(job_id, output_id)
                        return self._file(source, "image/png", filename=filename if parsed.query == "download=1" else None)
                    if cancel is None:
                        return self._json(200, self.server.manager.snapshot(job_id))
                if path == "/favicon.ico":
                    return self._headers(204, "image/x-icon", 0)
                if path in STATIC:
                    name, content_type = STATIC[path]
                    static_dir = self.server.root / "design"
                    target = static_dir / name
                    if static_dir.is_symlink() or target.is_symlink() or target.resolve().parent != static_dir.resolve():
                        raise APIError(404, "Static asset is unavailable")
                    try:
                        source = target.open("rb")
                    except OSError:
                        raise APIError(404, "Static asset is unavailable") from None
                    return self._file(source, content_type)
            elif method == "POST":
                if path == "/api/jobs":
                    request, images = validate_payload(self._body())
                    return self._json(202, self.server.manager.submit(request, images))
                if match and match.group(2):
                    if self._body() != {}:
                        raise APIError(400, "Cancellation body must be an empty object")
                    return self._json(200, self.server.manager.cancel(match.group(1)))
            raise APIError(404, "Unknown endpoint")
        except APIError as error:
            self._json(error.status, error.payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            pass
        except Exception:
            try:
                self._json(500, {"error": "Internal server error; no generation was confirmed by this response"})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
                pass

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_PUT(self):
        self._dispatch("PUT")

    do_DELETE = do_PUT
    do_PATCH = do_PUT
    do_OPTIONS = do_PUT


def main():
    parser = argparse.ArgumentParser(description="Start the local Qwen Image web console (local weights only).")
    parser.add_argument("--port", type=int, default=7860, help="Loopback port; 0 selects an available port")
    parser.add_argument("--model-dir", type=Path, help="Trusted local official Turbo checkpoint directory")
    parser.add_argument("--gguf", type=Path, help="Trusted local Q8 GGUF file")
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU debugging; extremely slow")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the default browser")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be from 0 to 65535")
    root = Path(__file__).resolve().parent
    try:
        server = LocalHTTPServer(root, args.port, model_path=args.model_dir, gguf_path=args.gguf, cpu=args.cpu)
    except OSError:
        parser.exit(1, "Cannot bind the loopback port. Try --port with a different port.\n")
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"Qwen Image console: {address}", flush=True)
    print("Local weights only. Ctrl+C requests cancellation and waits for a safe worker boundary.", flush=True)
    try:
        if not args.no_browser:
            try:
                webbrowser.open(address)
            except webbrowser.Error:
                print("Default browser could not be opened; use the printed address.", flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
