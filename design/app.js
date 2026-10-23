"use strict";

const form = document.querySelector("#generation-form");
const $ = (id) => document.getElementById(id);
let running = false;
let toastTimer;
const references = [];
const terminal = new Set(["completed", "cancelled", "failed"]);
const hardLimits = { batch: 8, count: 64, references: 10, width: 4096, height: 4096, resolution: 4096, max_image_bytes: 10485760, max_total_image_bytes: 41943040 };
let service = null;
let connected = false;
let jobId = null;
let job = null;
let operation = 0;
let pollTimer;
let statusTimer;
let systemSampleTimer;
let cancelPending = false;
let cancelRequested = false;
let cancelError = "";
let connectionNotice = false;
let outputs = [];
let selectedImage = null;
let editableConfig;
const storageKey = "qwen-active-job";

function limit(name) {
  const supplied = service?.limits?.[name];
  return Number.isSafeInteger(supplied) && supplied > 0 ? Math.min(hardLimits[name], supplied) : hardLimits[name];
}

function rememberJob(id) {
  try {
    if (id) localStorage.setItem(storageKey, id);
    else localStorage.removeItem(storageKey);
  } catch { /* Storage may be disabled; in-page reconnect still works. */ }
}

function execution(status, detail, error = false) {
  connectionNotice = false;
  $("execution-status").textContent = status;
  $("execution-detail").textContent = detail;
  $("execution-status").classList.toggle("error", error);
  $("execution-status").classList.toggle("running", running && !error);
  $("execution-detail").parentElement.classList.toggle("error", error);
}

async function request(path, body, timeoutMs = body ? 30000 : 8000) {
  const controller = new AbortController();
  // This timeout only ends the HTTP wait. It never cancels backend inference.
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
      cache: "no-store",
      signal: controller.signal,
    });
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`服务返回了非 JSON 响应（HTTP ${response.status}），请使用 start_web.bat 启动本地服务。`); }
    if (!response.ok) {
      const error = new Error(data.error || `请求失败（HTTP ${response.status}）`);
      error.status = response.status;
      error.activeJob = data.active_job;
      throw error;
    }
    return data;
  } finally { clearTimeout(timeout); }
}

function connection(online) {
  connected = online;
  $("connection-dot").classList.toggle("connected", online);
  $("connection-label").textContent = online ? "CONNECTED" : "DISCONNECTED";
  $("backend-label").textContent = `BACKEND / ${online ? "CONNECTED" : "DISCONNECTED"}`;
  availability();
}

function availability() {
  const model = service?.models?.find((item) => item.id === editableConfig?.model);
  const ready = connected && service?.device !== "unavailable" && model?.available;
  let message;
  if (location.protocol === "file:") message = "请运行 start_web.bat，并在浏览器中打开服务地址；直接打开文件无法连接 API。";
  else if (!connected) message = "无法连接本地服务，请运行 start_web.bat；连接恢复后会自动检查。";
  else if (!model?.available) {
    const script = editableConfig?.model === "q8" ? "download_turbo.bat 与 download_gguf.bat" : "download_turbo.bat";
    message = `${model?.reason || "所选模型权重未就绪"}。请检查 ${script}，本页面不会自动下载或切换模型。`;
  } else if (service.device === "unavailable") message = "服务已连接，但推理设备不可用，请检查服务日志。";
  else message = "本地推理 · 生成过程中可请求停止";
  $("generation-help").textContent = running ? (jobId ? "停止请求需等待后端安全结束当前操作" : "正在准备请求或确认服务任务") : message;
  $("generation-help").classList.toggle("error", !running && !ready);
  $("generate-button").disabled = running ? !jobId || cancelPending || cancelRequested : !ready;
}

function notify(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
}

const chartMetrics = {
  memory: { history: [], value: null },
  vram: { history: [], value: null },
  gpu: { history: [], value: null },
};
const chartPoints = 36;
const chartView = { width: 72, height: 40, pad: 3 };

function formatGib(bytes) {
  return `${(bytes / 1073741824).toFixed(1)} GiB`;
}

function metricPercent(value) {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
}

function drawMetric(name) {
  const { history } = chartMetrics[name];
  if (!history.length) return;
  const { width, height, pad } = chartView;
  const usableWidth = width - pad * 2;
  const usableHeight = height - pad * 2;
  const step = usableWidth / (history.length - 1);
  const points = history.map((value, index) => {
    const x = pad + index * step;
    const y = pad + usableHeight * (1 - value / 100);
    return `${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  const bottom = (height - pad).toFixed(2);
  $(`${name}-line`).setAttribute("d", `M ${points.join(" L ")}`);
  $(`${name}-area`).setAttribute("d", `M ${pad} ${bottom} L ${points.join(" L ")} L ${(width - pad).toFixed(2)} ${bottom} Z`);
}

function updateMetric(name, value) {
  const state = chartMetrics[name];
  const percent = metricPercent(value);
  state.value = percent;
  if (percent === null) {
    $(`${name}-value`).textContent = "--";
    return;
  }
  if (!state.history.length) state.history = Array(chartPoints).fill(percent);
  state.history.push(percent);
  if (state.history.length > chartPoints) state.history.shift();
  $(`${name}-value`).textContent = `${Math.round(percent)}%`;
  drawMetric(name);
}

function updateSystemMonitor(status) {
  const memory = status?.memory;
  if (memory && Number.isFinite(memory.total_bytes) && memory.total_bytes > 0) {
    const used = memory.total_bytes - memory.available_bytes;
    updateMetric("memory", used / memory.total_bytes * 100);
    $(`memory-value`).title = `已用 ${formatGib(used)} / ${formatGib(memory.total_bytes)}`;
  } else {
    updateMetric("memory", null);
    $("memory-value").title = "服务未报告系统内存";
  }
  const gpu = status?.gpu;
  if (gpu && Number.isFinite(gpu.total_bytes) && gpu.total_bytes > 0) {
    const used = gpu.total_bytes - gpu.free_bytes;
    updateMetric("vram", used / gpu.total_bytes * 100);
    $("vram-value").title = `已用 ${formatGib(used)} / ${formatGib(gpu.total_bytes)}`;
    updateMetric("gpu", gpu.utilization);
    $("gpu-value").title = Number.isFinite(gpu.utilization) ? `GPU 占用率 ${Math.round(gpu.utilization)}%` : "服务未报告 GPU 占用率";
  } else {
    updateMetric("vram", null);
    updateMetric("gpu", null);
    $("vram-value").title = "服务未报告 GPU 显存";
    $("gpu-value").title = "服务未报告 GPU 占用率";
  }
}

async function refreshSystemMonitor() {
  clearTimeout(systemSampleTimer);
  if (location.protocol === "file:") {
    updateSystemMonitor(null);
    systemSampleTimer = setTimeout(refreshSystemMonitor, 1000);
    return;
  }
  try {
    updateSystemMonitor(await request("/api/status", undefined, 30000));
  } catch {
    updateSystemMonitor(null);
  }
  systemSampleTimer = setTimeout(refreshSystemMonitor, 1000);
}

function configuration() {
  const data = new FormData(form);
  const mode = references.length ? "edit" : "text";
  return {
    mode,
    prompt: String(data.get("prompt") || "").trim(),
    reference_names: references.map((ref) => ref.file.name),
    encoder_strategy: data.get("encoder_strategy"),
    dit_strategy: data.get("dit_strategy"),
    model: data.get("model"),
    batch: Number(data.get("batch")),
    count: Number(data.get("count")),
    seed: Number(data.get("seed")),
    width: data.get("width") ? Number(data.get("width")) : null,
    height: data.get("height") ? Number(data.get("height")) : null,
    resolution: Number(data.get("resolution")),
    transparent: data.has("transparent"),
    steps: 8,
  };
}

function updateConfiguration() {
  if (running) return;
  const config = configuration();
  editableConfig = config;
  const q8 = config.model === "q8";
  const residentEncoder = config.encoder_strategy === "resident";
  const residentDit = config.dit_strategy === "resident";
  const validBatch = Number.isSafeInteger(config.batch) && config.batch >= 1 && config.batch <= limit("batch");
  const validCount = Number.isSafeInteger(config.count) && config.count >= 1 && config.count <= limit("count");
  const batch = Math.min(config.batch, config.count);
  const partialSize = (config.width === null) !== (config.height === null);
  let invalidSize = false;
  for (const name of ["width", "height", "resolution"]) {
    const value = config[name];
    const input = $(name);
    const valid = (name !== "resolution" && value === null) || (Number.isSafeInteger(value) && value >= 32 && value <= limit(name) && /^[1-9][0-9]*$/.test(input.value));
    const message = !valid ? `请输入 32 至 ${limit(name)} 的整数。` : name !== "resolution" && partialSize ? "请同时填写输出宽度和高度，或将两项都留空。" : "";
    input.setCustomValidity(message);
    if (name !== "resolution" && !valid) invalidSize = true;
  }
  $("batch").setCustomValidity(validBatch ? "" : `每批张数需为 1 至 ${limit("batch")} 的整数。`);
  $("count").setCustomValidity(validCount ? "" : `总生成张数需为 1 至 ${limit("count")} 的整数。`);
  const validSeed = Number.isSafeInteger(config.seed) && config.seed >= 0 && config.seed <= 2147483647 && validCount && config.seed + config.count - 1 <= 2147483647;
  $("seed").setCustomValidity(validSeed ? "" : "种子需为 0 至 2147483647 的整数，逐张递增后也不能超出上限。" );
  $("prompt").setCustomValidity(config.prompt ? "" : "请输入提示词。" );
  const explicitSize = config.width !== null && config.height !== null;
  $("output-size-hint").textContent = invalidSize
    ? `宽高需要是 32 至 ${Math.min(limit("width"), limit("height"))} 的整数。`
    : partialSize ? "请同时填写宽高，或同时留空。"
    : explicitSize ? ""
      : config.mode === "edit" ? "宽高留空：跟随参考图比例，输出大小由参考图缩放设置决定。"
        : "宽高留空：文生图默认输出 2048 × 2048。";
  $("output-size-hint").hidden = explicitSize && !invalidSize && !partialSize;
  const validResolution = $("resolution").validity.valid;
  $("resolution-hint").textContent = validResolution
    ? `模型读取参考图时，每张参考图被处理成约${Math.round(config.resolution * config.resolution / 10000)}万（${config.resolution} x ${config.resolution}）的像素规模`
    : `请输入 32 至 ${limit("resolution")} 的整数。`;
  $("prompt-length").textContent = `${$("prompt").value.length} 字`;
  $("encoder-hint").textContent = residentEncoder
    ? "编码阶段整模型上 GPU；编码完成后移回 CPU，为图像迭代腾出显存。"
    : "编码时逐层加载与卸载；完成后移回 CPU，进入生图阶段。";
  $("batch-hint").textContent = validBatch && validCount
    ? `分 ${Math.ceil(config.count / batch)} 批生成，每批最多 ${batch} 张；种子逐张递增。${config.batch > config.count ? " Batch 将按总张数截断。" : ""}`
    : "请填写有效的每批张数和总生成张数。";
  $("generate-count").textContent = `${validCount ? config.count : "—"} 张`;
  $("strategy-notice").querySelector("span").textContent = `${q8 ? "Q8 仅替换 DiT，文本编码器仍使用官方 Turbo。" : "原版 BF16 权重。"} ${residentDit ? "DiT 在 8 步迭代中整模型驻留 GPU，解码前移回 CPU。" : "DiT 按层搬运，解码前移回 CPU。"}`;
  ensureTrack(references.length > 0);
  trackSub("encoding", strategyLabel(config.encoder_strategy));
  trackSub("denoising", `${q8 ? "Q8" : "BF16"} · ${strategyLabel(config.dit_strategy)}`);
  availability();
}

form.addEventListener("input", updateConfiguration);
form.addEventListener("change", updateConfiguration);

document.querySelectorAll(".dimension-combo").forEach((combo) => {
  const input = combo.querySelector("input");
  const toggle = combo.querySelector(".dimension-toggle");
  const list = combo.querySelector(".dimension-options");
  let active = -1;
  const options = (input.id === "resolution" ? [128, 256, 512, 1024, 2048] : [128, 512, 1024, 2048, 4096]).map((value) => {
    const option = document.createElement("button");
    option.type = "button";
    option.value = String(value);
    option.textContent = value;
    option.id = `${input.id}-option-${value}`;
    option.tabIndex = -1;
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", "false");
    list.append(option);
    option.addEventListener("click", () => choose(option));
    return option;
  });
  function setOpen(open, index = options.findIndex((option) => option.value === input.value)) {
    list.hidden = !open;
    input.setAttribute("aria-expanded", String(open));
    active = open ? index : -1;
    options.forEach((option, index) => option.setAttribute("aria-selected", String(index === active)));
    if (active >= 0) input.setAttribute("aria-activedescendant", options[active].id);
    else input.removeAttribute("aria-activedescendant");
  }
  function choose(option) {
    input.value = option.value;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    setOpen(false);
    input.focus({ preventScroll: true });
  }
  toggle.addEventListener("click", () => {
    setOpen(list.hidden);
    input.focus({ preventScroll: true });
  });
  input.addEventListener("input", () => { if (!list.hidden) setOpen(true); });
  input.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      setOpen(true, list.hidden || active < 0 ? (direction > 0 ? 0 : options.length - 1) : (active + direction + options.length) % options.length);
    } else if (!list.hidden && ["Enter", "Escape", "Home", "End", "Tab"].includes(event.key)) {
      if (event.key !== "Tab") event.preventDefault();
      if (event.key === "Enter" && active >= 0) choose(options[active]);
      else if (event.key === "Home" || event.key === "End") setOpen(true, event.key === "Home" ? 0 : options.length - 1);
      else setOpen(false);
    }
  });
  combo.addEventListener("focusout", (event) => { if (!combo.contains(event.relatedTarget)) setOpen(false); });
  document.addEventListener("pointerdown", (event) => { if (!combo.contains(event.target)) setOpen(false); });
  form.addEventListener("submit", () => setOpen(false));
});

document.querySelectorAll("[data-step]").forEach((button) => {
  button.addEventListener("click", () => {
    const input = $(button.dataset.target);
    const value = Number(input.value) || Number(input.min);
    input.value = Math.min(Number(input.max), Math.max(Number(input.min), value + Number(button.dataset.step)));
    updateConfiguration();
  });
});

$("random-seed").addEventListener("click", () => {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  const count = Math.max(1, Math.min(limit("count"), Math.trunc(Number($("count").value) || 1)));
  $("seed").value = values[0] % (2147483648 - count + 1);
  updateConfiguration();
});

const promptHistoryKey = "qwen-prompt-history";
const promptHistoryLimit = 20;
let promptHistory = (() => {
  try {
    const saved = JSON.parse(localStorage.getItem(promptHistoryKey) || "[]");
    return Array.isArray(saved) ? saved.filter((item) => typeof item === "string" && item.trim()).slice(0, promptHistoryLimit) : [];
  } catch { return []; }
})();

function savePromptHistory() {
  try { localStorage.setItem(promptHistoryKey, JSON.stringify(promptHistory)); } catch { /* Storage may be disabled. */ }
}

function renderPromptHistory() {
  $("prompt-history-empty").hidden = promptHistory.length > 0;
  $("prompt-history-clear").hidden = !promptHistory.length;
  const list = $("prompt-history-list");
  list.replaceChildren();
  promptHistory.forEach((text) => {
    const item = document.createElement("li");
    const fill = document.createElement("button");
    fill.type = "button";
    fill.className = "history-entry";
    fill.textContent = text;
    fill.title = text;
    fill.addEventListener("click", () => {
      $("prompt").value = text;
      $("prompt").dispatchEvent(new Event("input", { bubbles: true }));
      setPromptHistory(false);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "history-remove";
    remove.setAttribute("aria-label", `删除历史提示词：${text.slice(0, 30)}`);
    remove.innerHTML = '<svg class="icon"><use href="#i-close"/></svg>';
    remove.addEventListener("click", () => {
      promptHistory = promptHistory.filter((item) => item !== text);
      savePromptHistory();
      renderPromptHistory();
    });
    item.append(fill, remove);
    list.append(item);
  });
}

function setPromptHistory(open) {
  $("prompt-history").hidden = !open;
  $("prompt-history-toggle").setAttribute("aria-expanded", String(open));
}

function rememberPrompt(prompt) {
  const text = String(prompt || "").trim();
  if (!text) return;
  promptHistory = [text, ...promptHistory.filter((item) => item !== text)].slice(0, promptHistoryLimit);
  savePromptHistory();
  renderPromptHistory();
}

$("prompt-history-toggle").addEventListener("click", () => setPromptHistory($("prompt-history").hidden));
$("prompt-history-clear").addEventListener("click", () => {
  promptHistory = [];
  savePromptHistory();
  renderPromptHistory();
});
form.addEventListener("submit", () => setPromptHistory(false));
document.addEventListener("pointerdown", (event) => {
  if ($("prompt-history").hidden || event.target.closest("#prompt-history") || event.target.closest("#prompt-history-toggle")) return;
  setPromptHistory(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("prompt-history").hidden) setPromptHistory(false);
});

function renderReferences() {
  $("reference-list").replaceChildren();
  references.forEach((ref, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "reference";
    const img = document.createElement("img");
    img.src = ref.url;
    img.alt = ref.file.name;
    const number = document.createElement("span");
    number.className = "reference-index";
    number.textContent = `image${index + 1}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.setAttribute("aria-label", `移除参考图 ${ref.file.name}`);
    remove.innerHTML = '<svg class="icon"><use href="#i-close"/></svg>';
    remove.disabled = running;
    remove.addEventListener("click", () => {
      URL.revokeObjectURL(ref.url);
      references.splice(index, 1);
      renderReferences();
    });
    wrapper.append(img, number, remove);
    $("reference-list").append(wrapper);
  });
  updateConfiguration();
}
$("reference-input").addEventListener("change", (event) => {
  if (running) return;
  const files = [...event.target.files];
  const error = referenceError([...references.map((ref) => ref.file), ...files]);
  if (error) {
    execution("参考图未添加", error, true);
    notify(error);
  } else {
    for (const file of files) references.push({ file, url: URL.createObjectURL(file) });
  }
  event.target.value = "";
  renderReferences();
});

function referenceError(files) {
  if (files.length > limit("references")) return `最多添加 ${limit("references")} 张参考图。此次选择未添加。`;
  if (files.some((file) => !file.type.startsWith("image/") || !file.size)) return "请选择非空图片文件。";
  if (files.some((file) => file.size > limit("max_image_bytes"))) return `每张参考图不能超过 ${limit("max_image_bytes") / 1048576} MiB。`;
  if (files.reduce((total, file) => total + file.size, 0) > limit("max_total_image_bytes")) return `参考图总大小不能超过 ${limit("max_total_image_bytes") / 1048576} MiB。`;
  return "";
}

function encodeFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({ name: file.name, data: String(reader.result).split(",")[1] });
    reader.onerror = () => reject(new Error(`无法读取参考图 ${file.name}，请重新选择。`));
    reader.onabort = () => reject(new Error(`参考图 ${file.name} 读取被中断。`));
    reader.readAsDataURL(file);
  });
}

const trackPhases = ["loading", "encoding", "reference_encoding", "denoising", "decoding"];
const trackTitles = { loading: "权重加载", encoding: "文本编码", reference_encoding: "参考图编码", denoising: "8 步迭代", decoding: "VAE 解码" };
let trackHasReferences = null;
let detailState = { phase: null, signature: null, terminal: null, refs: {} };
let stepTicker = null;
let stepStartedAt = null;

function formatSeconds(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? `${value.toFixed(1)}s` : "—";
}

function strategyLabel(value) {
  return value === "resident" ? "显存常驻" : "Sequential";
}

function trackPhasesFor(hasReferences) {
  return trackPhases.filter((phase) => phase !== "reference_encoding" || hasReferences);
}

function renderTrack(hasReferences) {
  trackHasReferences = hasReferences;
  const phases = trackPhasesFor(hasReferences);
  const track = $("pipeline-track");
  track.replaceChildren();
  phases.forEach((phase, index) => {
    if (index) {
      const connector = document.createElement("span");
      connector.className = "track-connector";
      connector.dataset.connector = phases[index - 1];
      connector.innerHTML = '<svg class="icon"><use href="#i-arrow"/></svg>';
      track.append(connector);
    }
    const node = document.createElement("div");
    node.className = "pipeline-phase";
    node.dataset.phase = phase;
    node.dataset.number = String(index + 1).padStart(2, "0");
    const badge = document.createElement("span");
    badge.className = "track-node";
    const info = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = trackTitles[phase];
    const sub = document.createElement("span");
    sub.className = "phase-sub";
    info.append(title, sub);
    const time = document.createElement("span");
    time.className = "phase-time";
    node.append(badge, info, time);
    track.append(node);
  });
  resetTrack();
}

function ensureTrack(hasReferences) {
  if (trackHasReferences !== hasReferences) renderTrack(hasReferences);
}

function trackNode(phase) {
  return document.querySelector(`#pipeline-track .pipeline-phase[data-phase="${phase}"]`);
}

function trackState(phase, state) {
  const node = trackNode(phase);
  if (!node) return;
  node.classList.toggle("running", state === "running");
  node.classList.toggle("complete", state === "complete");
  const badge = node.querySelector(".track-node");
  const marker = state === "complete" ? "check" : "number";
  if (badge.dataset.marker !== marker) {
    badge.dataset.marker = marker;
    if (marker === "check") badge.innerHTML = '<svg class="icon"><use href="#i-check"/></svg>';
    else badge.textContent = node.dataset.number;
  }
  const phases = trackPhasesFor(trackHasReferences);
  const index = phases.indexOf(phase);
  if (index > 0) {
    const connector = document.querySelector(`#pipeline-track .track-connector[data-connector="${phases[index - 1]}"]`);
    if (connector) connector.classList.toggle("complete", state !== "idle");
  }
}

function trackSub(phase, text) {
  const node = trackNode(phase);
  if (node) node.querySelector(".phase-sub").textContent = text;
}

function trackTime(phase, text) {
  const node = trackNode(phase);
  if (node) node.querySelector(".phase-time").textContent = text;
}

function resetTrack() {
  ensureTrack(references.length > 0);
  for (const phase of trackPhasesFor(trackHasReferences)) {
    trackState(phase, "idle");
    trackSub(phase, "");
    trackTime(phase, "—");
  }
  const config = editableConfig;
  trackSub("encoding", strategyLabel(config.encoder_strategy));
  trackSub("denoising", `${config.model === "q8" ? "Q8" : "BF16"} · ${strategyLabel(config.dit_strategy)}`);
  const detail = $("pipeline-detail");
  detail.classList.remove("swap");
  detail.textContent = "等待开始";
  detailState = { phase: null, signature: null, terminal: null, refs: {} };
  stepStartedAt = null;
  stopStepTicker();
}

function detailChip(label) {
  const chip = document.createElement("span");
  chip.className = "detail-chip";
  chip.textContent = label;
  const value = document.createElement("b");
  value.textContent = "—";
  chip.append(value);
  return { chip, value };
}

function setChip(ref, value) {
  ref.value.textContent = formatSeconds(value);
}

function paintStepCells(cells, times, running) {
  [...cells.children].forEach((cell, index) => {
    const value = times[index];
    const done = typeof value === "number" && Number.isFinite(value);
    const current = running && !done && index === times.length;
    cell.classList.toggle("done", done);
    cell.classList.toggle("current", current);
    cell.replaceChildren();
    if (done) {
      cell.innerHTML = '<svg class="icon"><use href="#i-check"/></svg>';
      const duration = document.createElement("b");
      duration.textContent = formatSeconds(value);
      cell.append(duration);
      cell.title = `第 ${index + 1} 步 ${formatSeconds(value)}${index === 0 ? "（建立 KV 缓存）" : "（复用缓存）"}`;
    } else if (current) {
      cell.append(document.createElement("b"));
      cell.title = `第 ${index + 1} 步进行中`;
    } else {
      cell.textContent = String(index + 1);
      cell.title = "";
    }
  });
  if (running) tickStepTimer();
}

function stopStepTicker() {
  clearInterval(stepTicker);
  stepTicker = null;
}

function tickStepTimer() {
  if (typeof stepStartedAt !== "number") return;
  const cells = detailState.refs.cells;
  const cell = cells && [...cells.children].find((child) => child.classList.contains("current"));
  const label = cell && cell.querySelector("b");
  if (label) label.textContent = formatSeconds(Math.max(0, Date.now() / 1000 - stepStartedAt));
}

function ensureStepTicker() {
  if (stepTicker === null) stepTicker = setInterval(tickStepTimer, 100);
}

function swapDetail(detail) {
  detail.classList.remove("swap");
  void detail.offsetWidth;
  detail.classList.add("swap");
}

function renderDetail(snapshot, effective) {
  const phase = snapshot.phase;
  const running = !terminal.has(snapshot.status);
  const signature = JSON.stringify([snapshot.status, phase, snapshot.step, snapshot.step_times, snapshot.timings, snapshot.reference_index, snapshot.reference_total]);
  if (signature === detailState.signature) return;
  detailState.signature = signature;
  const detail = $("pipeline-detail");
  if (!running) {
    if (detailState.terminal === snapshot.status) return;
    detailState.phase = null;
    detailState.terminal = snapshot.status;
    detailState.refs = {};
    detail.replaceChildren();
    const text = document.createElement("span");
    text.className = "detail-note";
    text.textContent = snapshot.status === "completed" ? "生成完成" : snapshot.status === "cancelled" ? "已停止" : "生成失败";
    detail.append(text);
    swapDetail(detail);
    return;
  }
  detailState.terminal = null;
  const timings = snapshot.timings || {};
  const phaseChanged = detailState.phase !== phase;
  detailState.phase = phase;
  if (phaseChanged) {
    detail.replaceChildren();
    detailState.refs = {};
    if (phase === "encoding") {
      const refs = { encoder_load: detailChip("上卡"), encoding: detailChip("编码"), encoder_park: detailChip("下卡") };
      detailState.refs.chips = refs;
      detail.append(...Object.values(refs).map((ref) => ref.chip));
    } else if (phase === "denoising") {
      const cells = document.createElement("div");
      cells.className = "step-cells";
      for (let index = 0; index < 8; index++) {
        const cell = document.createElement("span");
        cell.className = "step-cell";
        cells.append(cell);
      }
      const note = document.createElement("span");
      note.className = "detail-note";
      const load = detailChip("DiT 上卡");
      detailState.refs = { cells, note, chips: { dit_load: load } };
      detail.append(cells, note, load.chip);
    } else if (phase === "decoding") {
      const refs = { dit_park: detailChip("DiT 下卡"), decoding: detailChip("解码") };
      detailState.refs.chips = refs;
      detail.append(...Object.values(refs).map((ref) => ref.chip));
    } else {
      const text = document.createElement("span");
      text.className = "detail-note";
      detailState.refs.text = text;
      detail.append(text);
    }
  }
  const refs = detailState.refs;
  if (phase === "queued") refs.text.textContent = "等待调度";
  else if (phase === "loading") refs.text.textContent = "本地权重读盘 · 不可中断";
  else if (phase === "encoding") for (const [name, ref] of Object.entries(refs.chips)) setChip(ref, timings[name]);
  else if (phase === "reference_encoding") refs.text.textContent = snapshot.reference_total ? `第 ${Math.max(snapshot.reference_index, 1)} / ${snapshot.reference_total} 张参考图` : "参考图编码中…";
  else if (phase === "denoising") {
    const times = Array.isArray(snapshot.step_times) ? snapshot.step_times : [];
    paintStepCells(refs.cells, times, running);
    refs.note.textContent = times.length ? `KV 缓存：第 1 步建立，已复用 ${Math.max(0, times.length - 1)} / 7` : "KV 缓存：第 1 步建立";
    setChip(refs.chips.dit_load, timings.dit_load);
  } else if (phase === "decoding") for (const [name, ref] of Object.entries(refs.chips)) setChip(ref, timings[name]);
  else refs.text.textContent = "写入 PNG 与生成元数据";
  if (phaseChanged) swapDetail(detail);
}

function renderCandidates() {
  const gallery = $("candidate-gallery");
  gallery.replaceChildren();
  outputs.forEach((image, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate";
    button.dataset.index = index;
    button.setAttribute("aria-label", `查看图片 ${index + 1}，种子 ${image.seed}`);
    const img = document.createElement("img");
    img.src = image.url;
    img.alt = image.filename;
    const caption = document.createElement("span");
    caption.textContent = `SEED ${image.seed}`;
    const check = document.createElement("span");
    check.className = "candidate-check";
    check.innerHTML = '<svg class="icon"><use href="#i-check"/></svg>';
    button.append(img, caption, check);
    gallery.append(button);
  });
  selection();
}

function selection() {
  const index = outputs.findIndex((image) => image.id === selectedImage?.id && image.jobId === selectedImage?.jobId);
  [...$("candidate-gallery").children].forEach((button, position) => {
    button.classList.toggle("selected", position === index);
    button.setAttribute("aria-pressed", String(position === index));
  });
  $("results-count").textContent = `${String(index + 1).padStart(2, "0")} / ${String(outputs.length).padStart(2, "0")}`;
  $("preview-subtitle").textContent = `OUTPUT / ${String(outputs.length).padStart(2, "0")}`;
  $("gallery-description").textContent = outputs.length ? `已保存 ${outputs.length} 张原始 PNG` : "仅显示已保存的图片";
  $("sample-tag").textContent = selectedImage ? index < 0 ? "显示上次生成结果" : "已保存的生成结果" : "暂无生成结果";
  if (!selectedImage) return;
  const image = selectedImage;
  const title = `${image.filename} · ${image.width} × ${image.height} · ${image.mode} · ${image.model === "q8" ? "TURBO Q8" : "TURBO BF16"} · Seed ${image.seed}`;
  for (const img of [$("preview-image"), $("preview-dialog").querySelector("img")]) {
    if (img.getAttribute("src") !== image.url) img.src = image.url;
    img.alt = title;
    img.title = title;
  }
  $("preview-dialog").querySelector("p").textContent = title;
  $("preview-dialog").setAttribute("aria-label", title);
  $("artwork-caption").textContent = image.filename;
  $("canvas-label").textContent = "ORIGINAL PNG";
  $("canvas").classList.toggle("rgba", image.mode === "RGBA");
  $("image-meta").hidden = false;
  $("meta-model").textContent = image.model === "q8" ? "TURBO Q8" : "TURBO BF16";
  $("meta-size").textContent = `${image.width} × ${image.height}`;
  $("meta-seed").textContent = `Seed ${image.seed}`;
  $("meta-format").textContent = image.mode;
  $("meta-steps").textContent = image.steps ? `${image.steps} steps` : "";
  $("download-preview").hidden = false;
  $("download-preview").href = image.download_url;
  $("download-preview").download = image.filename;
  $("expand-preview").disabled = false;
}

$("candidate-gallery").addEventListener("click", (event) => {
  const candidate = event.target.closest(".candidate");
  if (!candidate) return;
  selectedImage = outputs[Number(candidate.dataset.index)];
  selection();
});
$("preview-image").addEventListener("error", () => {
  if (selectedImage) $("sample-tag").textContent = "图片加载失败，请检查服务连接";
});

function lock(locked) {
  running = locked;
  for (const control of form.querySelectorAll("input, select, textarea, button")) {
    if (control !== $("generate-button")) control.disabled = locked;
  }
  $("generate-button").type = locked ? "button" : "submit";
  $("generate-button").querySelector("span").textContent = locked ? "停止生成" : "开始生成";
  if (!locked) updateConfiguration();
  availability();
}

function beginJob(id = null) {
  clearTimeout(pollTimer);
  clearTimeout(statusTimer);
  operation++;
  jobId = id;
  job = null;
  cancelPending = false;
  cancelRequested = false;
  cancelError = "";
  connectionNotice = false;
  outputs = [];
  renderCandidates();
  lock(true);
  $("progress-fill").style.width = "0%";
  $("progress-value").textContent = "0 / 8";
  resetTrack();
  execution(id ? "正在恢复任务" : "正在准备请求", id ? `正在读取任务 ${id}，不会重复提交。` : "正在读取参考图并提交配置。" );
  if (id) {
    rememberJob(id);
    pollJob(operation);
  }
  return operation;
}

function finishJob() {
  operation++;
  clearTimeout(pollTimer);
  jobId = null;
  rememberJob(null);
  lock(false);
  $("execution-status").classList.remove("running");
  refreshStatus();
}

function applyJob(snapshot, token) {
  if (token !== operation || snapshot.id !== jobId) return;
  if (!Array.isArray(snapshot.outputs) || !["queued", "running", ...terminal].includes(snapshot.status)) throw new Error("服务返回了无效的任务状态。");
  job = snapshot;
  cancelRequested ||= Boolean(snapshot.cancel_requested);
  const previousLength = outputs.length;
  for (const output of snapshot.outputs) {
    const record = { ...output, jobId, steps: output.effective?.steps ?? output.config?.steps ?? snapshot.effective?.steps ?? snapshot.config?.steps };
    const index = outputs.findIndex((image) => image.id === output.id);
    if (index < 0) outputs.push(record);
    else outputs[index] = record;
  }
  if (outputs.length && selectedImage?.jobId !== jobId) selectedImage = outputs[0];
  else if (selectedImage?.jobId === jobId) selectedImage = outputs.find((image) => image.id === selectedImage.id) || selectedImage;
  if (previousLength !== outputs.length) renderCandidates();
  else selection();
  const effective = { ...snapshot.config, ...snapshot.effective };
  ensureTrack((snapshot.config?.reference_names || []).length > 0);
  trackSub("encoding", strategyLabel(effective.encoder_strategy));
  trackSub("reference_encoding", snapshot.reference_total ? `${snapshot.reference_index} / ${snapshot.reference_total}` : "");
  trackSub("denoising", `${effective.model === "q8" ? "Q8" : "BF16"} · ${strategyLabel(effective.dit_strategy)} · 批 ${snapshot.batch_index || 1} / ${snapshot.batches || 1}`);
  const done = terminal.has(snapshot.status);
  const order = trackPhasesFor(trackHasReferences);
  const current = order.indexOf(snapshot.phase);
  order.forEach((name, index) => {
    const complete = (done && snapshot.status === "completed") || snapshot.phase === "saving";
    const state = complete || index < current ? "complete" : current < 0 ? "idle" : index === current ? (done ? "idle" : "running") : "idle";
    trackState(name, state);
    trackTime(name, formatSeconds(snapshot.timings?.[name]));
  });
  stepStartedAt = typeof snapshot.step_started === "number" ? snapshot.step_started : null;
  renderDetail(snapshot, effective);
  if (!done && snapshot.phase === "denoising") ensureStepTicker();
  else stopStepTicker();
  const steps = snapshot.steps;
  const step = snapshot.step;
  $("progress-value").textContent = `${step} / ${steps}`;
  const progress = snapshot.batches > 0 && steps > 0 && snapshot.batch_index > 0 ? ((snapshot.batch_index - 1) * steps + step) / (snapshot.batches * steps) : 0;
  $("progress-fill").style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  const timingNames = { encoding: "文本编码", denoising: "迭代", decoding: "解码", reference_encoding: "参考图编码", saving: "保存" };
  const extraTimings = Object.entries(snapshot.timings || {}).filter(([name, value]) => timingNames[name] && typeof value === "number").map(([name, value]) => `${timingNames[name]} ${formatSeconds(value)}`);
  const detail = [snapshot.error || snapshot.detail, `已保存 ${outputs.length} / ${snapshot.count} 张`, snapshot.batch_index > 0 ? `第 ${snapshot.batch_index} / ${snapshot.batches} 批` : "", typeof snapshot.elapsed_seconds === "number" ? `耗时 ${formatSeconds(snapshot.elapsed_seconds)}` : "", ...extraTimings].filter(Boolean).join(" · ");
  const stopDetail = !done && cancelRequested ? "停止请求已提交，当前操作安全结束后停止。 " : !done && cancelPending ? "正在等待服务确认停止请求。 " : cancelError ? `停止请求未确认：${cancelError} ` : "";
  const phases = { queued: "等待调度", loading: "加载模型", encoding: "文本编码中", reference_encoding: "参考图编码中", denoising: "图像迭代中", decoding: "VAE 解码中", saving: "保存图片", completed: "生成完成", cancelled: "已停止", failed: "生成失败" };
  const status = !done && cancelRequested ? "正在停止，等待后端" : !done && cancelPending ? "正在请求停止" : !done && cancelError ? "停止请求未确认" : phases[snapshot.phase] || snapshot.status;
  execution(status, `${stopDetail}${detail}`, snapshot.status === "failed" || Boolean(cancelError));
  availability();
  if (done) finishJob();
}

async function pollJob(token) {
  if (token !== operation || !jobId) return;
  try {
    const snapshot = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (token !== operation) return;
    connection(true);
    applyJob(snapshot, token);
  } catch (error) {
    if (token !== operation) return;
    if (error.status === 404) {
      execution("任务不可用", `服务中已找不到任务 ${jobId}，可能已重启。已保存的预览保留。 ${error.message}`, true);
      finishJob();
      return;
    }
    connection(false);
    execution("任务连接中断", `${error.message} 正在重连，任务 ${jobId} 仍保留；不会重复提交或假定任务已停止。`, true);
  }
  if (token === operation && running) pollTimer = setTimeout(() => pollJob(token), 600);
}

async function refreshStatus() {
  clearTimeout(statusTimer);
  if (location.protocol === "file:") {
    connection(false);
    execution("未连接服务", $("generation-help").textContent, true);
    return;
  }
  const token = operation;
  try {
    const status = await request("/api/status");
    if (token !== operation) return;
    if (!Array.isArray(status.models)) throw new Error("服务状态缺少模型信息。");
    service = status;
    connection(true);
    if (!running) {
      updateConfiguration();
      if (connectionNotice) {
        connectionNotice = false;
        execution("等待开始", "编码 → 8 步迭代 → VAE 解码与保存");
      }
      if (status.active_job) beginJob(status.active_job);
    }
  } catch (error) {
    if (token !== operation) return;
    connection(false);
    if (!job && !running && (!$("execution-status").classList.contains("error") || connectionNotice)) {
      execution("未连接服务", `${error.message} 请运行 start_web.bat，页面将自动重连。`, true);
      connectionNotice = true;
    }
  }
  if (!running && token === operation) statusTimer = setTimeout(refreshStatus, 5000);
}

async function recoverSubmission(token, message) {
  if (token !== operation) return;
  try {
    const status = await request("/api/status");
    if (token !== operation) return;
    service = status;
    connection(true);
    if (status.active_job) {
      jobId = status.active_job;
      rememberJob(jobId);
      availability();
      pollJob(token);
      return;
    }
    execution("提交未成功确认", `${message} 服务当前没有活动任务，请检查已保存的结果后再手动重试。`, true);
    finishJob();
  } catch (error) {
    if (token !== operation) return;
    connection(false);
    execution("提交状态未知", `${message} ${error.message} 正在等待重连以确认是否已有任务；不会自动重新提交。`, true);
    pollTimer = setTimeout(() => recoverSubmission(token, message), 600);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (running) return;
  updateConfiguration();
  if (!form.reportValidity() || $("generate-button").disabled) return;
  // Snapshot FormData and ordered File objects before disabling any controls.
  const config = configuration();
  const files = references.map((ref) => ref.file);
  const invalidReferences = referenceError(files);
  if (invalidReferences) { execution("参考图无效", invalidReferences, true); return; }
  const token = beginJob();
  let submitted = false;
  try {
    const encoded = await Promise.all(files.map(encodeFile));
    if (token !== operation) return;
    execution("正在提交任务", "等待服务接受任务。" );
    submitted = true;
    const snapshot = await request("/api/jobs", { config, references: encoded });
    if (token !== operation) return;
    if (!snapshot.id) throw new Error("服务未返回任务 ID。" );
    jobId = snapshot.id;
    rememberPrompt(config.prompt);
    rememberJob(jobId);
    connection(true);
    applyJob(snapshot, token);
    if (token === operation) pollTimer = setTimeout(() => pollJob(token), 600);
  } catch (error) {
    if (token !== operation) return;
    if (error.status === 409 && error.activeJob) {
      jobId = error.activeJob;
      rememberJob(jobId);
      execution("服务已有任务", `${error.message} 正在接续查看任务 ${jobId}，不会再次提交。`);
      availability();
      pollJob(token);
    } else if (submitted && !error.status) {
      connection(false);
      execution("提交状态未知", `${error.message} 正在确认服务任务，配置保持锁定。`, true);
      recoverSubmission(token, error.message);
    } else {
      if (error.status === 503) {
        const model = service?.models?.find((item) => item.id === config.model);
        if (model) { model.available = false; model.reason = error.message; }
      }
      execution("提交失败", error.message, true);
      finishJob();
    }
  }
});

$("generate-button").addEventListener("click", async () => {
  if (!running || !jobId || cancelRequested || cancelPending) return;
  const token = operation;
  cancelPending = true;
  cancelError = "";
  availability();
  execution("正在请求停止", "等待服务确认停止请求；推理仍在后端运行。" );
  try {
    const snapshot = await request(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {});
    if (token !== operation) return;
    cancelRequested = true;
    connection(true);
    applyJob(snapshot, token);
  } catch (error) {
    if (token !== operation) return;
    cancelError = error.message;
    execution("停止请求未确认", `${error.message} 任务仍受监控，可以重试停止请求。`, true);
    notify(`停止请求未确认：${error.message}`);
  } finally {
    if (token === operation) { cancelPending = false; availability(); }
  }
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !$("preview-dialog").open) {
    event.preventDefault();
    if (!$("generate-button").disabled) {
      if (running) $("generate-button").click();
      else form.requestSubmit();
    }
  }
});

$("expand-preview").addEventListener("click", () => { if (selectedImage) $("preview-dialog").showModal(); });
$("close-preview").addEventListener("click", () => $("preview-dialog").close());
$("preview-dialog").addEventListener("click", (event) => {
  if (event.target !== $("preview-dialog")) return;
  const rect = event.target.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close();
});

updateConfiguration();
renderPromptHistory();
let savedJob;
try { savedJob = localStorage.getItem(storageKey); } catch { /* Optional persistence. */ }
if (savedJob && location.protocol !== "file:") beginJob(savedJob);
refreshStatus();
refreshSystemMonitor();
