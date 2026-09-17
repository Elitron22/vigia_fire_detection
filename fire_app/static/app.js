const $ = (selector) => document.querySelector(selector);
const state = { config: null, models: [], imageFile: null, videoFile: null, videoDuration: null, stream: null, socket: null, timer: null, waiting: false, resultObjectUrl: null };

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast visible${error ? " error" : ""}`;
  window.clearTimeout(element.hideTimer);
  element.hideTimer = window.setTimeout(() => element.className = "toast", 3800);
}

function decimal(value, digits = 2) { return Number(value).toLocaleString("es-ES", { minimumFractionDigits: digits, maximumFractionDigits: digits }); }

function readableDuration(seconds) {
  const rounded = Math.max(1, Math.round(seconds));
  if (rounded < 60) return `${rounded} s`;
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  return remainder ? `${minutes} min ${remainder} s` : `${minutes} min`;
}

function estimateVideoProcessing(file) {
  const estimate = $("#video-estimate");
  const preview = document.createElement("video");
  const objectUrl = URL.createObjectURL(file);
  preview.preload = "metadata";
  preview.onloadedmetadata = () => {
    URL.revokeObjectURL(objectUrl);
    const duration = Number(preview.duration);
    if (!Number.isFinite(duration) || duration <= 0) {
      estimate.textContent = "El tiempo dependerá de la duración y resolución del vídeo.";
      return;
    }
    state.videoDuration = duration;
    const lower = Math.max(5, duration * 1.1);
    const upper = Math.max(lower + 5, duration * 1.8);
    estimate.textContent = `Duración: ${readableDuration(duration)} · Tiempo estimado: ${readableDuration(lower)}–${readableDuration(upper)}.`;
  };
  preview.onerror = () => {
    URL.revokeObjectURL(objectUrl);
    state.videoDuration = null;
    estimate.textContent = "El tiempo dependerá de la duración y resolución del vídeo.";
  };
  preview.src = objectUrl;
}

function query() {
  const values = new URLSearchParams({
    hold_seconds: $("#hold-seconds").value,
    smoke_threshold: $("#smoke-threshold").value,
    fire_threshold: $("#fire-threshold").value,
  });
  values.set("experiment_id", $("#model-select").value);
  if (state.config?.inference.allow_runtime_overrides !== false) {
    values.set("confidence", Math.min(Number($("#smoke-threshold").value), Number($("#fire-threshold").value)).toString());
  }
  return values.toString();
}

function setBusy(button, busy, activeText) {
  if (!button.dataset.label) button.dataset.label = button.textContent;
  button.disabled = busy;
  button.textContent = busy ? activeText : button.dataset.label;
}

function showMedia(kind, source, focusResult = false) {
  $("#empty-state").hidden = true;
  const image = $("#result-image");
  const video = $("#result-video");

  if (state.resultObjectUrl && state.resultObjectUrl !== source) {
    URL.revokeObjectURL(state.resultObjectUrl);
    state.resultObjectUrl = null;
  }

  image.hidden = kind !== "image";
  video.hidden = kind !== "video";
  image.setAttribute("aria-hidden", String(kind !== "image"));
  video.setAttribute("aria-hidden", String(kind !== "video"));
  if (kind === "image") {
    video.pause();
    video.removeAttribute("src");
    video.load();
    image.src = source;
  } else {
    image.removeAttribute("src");
    video.src = source;
    if (source.startsWith("blob:")) state.resultObjectUrl = source;
    video.load();
  }
  if (focusResult) {
    const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    document.querySelector(".results-section").scrollIntoView({ behavior, block: "start" });
  }
}

function renderDetections(detections = []) {
  const container = $("#detection-summary");
  if (!detections.length) {
    container.innerHTML = '<p class="muted">No hay detecciones por encima del umbral.</p>';
    return;
  }
  const groups = detections.reduce((result, item) => {
    const group = result[item.class_name] || { count: 0, max: 0 };
    group.count += 1;
    group.max = Math.max(group.max, item.confidence);
    result[item.class_name] = group;
    return result;
  }, {});
  const labels = { fire: "Fuego", smoke: "Humo" };
  container.innerHTML = Object.entries(groups).map(([name, value]) =>
    `<div class="detection-chip ${name}"><span>${labels[name] || name} · ${value.count}</span><strong>${decimal(value.max * 100, 1)}%</strong></div>`
  ).join("");
}

function renderAlertState(snapshot = {}) {
  const active = Object.entries(snapshot).filter(([, details]) => details.state === "active").map(([name]) => name);
  const pending = Object.entries(snapshot).filter(([, details]) => details.state === "pending").map(([name]) => name);
  $("#alert-status").textContent = active.length ? `Activa: ${active.join(", ")}` : pending.length ? `Confirmando: ${pending.join(", ")}` : "Inactivo";
}

function renderEvents(events = []) {
  for (const event of events) {
    const label = event.event_type === "triggered" ? "Alerta activada" : "Alerta finalizada";
    const element = document.createElement("div");
    element.className = "event";
    element.textContent = `${label}: ${event.class_name} · ${decimal(event.timestamp_seconds, 1)} s`;
    $("#event-list").prepend(element);
  }
}

async function loadSystem() {
  try {
    const [healthResponse, modelsResponse, configResponse] = await Promise.all([
      fetch("/api/health"), fetch("/api/models"), fetch("/api/config")
    ]);
    if (!healthResponse.ok || !modelsResponse.ok || !configResponse.ok) throw new Error("La API no respondió correctamente.");
    const health = await healthResponse.json();
    const models = await modelsResponse.json();
    state.models = models.models;
    state.config = await configResponse.json();
    $("#health-dot").classList.add("online");
    $("#health-text").textContent = "disponible";
    $("#model-count").textContent = models.models.length;
    $("#device-label").textContent = health.device;
    $("#model-select").innerHTML = models.models.map(model =>
      `<option value="${model.experiment_id}" ${model.selected ? "selected" : ""}>${model.model_key} · ${model.inference_imgsz || model.trained_imgsz || "?"} px · ${model.backend}${model.frozen ? " · final" : ` · ${model.experiment_id.slice(-12)}`}</option>`
    ).join("");
    const defaults = state.config.inference;
    $("#hold-seconds").value = state.config.alerts.hold_seconds;
    if (!defaults.allow_runtime_overrides) {
      $("#model-select").disabled = models.models.length < 2;
    }
    syncSelectedModel();
    syncOutputs();
    const telegram = state.config.telegram;
    const messages = { dry_run: "Telegram en simulación: las alertas se registran sin enviar mensajes.", disabled: "Telegram desactivado.", live: telegram.credentials_configured ? "Telegram activo y configurado." : "Telegram activo, pero faltan credenciales." };
    const telegramMessage = messages[telegram.mode] || `Telegram: ${telegram.mode}`;
    const lockMessage = defaults.allow_runtime_overrides ? "" : " Resolución e IoU se aplican automáticamente según el perfil seleccionado.";
    $("#telegram-state").textContent = `${telegramMessage}${lockMessage}`;
    $("#test-telegram").disabled = !telegram.ready;
    if (!models.models.length) toast("No se encontró ningún modelo completo.", true);
  } catch (error) {
    $("#health-text").textContent = "no disponible";
    toast(error.message, true);
  }
}

function syncOutputs() {
  $("#smoke-threshold-output").textContent = decimal($("#smoke-threshold").value, 3);
  $("#fire-threshold-output").textContent = decimal($("#fire-threshold").value, 3);
  $("#hold-output").textContent = `${decimal($("#hold-seconds").value, 1)} s`;
  const model = state.models.find(item => item.experiment_id === $("#model-select").value);
  const defaults = model?.minimum_confidence;
  const customized = defaults && (
    Math.abs(Number($("#smoke-threshold").value) - Number(defaults.smoke)) > 1e-9 ||
    Math.abs(Number($("#fire-threshold").value) - Number(defaults.fire)) > 1e-9
  );
  $("#reset-thresholds").disabled = !customized;
}

function syncSelectedModel() {
  const model = state.models.find(item => item.experiment_id === $("#model-select").value);
  const thresholds = model?.minimum_confidence || state.config?.alerts.minimum_confidence;
  if (thresholds) {
    $("#smoke-threshold").value = thresholds.smoke;
    $("#fire-threshold").value = thresholds.fire;
  }
  syncOutputs();
}

function bindFileInput(inputSelector, dropzoneSelector, type) {
  const input = $(inputSelector);
  const zone = $(dropzoneSelector);
  const apply = (file) => {
    if (!file) return;
    state[`${type}File`] = file;
    zone.querySelector("strong").textContent = file.name;
    zone.querySelector("span:last-child").textContent = `${decimal(file.size / (1024 * 1024), 1)} MiB`;
    $(`#analyze-${type}`).disabled = false;
    if (type === "video") estimateVideoProcessing(file);
  };
  input.addEventListener("change", () => apply(input.files[0]));
  zone.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  ["dragenter", "dragover"].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove("dragging"); }));
  zone.addEventListener("drop", event => apply(event.dataTransfer.files[0]));
}

async function analyzeImage() {
  const button = $("#analyze-image");
  setBusy(button, true, "Analizando…");
  try {
    const response = await fetch(`/api/analyze/image?${query()}`, { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Filename": state.imageFile.name }, body: state.imageFile });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "No se pudo analizar la imagen.");
    showMedia("image", payload.annotated_image, true);
    renderDetections(payload.detections);
    $("#processing-time").textContent = `${decimal(payload.processing_ms, 0)} ms`;
    $("#run-badge").textContent = payload.run_id;
    $("#alert-status").textContent = "No aplica a una imagen";
  } catch (error) { toast(error.message, true); }
  finally { setBusy(button, false, ""); button.disabled = !state.imageFile; }
}

async function analyzeVideo() {
  const button = $("#analyze-video");
  const estimate = $("#video-estimate");
  const started = performance.now();
  setBusy(button, true, "Procesando fotogramas…");
  if (state.videoDuration) {
    const lower = Math.max(5, state.videoDuration * 1.1);
    const upper = Math.max(lower + 5, state.videoDuration * 1.8);
    estimate.textContent = `Procesando… tiempo estimado: ${readableDuration(lower)}–${readableDuration(upper)}.`;
  } else {
    estimate.textContent = "Procesando vídeo…";
  }
  try {
    const response = await fetch(`/api/analyze/video?${query()}`, { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Filename": state.videoFile.name }, body: state.videoFile });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "No se pudo procesar el vídeo.");
    }
    const blob = await response.blob();
    if (!blob.type.startsWith("video/")) throw new Error("La API no devolvió un vídeo reproducible.");
    showMedia("video", URL.createObjectURL(blob), true);
    renderDetections([]);
    $("#processing-time").textContent = "Incluido en metadatos";
    $("#run-badge").textContent = response.headers.get("X-TFM-Run-Id") || "Vídeo completado";
    $("#alert-status").textContent = `${response.headers.get("X-TFM-Event-Count") || 0} eventos`;
    estimate.textContent = `Procesamiento completado en ${readableDuration((performance.now() - started) / 1000)}.`;
    toast("Vídeo procesado y guardado.");
  } catch (error) {
    estimate.textContent = "No se pudo completar el procesamiento.";
    toast(error.message, true);
  }
  finally { setBusy(button, false, ""); button.disabled = !state.videoFile; }
}

async function testTelegram() {
  const button = $("#test-telegram");
  setBusy(button, true, "Enviando prueba…");
  try {
    const response = await fetch("/api/telegram/test", { method: "POST" });
    const payload = await response.json();
    if (!response.ok || payload.status === "error") throw new Error(payload.message || "No se pudo contactar con Telegram.");
    if (payload.status !== "sent") throw new Error(payload.message || "Telegram no está en modo activo.");
    toast("Mensaje de prueba enviado a Telegram.");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(button, false, "");
    button.disabled = !state.config?.telegram.ready;
  }
}

function sendCameraFrame() {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN || state.waiting || !state.stream) return;
  const video = $("#camera-source");
  if (!video.videoWidth) return;
  const canvas = $("#camera-canvas");
  const maxWidth = 960;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  state.waiting = true;
  canvas.toBlob(blob => { if (blob && state.socket?.readyState === WebSocket.OPEN) state.socket.send(blob); else state.waiting = false; }, "image/jpeg", .82);
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    $("#camera-source").srcObject = state.stream;
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    state.socket = new WebSocket(`${protocol}://${location.host}/api/live?${query()}`);
    state.socket.onmessage = event => {
      const payload = JSON.parse(event.data);
      if (payload.type === "error") { toast(payload.message, true); state.waiting = false; return; }
      if (payload.type === "prediction") {
        showMedia("image", payload.annotated_image);
        renderDetections(payload.detections);
        renderAlertState(payload.alert_state);
        renderEvents(payload.events);
        $("#processing-time").textContent = `${decimal(payload.processing_ms, 0)} ms`;
        state.waiting = false;
      }
      if (payload.type === "ready") $("#run-badge").textContent = payload.source_id;
    };
    state.socket.onerror = () => { toast("Se perdió la conexión de cámara.", true); state.waiting = false; };
    state.timer = window.setInterval(sendCameraFrame, 300);
    $("#start-camera").disabled = true;
    $("#stop-camera").disabled = false;
  } catch (error) { toast(`No se pudo abrir la cámara: ${error.message}`, true); }
}

function stopCamera() {
  window.clearInterval(state.timer);
  state.socket?.close();
  state.stream?.getTracks().forEach(track => track.stop());
  Object.assign(state, { stream: null, socket: null, timer: null, waiting: false });
  $("#start-camera").disabled = false;
  $("#stop-camera").disabled = true;
  $("#alert-status").textContent = "Inactivo";
}

document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(item => {
    const selected = item === tab;
    item.classList.toggle("active", selected);
    item.setAttribute("aria-selected", String(selected));
  });
  document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.toggle("active", panel.id === `panel-${tab.dataset.tab}`));
}));
["#smoke-threshold", "#fire-threshold", "#hold-seconds"].forEach(selector => $(selector).addEventListener("input", syncOutputs));
$("#model-select").addEventListener("change", syncSelectedModel);
$("#reset-thresholds").addEventListener("click", syncSelectedModel);
$("#test-telegram").addEventListener("click", testTelegram);
bindFileInput("#image-input", "#image-dropzone", "image");
bindFileInput("#video-input", "#video-dropzone", "video");
$("#analyze-image").addEventListener("click", analyzeImage);
$("#analyze-video").addEventListener("click", analyzeVideo);
$("#result-video").addEventListener("error", () => {
  if (!$("#result-video").hidden) toast("El navegador no pudo reproducir el vídeo generado.", true);
});
$("#start-camera").addEventListener("click", startCamera);
$("#stop-camera").addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);
loadSystem();
