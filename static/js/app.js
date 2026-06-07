const API = "/api/v1";
const ANALYZE_INTERVAL_MS = 2800;

const COLORS = {
  플라스틱: "#5CB85C",
  유리: "#2E6B4E",
  금속: "#4A90D9",
  종이: "#C4A574",
  기타: "#AAAAAA",
};
const CHART_ORDER = ["플라스틱", "유리", "금속", "기타"];

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const donut = document.getElementById("donut");
const legend = document.getElementById("legend");
const statusEl = document.getElementById("status");
const primaryLabel = document.getElementById("primaryLabel");
const capturePreview = document.getElementById("capturePreview");
const chartCard = document.getElementById("chartCard");
const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");
const aiBadge = document.getElementById("aiBadge");
const aiSummary = document.getElementById("aiSummary");
const contaminationBox = document.getElementById("contaminationBox");
const contaminationLevel = document.getElementById("contaminationLevel");
const contaminationDetail = document.getElementById("contaminationDetail");
const recyclableBox = document.getElementById("recyclableBox");
const recyclableLabel = document.getElementById("recyclableLabel");
const recyclableReason = document.getElementById("recyclableReason");
const disposalSteps = document.getElementById("disposalSteps");
const warningsEl = document.getElementById("warnings");
const aiCard = document.getElementById("aiCard");

const CONTAMINATION_KO = {
  clean: "깨끗함",
  low: "경미 오염",
  high: "심각 오염",
};

let stream = null;
let geminiConfigured = null;
let sessionId = null;
let analyzeTimer = null;
let rafId = null;
let lastDetections = [];
let lastLocked = false;

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function fetchGeminiStatus() {
  try {
    const res = await fetch(`${API}/health`);
    if (!res.ok) return;
    const body = await res.json();
    geminiConfigured = body.gemini?.configured ?? false;
    if (!geminiConfigured) {
      aiBadge.textContent = "미설정";
      aiBadge.className = "badge badge-warn";
      aiSummary.textContent =
        "GEMINI_API_KEY를 .env에 설정하면 오염도·재활용·분리배출 안내가 표시됩니다.";
    }
  } catch {
    /* ignore */
  }
}

function updateAiPanel(body) {
  if (body.ai_enabled && body.contamination) {
    const src = body.ai_source || "local";
    aiBadge.textContent = src === "gemini" ? "Gemini" : "로컬 AI";
    aiBadge.className = "badge badge-ok";
    aiCard.classList.add("ai-active");

    const level = body.contamination.level || "low";
    contaminationLevel.textContent = CONTAMINATION_KO[level] || level;
    contaminationDetail.textContent = body.contamination.detail || "";
    contaminationBox.className = `ai-box level-${level}`;

    if (body.recyclable) {
      recyclableLabel.textContent = body.recyclable.label || "—";
      recyclableReason.textContent = body.recyclable.reason || "";
      recyclableBox.className = `ai-box recyclable-${body.recyclable.possible ? "yes" : "no"}`;
    }

    aiSummary.textContent = body.ai_summary || "분석 완료";

    disposalSteps.innerHTML = "";
    (body.disposal_steps || []).forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step.replace(/^\d+\.\s*/, "");
      disposalSteps.appendChild(li);
    });

    warningsEl.innerHTML = "";
    (body.warnings || []).forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      warningsEl.appendChild(li);
    });
    warningsEl.classList.toggle("hidden", !(body.warnings || []).length);
  } else if (geminiConfigured === false) {
    aiBadge.textContent = "미설정";
    aiBadge.className = "badge badge-warn";
  } else {
    aiBadge.textContent = "분석 중";
    aiBadge.className = "badge";
  }
}

function summaryMap(summary) {
  const m = {};
  (summary || []).forEach((s) => {
    m[s.label] = s.percent;
  });
  return m;
}

function materialColor(material) {
  if (material === "금속") return COLORS.금속;
  if (material === "플라스틱") return COLORS.플라스틱;
  if (material === "유리") return COLORS.유리;
  if (material === "종이") return COLORS.종이;
  return COLORS.기타;
}

function drawDonut(summary) {
  const ctx = donut.getContext("2d");
  const w = donut.width;
  const h = donut.height;
  const cx = w / 2;
  const cy = h / 2 - 10;
  const outerR = 100;
  const innerR = 58;

  ctx.clearRect(0, 0, w, h);

  const data = CHART_ORDER.map((label) => ({
    label,
    value: Math.max(summary[label] || 0, 0),
    color: COLORS[label] || COLORS.기타,
  })).filter((d) => d.value > 0.5);

  if (!data.length) {
    data.push({ label: "기타", value: 100, color: COLORS.기타 });
  }

  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  let start = -Math.PI / 2;

  data.forEach((d) => {
    const slice = (d.value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, outerR, start, start + slice);
    ctx.arc(cx, cy, innerR, start + slice, start, true);
    ctx.closePath();
    ctx.fillStyle = d.color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    start += slice;
  });

  ctx.beginPath();
  ctx.arc(cx, cy, innerR - 2, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.fillStyle = "#5CB85C";
  ctx.font = "bold 15px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("ECO", cx, cy);

  legend.innerHTML = "";
  CHART_ORDER.forEach((label) => {
    const pct = summary[label] || 0;
    if (pct < 0.5) return;
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot" style="background:${COLORS[label]}"></span>${label} ${pct.toFixed(0)}%`;
    legend.appendChild(li);
  });
}

/** 박스 꼭짓점 + 연결선 + 중심 허브 (점·선 오버레이) */
function drawOverlay(detections, locked) {
  const ctx = overlay.getContext("2d");
  const w = overlay.width;
  const h = overlay.height;
  if (!w || !h) return;

  ctx.clearRect(0, 0, w, h);

  (detections || []).forEach((det) => {
    const color = materialColor(det.material);
    const [x1, y1, x2, y2] = det.bbox || [];
    if (x1 == null) return;

    const corners =
      det.corners?.length === 4
        ? det.corners.map((p) => [Number(p[0]), Number(p[1])])
        : [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
          ];

    const cx = Math.round((x1 + x2) / 2);
    const cy = Math.round((y1 + y2) / 2);

    // 1) 박스 외곽선 (4변)
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    corners.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.closePath();
    ctx.stroke();

    // 2) 대각선 + 중심→꼭짓점 (점·선 네트워크)
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    for (let i = 0; i < 4; i++) {
      const [x, y] = corners[i];
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    ctx.lineTo(corners[2][0], corners[2][1]);
    ctx.moveTo(corners[1][0], corners[1][1]);
    ctx.lineTo(corners[3][0], corners[3][1]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3) 꼭짓점 점 (큰 원)
    corners.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // 4) 중심 점
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // 5) 라벨 (중심 위)
    const wasteName = det.waste_type_ko || det.object_name_ko || "물체";
    const label = `${wasteName} · ${det.material} ${Math.round((det.confidence || 0) * 100)}%`;
    const ly = Math.max(y1 - 12, 22);
    ctx.font = "bold 15px Apple SD Gothic Neo, Malgun Gothic, sans-serif";
    const tw = ctx.measureText(label).width;
    const tx = Math.max(4, Math.min(cx - tw / 2, w - tw - 8));
    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.fillRect(tx - 4, ly - 18, tw + 12, 22);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(tx - 4, ly - 18, tw + 12, 22);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, tx, ly);
  });

  if (locked) {
    ctx.fillStyle = "rgba(92, 184, 92, 0.9)";
    ctx.fillRect(w - 118, 10, 108, 30);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText("캡처 완료", w - 108, 30);
  }
}

function syncCanvasSize() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  overlay.width = w;
  overlay.height = h;
  overlay.style.width = "100%";
  overlay.style.height = "100%";
}

function overlayLoop() {
  syncCanvasSize();
  drawOverlay(lastDetections, lastLocked);
  rafId = requestAnimationFrame(overlayLoop);
}

async function analyzeFrame() {
  if (!stream || video.readyState < 2) return;

  syncCanvasSize();
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
  if (!blob) return;

  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  const headers = {};
  if (sessionId) headers["X-Session-Id"] = sessionId;

  setStatus("분석 중…");
  try {
    const res = await fetch(`${API}/materials/analyze?use_gemini=true`, {
      method: "POST",
      headers,
      body: form,
    });
    if (!res.ok) throw new Error(await res.text());
    const body = await res.json();
    sessionId = body.session_id;

    lastDetections = body.detections || [];
    lastLocked = !!body.locked;

    drawDonut(summaryMap(body.summary));

    const waste = body.waste_type_ko || "미확인";
    const mat = body.primary_material;
    primaryLabel.textContent = `${waste} (${mat} ${Math.round(body.confidence * 100)}%)`;

    updateAiPanel(body);

    if (body.locked) {
      chartCard.classList.add("locked");
      const aiNote = body.ai_enabled ? " · AI 분석 완료" : "";
      setStatus(`인식 확정 — 캡처·차트 저장됨${aiNote}`);
      if (body.chart_image_base64) {
        capturePreview.src = `data:image/png;base64,${body.chart_image_base64}`;
        capturePreview.classList.remove("hidden");
      }
    } else {
      chartCard.classList.remove("locked");
      const n = lastDetections.length;
      const aiNote = body.ai_enabled ? " · Gemini 분석됨" : "";
      setStatus(
        n ? `감지 ${n}개 — 점·선 표시 중${aiNote}` : `물체를 화면 중앙에 비춰 주세요${aiNote}`
      );
    }
  } catch (e) {
    setStatus(`오류: ${e.message}`);
  }
}

function startAnalyzeLoop() {
  stopAnalyzeLoop();
  analyzeTimer = setInterval(analyzeFrame, ANALYZE_INTERVAL_MS);
  analyzeFrame();
}

function stopAnalyzeLoop() {
  if (analyzeTimer) clearInterval(analyzeTimer);
  analyzeTimer = null;
}

async function startCamera() {
  sessionId = null;
  lastDetections = [];
  lastLocked = false;
  chartCard.classList.remove("locked");
  capturePreview.classList.add("hidden");

  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  video.onloadedmetadata = syncCanvasSize;
  window.addEventListener("resize", syncCanvasSize);

  btnStart.disabled = true;
  btnStop.disabled = false;
  setStatus("카메라 실행 중");
  if (!rafId) overlayLoop();
  startAnalyzeLoop();
}

function stopCamera() {
  stopAnalyzeLoop();
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  lastDetections = [];
  drawOverlay([], false);
  btnStart.disabled = false;
  btnStop.disabled = true;
  setStatus("정지됨");
}

btnStart.addEventListener("click", () => startCamera().catch((e) => setStatus(e.message)));
btnStop.addEventListener("click", stopCamera);

drawDonut({ 플라스틱: 0, 유리: 0, 금속: 0, 기타: 100 });
primaryLabel.textContent = "";
fetchGeminiStatus();
