/**
 * eval_vlm Web UI - Modern Vanilla JS Reactive Engine
 * Zero CDN dependencies, 100% offline & local network compatible.
 * Full support for both standard eval and field-eval, with transparent Job Queue.
 */

// 全局应用状态
const state = {
  activeTab: "datasets",
  datasets: [],
  currentDataset: "",
  datasetDetail: null,

  // 画廊与样本
  samples: [],
  samplesTotal: 0,
  samplesOffset: 0,
  samplesLimit: 24,
  samplesFilter: "",
  testSha: "",

  // 灯箱
  lightbox: { open: false, url: "", ref: "" },

  // 删除弹窗
  deleteModal: { open: false, sampleId: "", mode: "record", imageIndex: null, reason: "" },

  // 任务启动模态框配置 (消除黑盒)
  jobModal: {
    open: false,
    type: "eval",
    dataset: "",
    backend: "",
    model: "",
    limit: "",
    failfast: false,
    match_mode: "exact",
    targets: "first",
    overwrite: false,
    scorer: "",
    eval_targets: "",
  },

  // 本地模型与全局设置
  models: { hf_models: [], mnn_models: [], llamacpp_models: [], hf_dir: null, mnn_dir: null, llamacpp_dir: null },
  settings: { workspace: "", media_root: "", image_strip_prefix: "", hf_models_dir: "", mnn_models_dir: "", llamacpp_models_dir: "" },

  // Sweep 跨集批量扫描
  sweep: {
    selectedDatasets: new Set(),
    backend: "openai",
    model: "",
    baseUrl: "",
    method: "",
    limit: "",
    stopOnError: false,
    overwrite: false,
    dryRun: false,
  },

  // Sweep 结果多维可视化看板
  sweepResults: {
    data: null,
    activeRunPath: "",
    runsList: [],
    runsLoadPromise: null,
    runsRequestId: 0,
    selectionRequestId: 0,
    selectedDatasetName: null,
    activeHeatmapField: null,
    filterMethod: "all",
    filterSearch: "",
    filterStatus: "all",
    sortBy: "default",
    subView: "cm", // "cm" | "pv"
  },

  // 可视化配置工坊
  configSubTab: "inference",
  configDirty: false,
  configData: { config: {}, raw: "", settableDoc: "" },

  // 任务队列
  jobs: [],
  currentJobId: null,
  terminalLogs: "",
  logDrawerOpen: false,
  autoScrollLogs: true,
  eventSource: null,
  jobLoadPromise: null,
  jobPollTimer: null,
  terminalSession: 0,
  terminalRetryTimer: null,

  // 评测结果与指标 (双模态: field-eval / eval)
  runs: [],
  selectedRunIndex: 0,
  selectedRun: null,
  activeRunMethod: "field-eval", // "field-eval" | "eval"

  // eval 模式数据
  runMetrics: null,
  scoredRecords: [],
  scoredTotal: 0,
  scoredOffset: 0,
  scoredLimit: 30,
  scoredOrder: "default",

  // field-eval 模式数据
  fieldMetrics: null,
  fieldMismatches: [],
  fieldMismatchesTotal: 0,
  fieldMismatchesOffset: 0,
  fieldMismatchesLimit: 20,
  fieldMismatchesFilter: "",

  // 回收站
  trashItems: [],

  // 健康体检
  healthReport: null,
  loaded: { datasets: false, settings: false, models: false },
  requests: { datasets: null, settings: null, models: null },
};

// 辅助工具函数
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function showToast(msg, type = "info") {
  const shelf = document.getElementById("toast-shelf");
  if (!shelf) return;
  const el = document.createElement("div");
  el.className = `toast-pill toast-${type}`;
  let icon = "ℹ️";
  if (type === "success") icon = "✅";
  if (type === "error") icon = "❌";
  if (type === "warning") icon = "⚠️";
  el.innerHTML = `<span>${icon}</span><span>${escapeHtml(msg)}</span>`;
  shelf.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateY(-10px)";
    el.style.transition = "all 0.25s ease";
    setTimeout(() => el.remove(), 250);
  }, 3500);
}

async function getApiError(response) {
  let detail = "";
  try {
    const body = await response.json();
    detail = typeof body.detail === "string" ? body.detail : body.detail?.message || body.message || "";
  } catch (_) {
    detail = await response.text().catch(() => "");
  }
  return detail || `请求失败 (HTTP ${response.status})`;
}

async function apiFetch(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await getApiError(response));
  return response;
}

// --------------------------------------------------------------------------
// 路由与 Tab 切换 (两层架构：全局视图 vs 数据集专属视图)
// --------------------------------------------------------------------------
const DATASET_SCOPED_TABS = ["gallery", "config", "runs", "health", "trash"];

function switchTab(tab, dataset = null, updateHash = true) {
  state.activeTab = tab;
  if (dataset) state.currentDataset = dataset;

  // 数据集专属页面若未选定数据集，默认选中第一个可用数据集
  if (DATASET_SCOPED_TABS.includes(tab) && !state.currentDataset && state.datasets.length > 0) {
    state.currentDataset = state.datasets[0].name;
  }

  if (updateHash) {
    if (state.currentDataset && DATASET_SCOPED_TABS.includes(tab)) {
      window.location.hash = `#/${tab}/${encodeURIComponent(state.currentDataset)}`;
    } else {
      window.location.hash = `#/${tab}`;
    }
  }

  // 更新顶部导航高亮
  document.querySelectorAll(".nav-tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });

  // 更新面板可见性
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `view-${tab}`);
  });

  updateHeaderDatasetDropdown();

  // 触发对应 Tab 数据获取
  if (tab === "datasets") loadDatasets();
  if (tab === "sweep") loadSweepData();
  if (tab === "sweep-results") loadSweepResultsData();
  if (tab === "jobs") {
    initInlineJobConsole();
    refreshJobs();
  }
  updateJobPolling();
  if (tab === "settings") loadSettings();
  if (tab === "gallery") loadSamples(0);
  if (tab === "config") loadConfig();
  if (tab === "runs") loadRuns();
  if (tab === "trash") loadTrash();
  if (tab === "health") loadHealth();
}

function handleHash() {
  const hash = window.location.hash || "#/datasets";
  const parts = hash.replace("#/", "").split("/");
  let tab = parts[0] || "datasets";
  if (tab === "global-config") tab = "settings";
  const ds = parts[1] ? decodeURIComponent(parts[1]) : state.currentDataset;
  if (["datasets", "sweep", "sweep-results", "jobs", "settings", "gallery", "config", "runs", "trash", "health"].includes(tab)) {
    switchTab(tab, ds, false);
  }
}

// --------------------------------------------------------------------------
// 全局统一数据集切换器 (随时随地在任何页面切换活动数据集)
// --------------------------------------------------------------------------
function toggleDatasetDropdown(forceClose = false) {
  const dd = document.getElementById("global-ds-dropdown");
  if (!dd) return;
  if (forceClose) {
    dd.classList.remove("open");
  } else {
    dd.classList.toggle("open");
    if (dd.classList.contains("open")) {
      const searchInput = document.getElementById("global-ds-search");
      if (searchInput) {
        searchInput.value = "";
        searchInput.focus();
      }
      renderDatasetDropdown();
    }
  }
}

function renderDatasetDropdown(filter = "") {
  const listEl = document.getElementById("global-ds-list");
  const countEl = document.getElementById("global-ds-count");
  if (!listEl) return;

  const query = filter.trim().toLowerCase();
  const filtered = state.datasets.filter((ds) => !query || ds.name.toLowerCase().includes(query));

  if (countEl) countEl.textContent = `${state.datasets.length} 个`;

  if (!filtered.length) {
    listEl.innerHTML = `<div style="padding: 1.25rem; text-align: center; color: var(--text-dim); font-size: 0.8rem;">无匹配数据集</div>`;
    return;
  }

  listEl.innerHTML = filtered
    .map((ds) => {
      const isSelected = ds.name === state.currentDataset;
      return `
      <div class="global-ds-item ${isSelected ? "selected" : ""}" onclick="selectGlobalDataset('${escapeHtml(ds.name)}')">
        <div style="display: flex; align-items: center; gap: 0.55rem; overflow: hidden;">
          <span style="font-size: 0.88rem;">📁</span>
          <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(ds.name)}</span>
        </div>
        <span class="global-ds-item-count">${ds.test_count} 样本</span>
      </div>`;
    })
    .join("");
}

function filterDatasetDropdown(query) {
  renderDatasetDropdown(query);
}

function selectGlobalDataset(name) {
  if (state.currentDataset === name) {
    toggleDatasetDropdown(true);
    return;
  }

  state.currentDataset = name;
  toggleDatasetDropdown(true);
  updateHeaderDatasetDropdown();

  // 若当前处于数据集专属页面，立即重新刷新当前页面数据
  if (DATASET_SCOPED_TABS.includes(state.activeTab)) {
    window.location.hash = `#/${state.activeTab}/${encodeURIComponent(name)}`;
    if (state.activeTab === "gallery") loadSamples(0);
    if (state.activeTab === "config") loadConfig();
    if (state.activeTab === "runs") loadRuns();
    if (state.activeTab === "health") loadHealth();
    if (state.activeTab === "trash") loadTrash();
  }

  showToast(`已切换当前活动数据集为: ${name}`, "info");
}

function updateHeaderDatasetDropdown() {
  const nameEl = document.getElementById("header-dataset-name");
  if (nameEl) {
    nameEl.textContent = state.currentDataset || "未选定";
  }
  // 同步更新页面中的各个局部数据集下拉框选择
  const galleryDs = document.getElementById("gallery-dataset-select");
  if (galleryDs && galleryDs.value !== state.currentDataset) galleryDs.value = state.currentDataset;
  const runsDs = document.getElementById("runs-dataset-select");
  if (runsDs && runsDs.value !== state.currentDataset) runsDs.value = state.currentDataset;
  const inlineDs = document.getElementById("job-inline-dataset");
  if (inlineDs && inlineDs.value !== state.currentDataset) inlineDs.value = state.currentDataset;
  const cfgBadge = document.getElementById("cfg-active-dataset-badge");
  if (cfgBadge) cfgBadge.textContent = state.currentDataset || "未选定";
  const cfgScopeDs = document.getElementById("cfg-scope-ds-name");
  if (cfgScopeDs) cfgScopeDs.textContent = state.currentDataset || "未选定";
}

function updateHeaderDatasetPill() {
  updateHeaderDatasetDropdown();
}

// 点击页面外部区域自动收起全局下拉框
document.addEventListener("click", (e) => {
  const dd = document.getElementById("global-ds-dropdown");
  if (dd && !dd.contains(e.target)) {
    dd.classList.remove("open");
  }
});

// --------------------------------------------------------------------------
// 数据集 (Datasets)
// --------------------------------------------------------------------------
async function loadDatasets(force = false) {
  if (!force && state.loaded.datasets) return state.datasets;
  if (!force && state.requests.datasets) return state.requests.datasets;
  state.requests.datasets = (async () => {
  try {
    const res = await apiFetch("/api/datasets");
    state.datasets = await res.json();
    if (!state.currentDataset && state.datasets.length > 0) {
      state.currentDataset = state.datasets[0].name;
    }
    updateHeaderDatasetDropdown();
    renderDatasetDropdown();
    renderDatasets();
    initInlineJobConsole();
    state.loaded.datasets = true;
    return state.datasets;
  } catch (err) {
    showToast(`获取数据集列表失败: ${err.message}`, "error");
  }
  })();
  try { return await state.requests.datasets; } finally { state.requests.datasets = null; }
}

function renderDatasets() {
  const container = document.getElementById("datasets-grid");
  if (!container) return;

  if (!state.datasets.length) {
    container.innerHTML = `
      <div class="empty-state-card">
        <div class="empty-state-icon">📂</div>
        <div class="empty-state-title">工作区暂无数据集</div>
        <div class="empty-state-desc">
          尚未检测到已初始化的数据集目录。<br>
          您可以在终端执行以下命令将测试集数据导入当前工作区：<br><br>
          <code>eval-vlm split --dataset your_dataset.json</code>
        </div>
        <button class="btn btn-primary" onclick="loadDatasets()">🔄 重新扫描</button>
      </div>`;
    return;
  }

  container.innerHTML = state.datasets
    .map((ds) => {
      const shaShort = ds.test_sha256 ? ds.test_sha256.slice(0, 8) : "—";
      return `
      <div class="dataset-card">
        <div>
          <div class="dataset-card-top">
            <span class="dataset-card-name">${escapeHtml(ds.name)}</span>
            ${
              ds.has_dirty_runs
                ? `<span class="badge-stale">存在过期 Run</span>`
                : `<span class="role-badge" style="background: var(--emerald-bg); border-color: var(--emerald-border); color: var(--emerald-500);">正常</span>`
            }
          </div>

          <div class="dataset-metrics-row">
            <div>
              <div class="metric-number" style="color: var(--cyan-500);">${ds.test_count}</div>
              <div class="metric-label">Test 样本</div>
            </div>
            <div>
              <div class="metric-number" style="color: var(--text-muted);">${ds.train_count}</div>
              <div class="metric-label">Train</div>
            </div>
            <div>
              <div class="metric-number" style="color: #a78bfa;">${ds.run_count}</div>
              <div class="metric-label">Runs 结果</div>
            </div>
          </div>

          <div style="font-size: 0.8rem; color: var(--text-dim); margin-bottom: 1rem;">
            <span>SHA: </span><code style="color: var(--text-muted);">${shaShort}</code>
          </div>
        </div>

        <div class="dataset-card-footer">
          <button class="btn btn-sm btn-primary" onclick="selectDatasetAndNavigate('${escapeHtml(ds.name)}', 'gallery')">
            🖼️ 浏览样本
          </button>
          <button class="btn btn-sm" onclick="selectDatasetAndNavigate('${escapeHtml(ds.name)}', 'config')">
            ⚙️ 配置
          </button>
          <button class="btn btn-sm" onclick="selectDatasetAndNavigate('${escapeHtml(ds.name)}', 'runs')">
            📊 评测结果
          </button>
          <button class="btn btn-sm" onclick="openDatasetHtmlModal('${escapeHtml(ds.name)}')" title="查看该数据集下的所有 HTML 报告文件">
            📑 HTML
          </button>
          <button class="btn btn-sm" onclick="selectDatasetAndNavigate('${escapeHtml(ds.name)}', 'health')">
            🩺 体检
          </button>
        </div>
      </div>
    `;
    })
    .join("");
}

function selectDatasetAndNavigate(name, tab) {
  state.currentDataset = name;
  updateHeaderDatasetDropdown();
  switchTab(tab, name);
}

// --------------------------------------------------------------------------
// 样本画廊 (Gallery)
// --------------------------------------------------------------------------
async function loadSamples(offset = 0) {
  if (!state.currentDataset) {
    const container = document.getElementById("gallery-grid");
    if (container) {
      container.innerHTML = `
        <div class="empty-state-card">
          <div class="empty-state-icon">🖼️</div>
          <div class="empty-state-title">未选定数据集</div>
          <div class="empty-state-desc">请先在【数据集】页面选择已有数据集，或通过 CLI 初始化一个测试集。</div>
          <button class="btn btn-primary" onclick="switchTab('datasets')">去选择数据集</button>
        </div>`;
    }
    return;
  }
  state.samplesOffset = offset;

  const q = new URLSearchParams({
    offset: state.samplesOffset,
    limit: state.samplesLimit,
  });
  if (state.samplesFilter) {
    q.append("filter", state.samplesFilter);
  }

  const container = document.getElementById("gallery-grid");
  if (container) {
    container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 4rem; color: var(--text-dim);">加载样本数据中...</div>`;
  }

  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/samples?${q.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.samples = data.samples;
    state.samplesTotal = data.total;
    state.testSha = data.test_sha256;
    updateHeaderDatasetDropdown();
    renderGallery();
  } catch (err) {
    showToast("加载测试样本失败", "error");
    if (container) {
      container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 4rem; color: #fb7185;">加载失败: ${escapeHtml(err.message)}</div>`;
    }
  }
}

function renderGallery() {
  const container = document.getElementById("gallery-grid");
  const counter = document.getElementById("gallery-counter");
  const prevBtn = document.getElementById("gallery-prev");
  const nextBtn = document.getElementById("gallery-next");
  const bottomPrev = document.getElementById("gallery-prev-bottom");
  const bottomNext = document.getElementById("gallery-next-bottom");
  const selectDs = document.getElementById("gallery-dataset-select");

  if (selectDs && state.datasets.length) {
    selectDs.innerHTML = state.datasets
      .map((d) => `<option value="${escapeHtml(d.name)}" ${d.name === state.currentDataset ? "selected" : ""}>${escapeHtml(d.name)}</option>`)
      .join("");
  }

  if (counter) {
    const from = state.samplesTotal === 0 ? 0 : state.samplesOffset + 1;
    const to = Math.min(state.samplesOffset + state.samplesLimit, state.samplesTotal);
    counter.textContent = `共 ${state.samplesTotal} 条 (当前第 ${from} - ${to} 条)`;
  }

  const canPrev = state.samplesOffset > 0;
  const canNext = state.samplesOffset + state.samplesLimit < state.samplesTotal;
  if (prevBtn) prevBtn.disabled = !canPrev;
  if (nextBtn) nextBtn.disabled = !canNext;
  if (bottomPrev) bottomPrev.disabled = !canPrev;
  if (bottomNext) bottomNext.disabled = !canNext;

  if (!container) return;

  if (!state.samples.length) {
    container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 4rem; color: var(--text-dim);">
      未检索到符合条件的测试样本
    </div>`;
    return;
  }

  container.innerHTML = state.samples
    .map((s) => {
      const hasMissing = s.images.some((im) => !im.exists);
      const isMultiImage = s.images.length > 1;

      // 渲染图片栏
      let imagesHtml = "";
      if (!s.images.length) {
        imagesHtml = `<div style="font-size: 0.8rem; color: var(--text-dim); padding: 1.5rem;">无关联图片 (纯文本样本)</div>`;
      } else {
        imagesHtml = s.images
          .map((img, imgIdx) => {
            if (!img.exists) {
              return `
              <div class="image-tile" title="${escapeHtml(img.ref)}">
                <div class="missing-image-box">
                  <span style="font-size: 1.1rem;">⚠️</span>
                  <span style="font-size: 0.72rem; font-weight: 700; margin-top: 2px;">图片丢失</span>
                  <span style="font-size: 0.65rem; word-break: break-all; opacity: 0.8;">${escapeHtml(img.ref)}</span>
                </div>
              </div>`;
            }
            return `
            <div class="image-tile" title="${escapeHtml(img.ref)}">
              <img src="${escapeHtml(img.url)}&thumb=1" loading="lazy" alt="${escapeHtml(img.ref)}" onclick="openLightbox('${escapeHtml(img.url)}', '${escapeHtml(img.ref)}')">
              <div class="image-tile-caption" title="${escapeHtml(img.ref)}">
                ${escapeHtml(img.ref)}
              </div>
              <div class="image-hover-actions">
                <button class="btn btn-sm" style="background: rgba(255,255,255,0.2); color: #fff; font-size: 0.7rem;" onclick="openLightbox('${escapeHtml(img.url)}', '${escapeHtml(img.ref)}')">
                  🔍 放大查看
                </button>
                ${
                  isMultiImage
                    ? `<button class="btn btn-sm btn-outline-danger" style="font-size: 0.7rem; padding: 2px 6px;" onclick="promptDelete('${escapeHtml(s.id)}', 'image', ${imgIdx})">
                         删此单图
                       </button>`
                    : ""
                }
              </div>
            </div>`;
          })
          .join("");
      }

      // 渲染样本ID下方的图片路径列表（直接可见，无需放大）
      let headImagesHtml = "";
      if (s.images && s.images.length) {
        headImagesHtml = `
        <div class="sample-head-images-list">
          ${s.images
            .map(
              (img, idx) => `
              <div class="sample-head-image-row ${!img.exists ? "is-missing" : ""}" title="点击复制路径" onclick="navigator.clipboard && navigator.clipboard.writeText('${escapeHtml(img.ref)}').then(() => showToast('已复制图片路径: ${escapeHtml(img.ref)}', 'info'))">
                <span class="sample-head-image-icon">🖼️${s.images.length > 1 ? ` <span class="sample-head-image-index">[${idx + 1}]</span>` : ""}</span>
                <span class="sample-head-image-path">${escapeHtml(img.ref)}</span>
                ${!img.exists ? '<span class="sample-head-image-missing">(丢失)</span>' : ""}
              </div>`
            )
            .join("")}
        </div>`;
      } else {
        headImagesHtml = `
        <div class="sample-head-images-list">
          <div class="sample-head-image-row is-empty">
            <span class="sample-head-image-icon">📝</span>
            <span class="sample-head-image-path" style="opacity: 0.65;">无关联图片 (纯文本样本)</span>
          </div>
        </div>`;
      }

      // 渲染对话
      const turnsHtml = s.turns
        .map((t, tIdx) => {
          const isTarget = s.targets.some((tgt) => tgt.turn_index === tIdx);
          return `
          <div class="turn-bubble">
            <div class="turn-role-badge">
              <span>${escapeHtml(t.role)}</span>
              ${isTarget ? `<span class="target-chip">评测目标</span>` : ""}
            </div>
            <div class="turn-text">${escapeHtml(t.content)}</div>
          </div>`;
        })
        .join("");

      return `
      <div class="sample-card ${hasMissing ? "has-missing-img" : ""}">
        <div class="sample-card-head">
          <div class="sample-card-head-main">
            <div class="sample-card-id-row">
              <span class="sample-index-badge">#${s.position}</span>
              <span class="sample-id-code">${escapeHtml(s.id)}</span>
            </div>
            ${headImagesHtml}
          </div>
          <button class="btn btn-sm btn-outline-danger" style="align-self: flex-start; margin-top: 2px;" onclick="promptDelete('${escapeHtml(s.id)}', 'record')">
            🗑️ 删除样本
          </button>
        </div>

        <div class="sample-images-shelf">
          ${imagesHtml}
        </div>

        <div class="sample-dialogue-flow">
          ${turnsHtml}
        </div>
      </div>
    `;
    })
    .join("");
}

// --------------------------------------------------------------------------
// 大图灯箱 (Lightbox)
// --------------------------------------------------------------------------
function openLightbox(url, ref) {
  state.lightbox = { open: true, url, ref };
  const modal = document.getElementById("lightbox-modal");
  const imgEl = document.getElementById("lightbox-img");
  const capEl = document.getElementById("lightbox-cap");
  if (modal && imgEl && capEl) {
    imgEl.src = url;
    capEl.textContent = ref;
    modal.showModal();
  }
}

function closeLightbox() {
  state.lightbox.open = false;
  const modal = document.getElementById("lightbox-modal");
  if (modal) modal.close();
}

// --------------------------------------------------------------------------
// 样本删除与回收站 (Core Sync)
// --------------------------------------------------------------------------
function promptDelete(sampleId, mode = "record", imageIndex = null) {
  state.deleteModal = {
    open: true,
    sampleId,
    mode,
    imageIndex,
    reason: "",
  };
  const modal = document.getElementById("delete-modal");
  const idEl = document.getElementById("del-target-id");
  const modeEl = document.getElementById("del-target-mode");
  const imgWarnEl = document.getElementById("del-image-warning");
  const reasonInput = document.getElementById("del-reason-input");

  if (idEl) idEl.textContent = sampleId;
  if (modeEl) modeEl.textContent = mode === "image" ? "删除单张图片" : "删除整条测试样本";
  if (imgWarnEl) {
    imgWarnEl.style.display = mode === "image" ? "block" : "none";
  }
  if (reasonInput) reasonInput.value = "";
  if (modal) modal.showModal();
}

function closeDeleteModal() {
  state.deleteModal.open = false;
  const modal = document.getElementById("delete-modal");
  if (modal) modal.close();
}

async function confirmDelete() {
  const { sampleId, mode, imageIndex } = state.deleteModal;
  const reasonInput = document.getElementById("del-reason-input");
  const reason = reasonInput ? reasonInput.value.trim() : "";

  try {
    const res = await fetch(
      `/api/datasets/${encodeURIComponent(state.currentDataset)}/samples/${encodeURIComponent(sampleId)}`,
      {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_sha256: state.testSha,
          reason,
          mode,
          image_index: imageIndex,
        }),
      }
    );

    if (res.status === 409) {
      showToast("test.json 已被其他操作修改，正在同步最新数据...", "warning");
      closeDeleteModal();
      await loadSamples(state.samplesOffset);
      return;
    }

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    closeDeleteModal();
    showToast(`样本 ${sampleId} 已成功删除，已进入回收站`, "success");
    await loadSamples(state.samplesOffset);
  } catch (err) {
    showToast(`删除失败: ${err.message}`, "error");
  }
}

async function loadTrash() {
  if (!state.currentDataset) return;
  const tbody = document.getElementById("trash-tbody");
  if (tbody) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:2rem; color:var(--text-dim);">加载回收站中...</td></tr>`;
  }
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/trash`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.trashItems = await res.json();
    renderTrash();
  } catch (err) {
    showToast("获取回收站失败", "error");
  }
}

function renderTrash() {
  const tbody = document.getElementById("trash-tbody");
  if (!tbody) return;

  if (!state.trashItems.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:3rem; color:var(--text-dim);">回收站为空</td></tr>`;
    return;
  }

  tbody.innerHTML = state.trashItems
    .map(
      (item) => `
    <tr>
      <td style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-dim);">${escapeHtml(item.trash_id)}</td>
      <td style="font-family: var(--font-mono); font-weight: 600;">${escapeHtml(item.sample_id)}</td>
      <td><span class="role-badge">${escapeHtml(item.mode)}</span></td>
      <td>#${item.position}</td>
      <td>${escapeHtml(item.deleted_by)}</td>
      <td>${escapeHtml(item.reason || "—")}</td>
      <td style="font-size: 0.8rem; color: var(--text-dim);">${new Date(item.ts).toLocaleString()}</td>
      <td>
        <button class="btn btn-sm btn-primary" onclick="restoreTrashItem('${escapeHtml(item.trash_id)}')">
          ↩️ 一键恢复
        </button>
      </td>
    </tr>
  `
    )
    .join("");
}

async function restoreTrashItem(trashId) {
  if (!confirm("确认恢复该样本吗？系统将插回原位置并同步修复 split_meta 索引。")) return;
  try {
    const res = await fetch(
      `/api/datasets/${encodeURIComponent(state.currentDataset)}/trash/${encodeURIComponent(trashId)}/restore`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_sha256: state.testSha }),
      }
    );

    if (res.status === 409) {
      showToast("test.json 在此期间已被修改，请刷新重试", "warning");
      await loadSamples(state.samplesOffset);
      return;
    }

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    showToast("样本已成功恢复！", "success");
    await loadTrash();
    await loadSamples(state.samplesOffset);
  } catch (err) {
    showToast(`恢复失败: ${err.message}`, "error");
  }
}

// --------------------------------------------------------------------------
// 本地模型管理 (Local Models Detection)
// --------------------------------------------------------------------------
async function loadModels(force = false) {
  if (!force && state.loaded.models) return state.models;
  if (!force && state.requests.models) return state.requests.models;
  state.requests.models = (async () => {
  try {
    const res = await apiFetch("/api/models");
    const data = await res.json();
    state.models = {
      hf_models: data.hf_models || [],
      mnn_models: data.mnn_models || [],
      llamacpp_models: data.llamacpp_models || [],
      hf_dir: data.hf_dir || null,
      mnn_dir: data.mnn_dir || null,
      llamacpp_dir: data.llamacpp_dir || null,
    };
    renderModelSelects();
    renderModelCards();
    updateGGUFConvertHfOptions();
    state.loaded.models = true;
    return state.models;
  } catch (err) {
    console.warn("加载本地模型失败:", err);
  }
  })();
  try { return await state.requests.models; } finally { state.requests.models = null; }
}

function renderModelSelects() {
  // 1. Task Queue (任务队列) 与 任务启动弹窗 模型下拉框
  const optHF = (state.models.hf_models || [])
    .map((m) => `<option value="${escapeHtml(m.path)}" data-type="hf" data-name="${escapeHtml(m.name)}" data-modelname="${escapeHtml(m.model_name || '')}">🤗 ${escapeHtml(m.name)}</option>`)
    .join("");
  const optMNN = (state.models.mnn_models || [])
    .map((m) => `<option value="${escapeHtml(m.path)}" data-type="mnn" data-name="${escapeHtml(m.name)}" data-modelname="${escapeHtml(m.model_name || '')}">⚡ ${escapeHtml(m.name)}</option>`)
    .join("");
  const optLlamaCpp = (state.models.llamacpp_models || [])
    .map((m) => `<option value="${escapeHtml(m.path)}" data-type="llamacpp" data-name="${escapeHtml(m.name)}" data-modelname="${escapeHtml(m.model_name || '')}" data-mmproj="${escapeHtml(m.mmproj_path || '')}">🦙 ${escapeHtml(m.name)}</option>`)
    .join("");

  const groupedModelOptions = `<option value="">-- 选择已探测模型 --</option>` +
    (optLlamaCpp ? `<optgroup label="🦙 llama.cpp GGUF 模型 (成对双 GGUF)">${optLlamaCpp}</optgroup>` : "") +
    (optHF ? `<optgroup label="🤗 HF / vLLM 模型 (本地权重)">${optHF}</optgroup>` : "") +
    (optMNN ? `<optgroup label="⚡ MNN 离线模型 (本地文件/目录)">${optMNN}</optgroup>` : "") +
    `<option value="__custom__">✏️ 自定义输入 / 路径</option>`;

  const jobModelSelect = document.getElementById("job-inline-model-select");
  if (jobModelSelect) {
    const curVal = jobModelSelect.value;
    jobModelSelect.innerHTML = groupedModelOptions;
    if (curVal) jobModelSelect.value = curVal;
  }

  const jobModalModelSelect = document.getElementById("job-modal-model-select");
  if (jobModalModelSelect) {
    const curVal = jobModalModelSelect.value;
    jobModalModelSelect.innerHTML = groupedModelOptions;
    if (curVal) jobModalModelSelect.value = curVal;
  }

  // 2. Sweep 页面模型下拉框
  const sweepSelect = document.getElementById("sweep-model-select");
  if (sweepSelect) {
    const backend = document.getElementById("sweep-backend")?.value || "openai";
    let options = `<option value="">(选择本地已探测模型)</option>`;
    if (backend === "mnn") {
      options += (state.models.mnn_models || [])
        .map((m) => `<option value="${escapeHtml(m.path)}" data-modelname="${escapeHtml(m.model_name || '')}">⚡ ${escapeHtml(m.name)}</option>`)
        .join("");
    } else if (backend === "llamacpp") {
      options += (state.models.llamacpp_models || [])
        .map((m) => `<option value="${escapeHtml(m.path)}" data-modelname="${escapeHtml(m.model_name || '')}" data-mmproj="${escapeHtml(m.mmproj_path || '')}">🦙 ${escapeHtml(m.name)}</option>`)
        .join("");
    } else {
      options += (state.models.hf_models || [])
        .map((m) => `<option value="${escapeHtml(m.path)}">🤗 ${escapeHtml(m.name)}</option>`)
        .join("");
    }
    sweepSelect.innerHTML = options;
  }

  // 3. Config 页面 各后端模型下拉框
  const cfgOpenAISelect = document.getElementById("cfg-form-openai-model-select");
  if (cfgOpenAISelect) {
    cfgOpenAISelect.innerHTML = `<option value="">(选择本地已探测模型)</option>` +
      (state.models.hf_models || [])
        .map((m) => `<option value="${escapeHtml(m.name)}">🤗 ${escapeHtml(m.name)}</option>`)
        .join("");
  }

  const cfgMNNSelect = document.getElementById("cfg-form-mnn-model-select");
  if (cfgMNNSelect) {
    cfgMNNSelect.innerHTML = `<option value="">(选择本地已探测 MNN 模型)</option>` +
      (state.models.mnn_models || [])
        .map((m) => `<option value="${escapeHtml(m.path)}">⚡ ${escapeHtml(m.name)}</option>`)
        .join("");
  }

  const cfgVLLMSelect = document.getElementById("cfg-form-vllm-model-select");
  if (cfgVLLMSelect) {
    cfgVLLMSelect.innerHTML = `<option value="">(从已探测模型中选取)</option>` +
      (state.models.hf_models || [])
        .map((m) => `<option value="${escapeHtml(m.path)}">🤗 ${escapeHtml(m.name)}</option>`)
        .join("");
  }

  const cfgHFSelect = document.getElementById("cfg-form-hf-model-select");
  if (cfgHFSelect) {
    cfgHFSelect.innerHTML = `<option value="">(从已探测模型中选取)</option>` +
      (state.models.hf_models || [])
        .map((m) => `<option value="${escapeHtml(m.path)}">🤗 ${escapeHtml(m.name)}</option>`)
        .join("");
  }

  const cfgLlamaCppSelect = document.getElementById("cfg-form-llamacpp-model-select");
  if (cfgLlamaCppSelect) {
    cfgLlamaCppSelect.innerHTML = `<option value="">(从已探测 llamacpp 成对模型中选择)</option>` +
      (state.models.llamacpp_models || [])
        .map((m) => `<option value="${escapeHtml(m.path)}" data-mmproj="${escapeHtml(m.mmproj_path || '')}" data-modelname="${escapeHtml(m.model_name || '')}">🦙 ${escapeHtml(m.name)}</option>`)
        .join("");
  }
}

function renderModelCards() {
  const hfBadge = document.getElementById("settings-scan-badge-hf");
  if (hfBadge) hfBadge.textContent = `HF/vLLM: ${(state.models.hf_models || []).length} 个`;
  const mnnBadge = document.getElementById("settings-scan-badge-mnn");
  if (mnnBadge) mnnBadge.textContent = `MNN: ${(state.models.mnn_models || []).length} 个`;
  const llamacppBadge = document.getElementById("settings-scan-badge-llamacpp");
  if (llamacppBadge) llamacppBadge.textContent = `llama.cpp(GGUF): ${(state.models.llamacpp_models || []).length} 个`;
}


// --------------------------------------------------------------------------
// 全局设置 (Global Settings)
// --------------------------------------------------------------------------
async function loadSettings(force = false) {
  if (!force && state.loaded.settings) return state.settings;
  if (!force && state.requests.settings) return state.requests.settings;
  state.requests.settings = (async () => {
  try {
    const res = await apiFetch("/api/settings");
    const cfg = await res.json();
    state.settings = cfg;

    const wsIn = document.getElementById("settings-workspace");
    const mrIn = document.getElementById("settings-mediaroot");
    const spIn = document.getElementById("settings-strip-prefix");
    const hfIn = document.getElementById("settings-hf-dir");
    const mnnIn = document.getElementById("settings-mnn-dir");
    const llamacppIn = document.getElementById("settings-llamacpp-dir");
    const trIn = document.getElementById("settings-train-out");
    const vaIn = document.getElementById("settings-val-out");
    const teIn = document.getElementById("settings-test-out");
    const pathEl = document.getElementById("settings-file-path");
    const rawEl = document.getElementById("settings-raw-yaml");

    const spTrain = document.getElementById("settings-split-train");
    const spTest = document.getElementById("settings-split-test");
    const spVal = document.getElementById("settings-split-val");
    const spSeed = document.getElementById("settings-split-seed");
    const spStratify = document.getElementById("settings-split-stratify");

    if (wsIn) wsIn.value = cfg.workspace || "";
    if (mrIn) mrIn.value = cfg.media_root || "";
    if (spIn) spIn.value = cfg.image_strip_prefix || "";
    if (hfIn) {
      if (Array.isArray(cfg.hf_models_dir)) {
        hfIn.value = cfg.hf_models_dir.join("\n");
      } else {
        hfIn.value = cfg.hf_models_dir || "";
      }
    }
    if (mnnIn) {
      if (Array.isArray(cfg.mnn_models_dir)) {
        mnnIn.value = cfg.mnn_models_dir.join("\n");
      } else {
        mnnIn.value = cfg.mnn_models_dir || "";
      }
    }
    if (llamacppIn) {
      if (Array.isArray(cfg.llamacpp_models_dir)) {
        llamacppIn.value = cfg.llamacpp_models_dir.join("\n");
      } else {
        llamacppIn.value = cfg.llamacpp_models_dir || "";
      }
    }
    if (trIn) trIn.value = cfg.train_out_dir || "";
    if (vaIn) vaIn.value = cfg.val_out_dir || "";
    if (teIn) teIn.value = cfg.test_out_dir || "";
    if (pathEl && cfg.config_file) pathEl.textContent = cfg.config_file;
    if (rawEl && cfg.raw_yaml) rawEl.textContent = cfg.raw_yaml;

    if (cfg.split) {
      if (spTrain) spTrain.value = cfg.split.train ?? 0.95;
      if (spTest) spTest.value = cfg.split.test ?? 0.05;
      if (spVal) spVal.value = cfg.split.val ?? 0.0;
      if (spSeed) spSeed.value = cfg.split.seed ?? 42;
      if (spStratify) spStratify.value = cfg.split.stratify_by || "";
    }

    state.loaded.settings = true;
    return state.settings;
  } catch (err) {
    showToast(`获取全局设置失败: ${err.message}`, "error");
  }
  })();
  try { return await state.requests.settings; } finally { state.requests.settings = null; }
}

async function saveSettings() {
  try {
    const splitPayload = {
      train: parseFloat(document.getElementById("settings-split-train")?.value) || 0.95,
      test: parseFloat(document.getElementById("settings-split-test")?.value) || 0.05,
      val: parseFloat(document.getElementById("settings-split-val")?.value) || 0.0,
      seed: parseInt(document.getElementById("settings-split-seed")?.value) || 42,
      stratify_by: document.getElementById("settings-split-stratify")?.value.trim() || null,
    };

    const payload = {
      workspace: document.getElementById("settings-workspace")?.value.trim() || undefined,
      media_root: document.getElementById("settings-mediaroot")?.value.trim() || undefined,
      image_strip_prefix: document.getElementById("settings-strip-prefix")?.value.trim() || "",
      hf_models_dir: document.getElementById("settings-hf-dir")?.value.trim() || "",
      mnn_models_dir: document.getElementById("settings-mnn-dir")?.value.trim() || "",
      llamacpp_models_dir: document.getElementById("settings-llamacpp-dir")?.value.trim() || "",
      train_out_dir: document.getElementById("settings-train-out")?.value.trim() || "",
      val_out_dir: document.getElementById("settings-val-out")?.value.trim() || "",
      test_out_dir: document.getElementById("settings-test-out")?.value.trim() || "",
      split: splitPayload,
    };

    const res = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.settings = data;
    showToast("全局配置 (~/.eval_vlm/config.yaml) 保存成功并已生效！", "success");
    state.loaded.settings = false;
    state.loaded.models = false;
    await Promise.all([loadSettings(true), loadModels(true)]);
  } catch (err) {
    showToast(`保存全局设置失败: ${err.message}`, "error");
  }
}

// --------------------------------------------------------------------------
// 批量扫描 (Sweep Studio)
// --------------------------------------------------------------------------
async function loadSweepData() {
  await Promise.all([loadDatasets(), loadModels()]);

  if (state.sweep.selectedDatasets.size === 0 && state.datasets.length > 0) {
    // 默认全选工作区内所有数据集
    state.datasets.forEach((d) => state.sweep.selectedDatasets.add(d.name));
  }

  renderSweepDatasets();
  updateSweepCmdPreview();
}

function renderSweepDatasets(filter = "") {
  const listEl = document.getElementById("sweep-datasets-list");
  const countEl = document.getElementById("sweep-selected-count");
  if (!listEl) return;

  const query = (filter || document.getElementById("sweep-ds-search")?.value || "").trim().toLowerCase();
  const filtered = state.datasets.filter((d) => !query || d.name.toLowerCase().includes(query));

  if (countEl) {
    countEl.textContent = `已选择 ${state.sweep.selectedDatasets.size} / 共 ${state.datasets.length} 个数据集`;
  }

  if (!filtered.length) {
    listEl.innerHTML = `<div style="padding: 2rem; text-align: center; color: var(--text-dim); font-size: 0.82rem;">无匹配数据集</div>`;
    return;
  }

  listEl.innerHTML = filtered
    .map((ds) => {
      const isChecked = state.sweep.selectedDatasets.has(ds.name);
      return `
      <div class="sweep-ds-item ${isChecked ? "checked" : ""}" onclick="toggleSweepDataset('${escapeHtml(ds.name)}')">
        <div class="sweep-ds-info">
          <input type="checkbox" class="sweep-ds-checkbox" ${isChecked ? "checked" : ""} onclick="event.stopPropagation(); toggleSweepDataset('${escapeHtml(ds.name)}');">
          <span class="sweep-ds-title">${escapeHtml(ds.name)}</span>
        </div>
        <div class="sweep-ds-meta">
          <span>${ds.test_count} 样本</span>
          ${ds.has_dirty_runs ? `<span class="badge-stale">有过期Run</span>` : ""}
        </div>
      </div>`;
    })
    .join("");
}

function toggleSweepDataset(name) {
  if (state.sweep.selectedDatasets.has(name)) {
    state.sweep.selectedDatasets.delete(name);
  } else {
    state.sweep.selectedDatasets.add(name);
  }
  renderSweepDatasets();
  updateSweepCmdPreview();
}

function toggleAllSweepDatasets(selectAll) {
  state.sweep.selectedDatasets.clear();
  if (selectAll) {
    state.datasets.forEach((d) => state.sweep.selectedDatasets.add(d.name));
  }
  renderSweepDatasets();
  updateSweepCmdPreview();
}

function invertSweepDatasets() {
  state.datasets.forEach((d) => {
    if (state.sweep.selectedDatasets.has(d.name)) {
      state.sweep.selectedDatasets.delete(d.name);
    } else {
      state.sweep.selectedDatasets.add(d.name);
    }
  });
  renderSweepDatasets();
  updateSweepCmdPreview();
}

function filterSweepDatasets(q) {
  renderSweepDatasets(q);
}

function onSweepBackendChange() {
  renderModelSelects();
  const backend = document.getElementById("sweep-backend")?.value || "openai";
  const urlRow = document.getElementById("sweep-baseurl-row");
  if (urlRow) {
    urlRow.style.display = (backend === "openai" || backend === "vllm") ? "block" : "none";
  }
  updateSweepCmdPreview();
}

function onSweepModelSelect(val) {
  if (val) {
    const sel = document.getElementById("sweep-model-select");
    const opt = sel?.options[sel.selectedIndex];
    const backend = document.getElementById("sweep-backend")?.value || "openai";
    const input = document.getElementById("sweep-model-input");
    if (backend === "llamacpp") {
      const modelName = opt?.getAttribute("data-modelname");
      const mmprojPath = opt?.getAttribute("data-mmproj");
      if (input) {
        input.value = modelName || val;
        input.setAttribute("data-modelpath", val);
        if (mmprojPath) input.setAttribute("data-mmproj", mmprojPath);
        else input.removeAttribute("data-mmproj");
      }
    } else {
      if (input) {
        input.value = val;
        input.removeAttribute("data-modelpath");
        input.removeAttribute("data-mmproj");
      }
    }
  }
  updateSweepCmdPreview();
}

function updateSweepCmdPreview() {
  const backend = document.getElementById("sweep-backend")?.value || "";
  const modelInput = document.getElementById("sweep-model-input");
  const model = modelInput?.value.trim() || "";
  const baseUrl = document.getElementById("sweep-base-url")?.value.trim() || "";
  const method = document.getElementById("sweep-method")?.value || "";
  const limit = document.getElementById("sweep-limit")?.value.trim() || "";
  const stopOnError = document.getElementById("sweep-stop-on-error")?.checked;
  const overwrite = document.getElementById("sweep-overwrite")?.checked;
  const dryRun = document.getElementById("sweep-dry-run")?.checked;

  const selectedList = Array.from(state.sweep.selectedDatasets);
  const dsParam = selectedList.length === state.datasets.length ? "" : selectedList.join(",");

  const parts = ["eval-vlm", "sweep"];
  if (dsParam) {
    parts.push("-d", dsParam);
  }
  if (backend) parts.push("--backend", backend);
  if (model) parts.push("--model", model.includes(" ") ? `"${model}"` : model);
  if (baseUrl) parts.push("--base-url", baseUrl);
  if (method) parts.push("--method", method);
  if (limit) parts.push("--limit", limit);
  if (stopOnError) parts.push("--stop-on-error");
  if (overwrite) parts.push("--overwrite");
  if (dryRun) parts.push("--dry-run");

  const cmdEl = document.getElementById("sweep-cmd-preview");
  if (cmdEl) cmdEl.textContent = parts.join(" ");

  const summaryEl = document.getElementById("sweep-summary-path");
  if (summaryEl) {
    const safeModel = (model || "<model>").replace(/[\\/:]/g, "_");
    const safeBackend = backend || "<backend>";
    summaryEl.textContent = `<workspace>/_sweep/${safeModel}/${safeBackend}/summary.md`;
  }
}

async function submitSweepJob() {
  if (state.sweep.selectedDatasets.size === 0) {
    showToast("请至少勾选 1 个待测试集", "warning");
    return;
  }

  const backend = document.getElementById("sweep-backend")?.value || "";
  const model = document.getElementById("sweep-model-input")?.value.trim() || "";
  const baseUrl = document.getElementById("sweep-base-url")?.value.trim() || "";
  const method = document.getElementById("sweep-method")?.value || "";
  const limit = document.getElementById("sweep-limit")?.value.trim() || "";
  const stopOnError = document.getElementById("sweep-stop-on-error")?.checked;
  const overwrite = document.getElementById("sweep-overwrite")?.checked;
  const dryRun = document.getElementById("sweep-dry-run")?.checked;

  const params = {};
  if (backend) params.backend = backend;
  if (model) params.model = model;
  if (baseUrl) params.base_url = baseUrl;
  if (method) params.method = method;
  if (limit) params.limit = limit;
  if (stopOnError) params.stop_on_error = true;
  if (overwrite) params.overwrite = true;
  if (dryRun) params.dry_run = true;

  const datasetParam = Array.from(state.sweep.selectedDatasets).join(",");

  try {
    const res = await fetch("/api/sweep/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "sweep",
        dataset: datasetParam,
        params,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const job = await res.json();
    showToast(`Sweep 批量扫描任务已成功提交入队 (ID: ${job.id})`, "success");
    switchTab("jobs");
    await refreshJobs();
    await openTerminal(job.id);
  } catch (err) {
    showToast(`提交 Sweep 任务失败: ${err.message}`, "error");
  }
}

// --------------------------------------------------------------------------
// 可视化配置工坊 (Config Studio)
// --------------------------------------------------------------------------
function switchConfigSubTab(subtab) {
  state.configSubTab = subtab;
  document.querySelectorAll(".config-subnav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.subtab === subtab);
  });
  document.querySelectorAll(".config-section").forEach((sec) => {
    sec.classList.toggle("active", sec.id === `cfg-sub-${subtab}`);
  });
}

function setConfigBackend(backend) {
  document.querySelectorAll(".config-backend-card").forEach((card) => {
    card.classList.toggle("active", card.id === `cfg-card-backend-${backend}`);
  });
  const grpOpenAI = document.getElementById("cfg-group-openai");
  const grpMNN = document.getElementById("cfg-group-mnn");
  const grpVLLM = document.getElementById("cfg-group-vllm_offline");
  const grpHF = document.getElementById("cfg-group-hf");
  const grpLlamaCpp = document.getElementById("cfg-group-llamacpp");
  if (grpOpenAI) grpOpenAI.style.display = (backend === "openai" || backend === "vllm" || backend === "fake") ? "block" : "none";
  if (grpMNN) grpMNN.style.display = backend === "mnn" ? "block" : "none";
  if (grpVLLM) grpVLLM.style.display = backend === "vllm_offline" ? "block" : "none";
  if (grpHF) grpHF.style.display = backend === "hf" ? "block" : "none";
  if (grpLlamaCpp) grpLlamaCpp.style.display = backend === "llamacpp" ? "block" : "none";
  markConfigDirty();
}

function markConfigDirty() {
  state.configDirty = true;
  const ind = document.getElementById("cfg-dirty-indicator");
  if (ind) ind.style.display = "inline-block";
}

function onConfigEvalTargetsChange() {
  const sel = document.getElementById("cfg-form-eval-targets");
  const customInput = document.getElementById("cfg-form-eval-targets-custom");
  if (!sel || !customInput) return;
  if (sel.value === "__custom__") {
    customInput.style.display = "block";
    customInput.focus();
  } else {
    customInput.style.display = "none";
  }
  markConfigDirty();
}


async function loadConfig() {
  if (!state.currentDataset) {
    if (state.datasets.length > 0) {
      state.currentDataset = state.datasets[0].name;
      updateHeaderDatasetDropdown();
    } else {
      showToast("请先在工作区创建或选择数据集", "warning");
      return;
    }
  }

  const badge = document.getElementById("cfg-active-dataset-badge");
  if (badge) badge.textContent = state.currentDataset;

  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/config`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.configData = await res.json();
    state.configDirty = false;
    const ind = document.getElementById("cfg-dirty-indicator");
    if (ind) ind.style.display = "none";
    renderConfig();
    await loadModels();
  } catch (err) {
    showToast(`读取配置文件失败: ${err.message}`, "error");
  }
}

function renderConfig() {
  const cfg = state.configData.config || {};
  const rawPre = document.getElementById("config-raw-pre");
  if (rawPre) rawPre.textContent = state.configData.raw || "";

  // 1. 推理后端
  const backend = cfg.inference?.backend || "openai";
  setConfigBackend(backend);

  // OpenAI / vLLM 字段
  const oai = cfg.inference?.openai || {};
  const oaiModel = document.getElementById("cfg-form-openai-model");
  const oaiBaseUrl = document.getElementById("cfg-form-openai-baseurl");
  const oaiApiKeyEnv = document.getElementById("cfg-form-openai-apikeyenv");
  const oaiConcurrency = document.getElementById("cfg-form-openai-concurrency");
  const oaiMaxTokens = document.getElementById("cfg-form-openai-maxtokens");
  const oaiTemp = document.getElementById("cfg-form-openai-temp");
  const oaiTempSlider = document.getElementById("cfg-form-openai-temp-slider");
  const oaiTopP = document.getElementById("cfg-form-openai-topp");
  const oaiTimeout = document.getElementById("cfg-form-openai-timeout");
  const oaiMaxRetries = document.getElementById("cfg-form-openai-maxretries");
  const oaiImageDetail = document.getElementById("cfg-form-openai-imagedetail");
  const oaiSysPrompt = document.getElementById("cfg-form-openai-sysprompt");

  if (oaiModel) oaiModel.value = oai.model || "";
  if (oaiBaseUrl) oaiBaseUrl.value = oai.base_url || "";
  if (oaiApiKeyEnv) oaiApiKeyEnv.value = oai.api_key_env || "OPENAI_API_KEY";
  if (oaiConcurrency) oaiConcurrency.value = oai.max_concurrency ?? 8;
  if (oaiMaxTokens) oaiMaxTokens.value = oai.max_tokens ?? 512;
  if (oaiTemp) oaiTemp.value = oai.temperature ?? 0.0;
  if (oaiTempSlider) oaiTempSlider.value = oai.temperature ?? 0.0;
  if (oaiTopP) oaiTopP.value = oai.top_p ?? 1.0;
  if (oaiTimeout) oaiTimeout.value = oai.request_timeout ?? 120.0;
  if (oaiMaxRetries) oaiMaxRetries.value = oai.max_retries ?? 3;
  if (oaiImageDetail) oaiImageDetail.value = oai.image_detail || "high";
  if (oaiSysPrompt) oaiSysPrompt.value = oai.system_prompt || "";

  // MNN 字段
  const mnn = cfg.inference?.mnn || {};
  const mnnConfigPath = document.getElementById("cfg-form-mnn-configpath");
  const mnnMaxTokens = document.getElementById("cfg-form-mnn-maxtokens");
  const mnnTemp = document.getElementById("cfg-form-mnn-temp");
  const mnnTempSlider = document.getElementById("cfg-form-mnn-temp-slider");
  const mnnTopP = document.getElementById("cfg-form-mnn-topp");
  const mnnTopK = document.getElementById("cfg-form-mnn-topk");
  const mnnRepPenalty = document.getElementById("cfg-form-mnn-reppenalty");
  const mnnFreqPenalty = document.getElementById("cfg-form-mnn-freqpenalty");
  const mnnPresPenalty = document.getElementById("cfg-form-mnn-prespenalty");
  const mnnPenWindow = document.getElementById("cfg-form-mnn-penwindow");
  const mnnMaxSide = document.getElementById("cfg-form-mnn-maxside");
  const mnnImgMaxPixels = document.getElementById("cfg-form-mnn-imgmaxpixels");
  const mnnImgMinPixels = document.getElementById("cfg-form-mnn-imgminpixels");
  const mnnQuant = document.getElementById("cfg-form-mnn-quant");
  const mnnSysPrompt = document.getElementById("cfg-form-mnn-sysprompt");

  if (mnnConfigPath) mnnConfigPath.value = mnn.config_path || "";
  if (mnnMaxTokens) mnnMaxTokens.value = mnn.max_tokens ?? 1024;
  if (mnnTemp) mnnTemp.value = (mnn.temperature !== null && mnn.temperature !== undefined) ? mnn.temperature : 0.0;
  if (mnnTempSlider) mnnTempSlider.value = (mnn.temperature !== null && mnn.temperature !== undefined) ? mnn.temperature : 0.0;
  if (mnnTopP) mnnTopP.value = (mnn.top_p !== null && mnn.top_p !== undefined) ? mnn.top_p : "";
  if (mnnTopK) mnnTopK.value = (mnn.top_k !== null && mnn.top_k !== undefined) ? mnn.top_k : "";
  if (mnnRepPenalty) mnnRepPenalty.value = mnn.repetition_penalty ?? 1.1;
  if (mnnFreqPenalty) mnnFreqPenalty.value = mnn.frequency_penalty ?? 0.0;
  if (mnnPresPenalty) mnnPresPenalty.value = mnn.presence_penalty ?? 0.0;
  if (mnnPenWindow) mnnPenWindow.value = mnn.penalty_window ?? 0;
  if (mnnMaxSide) mnnMaxSide.value = mnn.image_max_side ?? 2048;
  if (mnnImgMaxPixels) mnnImgMaxPixels.value = mnn.image_max_pixels ?? 589824;
  if (mnnImgMinPixels) mnnImgMinPixels.value = mnn.image_min_pixels ?? 1024;
  if (mnnQuant) mnnQuant.value = mnn.quant || "";
  if (mnnSysPrompt) mnnSysPrompt.value = mnn.system_prompt || "";

  // vLLM Offline 字段
  const vllm = cfg.inference?.vllm_offline || {};
  const vllmModel = document.getElementById("cfg-form-vllm-model");
  const vllmGpuUtil = document.getElementById("cfg-form-vllm-gpuutil");
  const vllmGpuUtilSlider = document.getElementById("cfg-form-vllm-gpuutil-slider");
  const vllmMaxModelLen = document.getElementById("cfg-form-vllm-maxmodellen");
  const vllmMaxNumSeqs = document.getElementById("cfg-form-vllm-maxnumseqs");
  const vllmMaxBatchedTokens = document.getElementById("cfg-form-vllm-maxbatchedtokens");
  const vllmMaxTokens = document.getElementById("cfg-form-vllm-maxtokens");
  const vllmTemp = document.getElementById("cfg-form-vllm-temp");
  const vllmTempSlider = document.getElementById("cfg-form-vllm-temp-slider");
  const vllmTopP = document.getElementById("cfg-form-vllm-topp");
  const vllmTopK = document.getElementById("cfg-form-vllm-topk");
  const vllmRepPenalty = document.getElementById("cfg-form-vllm-reppenalty");
  const vllmDtype = document.getElementById("cfg-form-vllm-dtype");
  const vllmMaxImages = document.getElementById("cfg-form-vllm-maximages");
  const vllmImgMinPixels = document.getElementById("cfg-form-vllm-imgminpixels");
  const vllmImgMaxPixels = document.getElementById("cfg-form-vllm-imgmaxpixels");
  const vllmRemoteCode = document.getElementById("cfg-form-vllm-remotecode");
  const vllmSysPrompt = document.getElementById("cfg-form-vllm-sysprompt");

  if (vllmModel) vllmModel.value = vllm.model_path || "";
  if (vllmGpuUtil) vllmGpuUtil.value = vllm.gpu_memory_utilization ?? 0.9;
  if (vllmGpuUtilSlider) vllmGpuUtilSlider.value = vllm.gpu_memory_utilization ?? 0.9;
  if (vllmMaxModelLen) vllmMaxModelLen.value = vllm.max_model_len ?? 4096;
  if (vllmMaxNumSeqs) vllmMaxNumSeqs.value = vllm.max_num_seqs ?? 128;
  if (vllmMaxBatchedTokens) vllmMaxBatchedTokens.value = vllm.max_num_batched_tokens ?? 20480;
  if (vllmMaxTokens) vllmMaxTokens.value = vllm.max_tokens ?? 512;
  if (vllmTemp) vllmTemp.value = vllm.temperature ?? 0.0;
  if (vllmTempSlider) vllmTempSlider.value = vllm.temperature ?? 0.0;
  if (vllmTopP) vllmTopP.value = vllm.top_p ?? 1.0;
  if (vllmTopK) vllmTopK.value = vllm.top_k ?? -1;
  if (vllmRepPenalty) vllmRepPenalty.value = vllm.repetition_penalty ?? 1.0;
  if (vllmDtype) vllmDtype.value = vllm.dtype || "auto";
  if (vllmMaxImages) vllmMaxImages.value = vllm.max_images_per_prompt ?? 4;
  if (vllmImgMinPixels) vllmImgMinPixels.value = vllm.image_min_pixels ?? 784;
  if (vllmImgMaxPixels) vllmImgMaxPixels.value = vllm.image_max_pixels ?? 564480;
  if (vllmRemoteCode) vllmRemoteCode.checked = vllm.trust_remote_code !== false;
  if (vllmSysPrompt) vllmSysPrompt.value = vllm.system_prompt || "";

  // HF 字段
  const hf = cfg.inference?.hf || {};
  const hfModel = document.getElementById("cfg-form-hf-model");
  const hfDevice = document.getElementById("cfg-form-hf-device");
  const hfDtype = document.getElementById("cfg-form-hf-dtype");
  const hfMaxTokens = document.getElementById("cfg-form-hf-maxtokens");
  const hfAttn = document.getElementById("cfg-form-hf-attn");
  const hfMaxSide = document.getElementById("cfg-form-hf-maxside");
  const hfImgMinPixels = document.getElementById("cfg-form-hf-imgminpixels");
  const hfImgMaxPixels = document.getElementById("cfg-form-hf-imgmaxpixels");
  const hfGreedy = document.getElementById("cfg-form-hf-greedy");
  const hfSysPrompt = document.getElementById("cfg-form-hf-sysprompt");

  if (hfModel) hfModel.value = hf.model_path || "";
  if (hfDevice) hfDevice.value = hf.device || "auto";
  if (hfDtype) hfDtype.value = hf.dtype || "auto";
  if (hfMaxTokens) hfMaxTokens.value = hf.max_tokens ?? 1024;
  if (hfAttn) hfAttn.value = hf.attn_implementation || "";
  if (hfMaxSide) hfMaxSide.value = hf.image_max_side ?? 0;
  if (hfImgMinPixels) hfImgMinPixels.value = hf.image_min_pixels ?? 1024;
  if (hfImgMaxPixels) hfImgMaxPixels.value = hf.image_max_pixels ?? 1048576;
  if (hfGreedy) hfGreedy.checked = hf.greedy !== false;
  if (hfSysPrompt) hfSysPrompt.value = hf.system_prompt || "";

  // llama.cpp 字段
  const lcpp = cfg.inference?.llamacpp || {};
  const lcppMode = document.getElementById("cfg-form-llamacpp-mode");
  const lcppBaseUrl = document.getElementById("cfg-form-llamacpp-baseurl");
  const lcppModelPath = document.getElementById("cfg-form-llamacpp-modelpath");
  const lcppMmprojPath = document.getElementById("cfg-form-llamacpp-mmprojpath");
  const lcppCliBinary = document.getElementById("cfg-form-llamacpp-clibinary");
  const lcppMaxTokens = document.getElementById("cfg-form-llamacpp-maxtokens");
  const lcppTemp = document.getElementById("cfg-form-llamacpp-temp");
  const lcppTopP = document.getElementById("cfg-form-llamacpp-topp");
  const lcppRepPenalty = document.getElementById("cfg-form-llamacpp-reppenalty");
  const lcppConcurrency = document.getElementById("cfg-form-llamacpp-concurrency");
  const lcppMaxSide = document.getElementById("cfg-form-llamacpp-maxside");
  const lcppSysPrompt = document.getElementById("cfg-form-llamacpp-sysprompt");

  if (lcppMode) lcppMode.value = lcpp.mode || "server";
  if (lcppBaseUrl) lcppBaseUrl.value = lcpp.base_url || "http://127.0.0.1:8080/v1";
  if (lcppModelPath) lcppModelPath.value = lcpp.model_path || "";
  if (lcppMmprojPath) lcppMmprojPath.value = lcpp.mmproj_path || "";
  if (lcppCliBinary) lcppCliBinary.value = lcpp.cli_binary || "";
  if (lcppMaxTokens) lcppMaxTokens.value = lcpp.max_tokens ?? 512;
  if (lcppTemp) lcppTemp.value = (lcpp.temperature !== null && lcpp.temperature !== undefined) ? lcpp.temperature : 0.0;
  if (lcppTopP) lcppTopP.value = (lcpp.top_p !== null && lcpp.top_p !== undefined) ? lcpp.top_p : 1.0;
  if (lcppRepPenalty) lcppRepPenalty.value = lcpp.repetition_penalty ?? 1.0;
  if (lcppConcurrency) lcppConcurrency.value = lcpp.max_concurrency ?? 4;
  if (lcppMaxSide) lcppMaxSide.value = lcpp.image_max_side ?? 2048;
  if (lcppSysPrompt) lcppSysPrompt.value = lcpp.system_prompt || "";
  onLlamaCppModeChange();

  // 2. 评测策略
  const evMethod = document.getElementById("cfg-form-eval-method");
  const evScorer = document.getElementById("cfg-form-scoring-scorer");
  const evTargets = document.getElementById("cfg-form-eval-targets");
  const evContext = document.getElementById("cfg-form-eval-context");

  if (evMethod) evMethod.value = cfg.eval?.method || "field-eval";
  if (evScorer) evScorer.value = cfg.scoring?.scorer || "exact_match";
  const evTargetsCustom = document.getElementById("cfg-form-eval-targets-custom");
  const rawTargetsVal = cfg.eval?.targets !== undefined && cfg.eval?.targets !== null ? String(cfg.eval.targets).trim() : "first";
  if (evTargets) {
    if (["first", "all", "last", "1", "2", "3"].includes(rawTargetsVal)) {
      evTargets.value = rawTargetsVal;
      if (evTargetsCustom) {
        evTargetsCustom.value = "";
        evTargetsCustom.style.display = "none";
      }
    } else {
      evTargets.value = "__custom__";
      if (evTargetsCustom) {
        evTargetsCustom.value = rawTargetsVal;
        evTargetsCustom.style.display = "block";
      }
    }
  }
  if (evContext) evContext.value = cfg.eval?.context || "rollout";

  // 3. 字段抽取
  const le = cfg.label_extract || {};
  const leMatchMode = document.getElementById("cfg-form-le-matchmode");
  const leValuePath = document.getElementById("cfg-form-le-valuepath");
  const leUrl = document.getElementById("cfg-form-le-url");
  const leModel = document.getElementById("cfg-form-le-model");

  if (leMatchMode) leMatchMode.value = le.match_mode || "exact";
  if (leValuePath) leValuePath.value = le.value_path || "";
  if (leUrl) leUrl.value = le.url || "";
  if (leModel) leModel.value = le.model || "";

  // 4. 数据与映射
  const dtType = document.getElementById("cfg-form-data-type");
  const dtMediaRoot = document.getElementById("cfg-form-data-mediaroot");
  const mpMessages = document.getElementById("cfg-form-map-messages");
  const mpImages = document.getElementById("cfg-form-map-images");
  const mpRole = document.getElementById("cfg-form-map-role");
  const mpContent = document.getElementById("cfg-form-map-content");

  if (dtType) dtType.value = cfg.data?.type || "llamafactory";
  if (dtMediaRoot) dtMediaRoot.value = cfg.data?.media_root || ".";
  if (mpMessages) mpMessages.value = cfg.data?.mapping?.messages || "messages";
  if (mpImages) mpImages.value = cfg.data?.mapping?.images || "images";
  if (mpRole) mpRole.value = cfg.data?.mapping?.role || "role";
  if (mpContent) mpContent.value = cfg.data?.mapping?.content || "content";

  // 5. 切分比例
  const sp = cfg.split || {};
  const spTest = document.getElementById("cfg-form-split-test");
  const spTestSlider = document.getElementById("cfg-form-split-test-slider");
  const spTrain = document.getElementById("cfg-form-split-train");
  const spTrainSlider = document.getElementById("cfg-form-split-train-slider");
  const spVal = document.getElementById("cfg-form-split-val");
  const spValSlider = document.getElementById("cfg-form-split-val-slider");
  const spSeed = document.getElementById("cfg-form-split-seed");

  if (spTest) spTest.value = sp.test ?? 0.05;
  if (spTestSlider) spTestSlider.value = sp.test ?? 0.05;
  if (spTrain) spTrain.value = sp.train ?? 0.95;
  if (spTrainSlider) spTrainSlider.value = sp.train ?? 0.95;
  if (spVal) spVal.value = sp.val ?? 0.0;
  if (spValSlider) spValSlider.value = sp.val ?? 0.0;
  if (spSeed) spSeed.value = sp.seed ?? 42;

  state.configDirty = false;
  const ind = document.getElementById("cfg-dirty-indicator");
  if (ind) ind.style.display = "none";
}

async function saveAllConfigChanges() {
  if (!state.currentDataset) return;

  // 确定激活的后端
  let activeBackend = "openai";
  if (document.getElementById("cfg-card-backend-mnn")?.classList.contains("active")) activeBackend = "mnn";
  if (document.getElementById("cfg-card-backend-vllm_offline")?.classList.contains("active")) activeBackend = "vllm_offline";
  if (document.getElementById("cfg-card-backend-llamacpp")?.classList.contains("active")) activeBackend = "llamacpp";
  if (document.getElementById("cfg-card-backend-hf")?.classList.contains("active")) activeBackend = "hf";

  let evalTargetsVal = document.getElementById("cfg-form-eval-targets")?.value;
  if (evalTargetsVal === "__custom__") {
    evalTargetsVal = document.getElementById("cfg-form-eval-targets-custom")?.value?.trim() || "first";
  }
  if (typeof evalTargetsVal === "string" && /^\d+$/.test(evalTargetsVal.trim())) {
    evalTargetsVal = parseInt(evalTargetsVal.trim(), 10);
  }

  const updates = [
    { key: "inference.backend", value: activeBackend },
    { key: "eval.method", value: document.getElementById("cfg-form-eval-method")?.value },
    { key: "scoring.scorer", value: document.getElementById("cfg-form-scoring-scorer")?.value },
    { key: "eval.targets", value: evalTargetsVal },
    { key: "eval.context", value: document.getElementById("cfg-form-eval-context")?.value },
  ];

  if (activeBackend === "mnn") {
    const cp = document.getElementById("cfg-form-mnn-configpath")?.value.trim();
    const mt = parseInt(document.getElementById("cfg-form-mnn-maxtokens")?.value || "1024", 10);
    const tpVal = document.getElementById("cfg-form-mnn-temp")?.value.trim();
    const tp = tpVal === "" ? null : parseFloat(tpVal);
    const topPVal = document.getElementById("cfg-form-mnn-topp")?.value.trim();
    const topP = topPVal === "" ? null : parseFloat(topPVal);
    const topKVal = document.getElementById("cfg-form-mnn-topk")?.value.trim();
    const topK = topKVal === "" ? null : parseInt(topKVal, 10);
    const rp = parseFloat(document.getElementById("cfg-form-mnn-reppenalty")?.value || "1.1");
    const fp = parseFloat(document.getElementById("cfg-form-mnn-freqpenalty")?.value || "0.0");
    const pp = parseFloat(document.getElementById("cfg-form-mnn-prespenalty")?.value || "0.0");
    const pw = parseInt(document.getElementById("cfg-form-mnn-penwindow")?.value || "0", 10);
    const ms = parseInt(document.getElementById("cfg-form-mnn-maxside")?.value || "2048", 10);
    const maxPx = parseInt(document.getElementById("cfg-form-mnn-imgmaxpixels")?.value || "589824", 10);
    const minPx = parseInt(document.getElementById("cfg-form-mnn-imgminpixels")?.value || "1024", 10);
    const quant = document.getElementById("cfg-form-mnn-quant")?.value.trim() || null;
    const sp = document.getElementById("cfg-form-mnn-sysprompt")?.value || "";

    if (cp) updates.push({ key: "inference.mnn.config_path", value: cp });
    updates.push({ key: "inference.mnn.max_tokens", value: mt });
    updates.push({ key: "inference.mnn.temperature", value: tp });
    updates.push({ key: "inference.mnn.top_p", value: topP });
    updates.push({ key: "inference.mnn.top_k", value: topK });
    updates.push({ key: "inference.mnn.repetition_penalty", value: rp });
    updates.push({ key: "inference.mnn.frequency_penalty", value: fp });
    updates.push({ key: "inference.mnn.presence_penalty", value: pp });
    updates.push({ key: "inference.mnn.penalty_window", value: pw });
    updates.push({ key: "inference.mnn.image_max_side", value: ms });
    updates.push({ key: "inference.mnn.image_max_pixels", value: maxPx });
    updates.push({ key: "inference.mnn.image_min_pixels", value: minPx });
    updates.push({ key: "inference.mnn.quant", value: quant });
    updates.push({ key: "inference.mnn.system_prompt", value: sp });
  } else if (activeBackend === "vllm_offline") {
    const mp = document.getElementById("cfg-form-vllm-model")?.value.trim();
    const gu = parseFloat(document.getElementById("cfg-form-vllm-gpuutil")?.value || "0.9");
    const ml = parseInt(document.getElementById("cfg-form-vllm-maxmodellen")?.value || "4096", 10);
    const ns = parseInt(document.getElementById("cfg-form-vllm-maxnumseqs")?.value || "128", 10);
    const bt = parseInt(document.getElementById("cfg-form-vllm-maxbatchedtokens")?.value || "20480", 10);
    const mt = parseInt(document.getElementById("cfg-form-vllm-maxtokens")?.value || "512", 10);
    const tp = parseFloat(document.getElementById("cfg-form-vllm-temp")?.value || "0");
    const topP = parseFloat(document.getElementById("cfg-form-vllm-topp")?.value || "1.0");
    const topK = parseInt(document.getElementById("cfg-form-vllm-topk")?.value || "-1", 10);
    const rp = parseFloat(document.getElementById("cfg-form-vllm-reppenalty")?.value || "1.0");
    const dt = document.getElementById("cfg-form-vllm-dtype")?.value || "auto";
    const mi = parseInt(document.getElementById("cfg-form-vllm-maximages")?.value || "4", 10);
    const minPx = parseInt(document.getElementById("cfg-form-vllm-imgminpixels")?.value || "784", 10);
    const maxPx = parseInt(document.getElementById("cfg-form-vllm-imgmaxpixels")?.value || "564480", 10);
    const rc = document.getElementById("cfg-form-vllm-remotecode")?.checked ?? true;
    const sp = document.getElementById("cfg-form-vllm-sysprompt")?.value || "";

    if (mp) updates.push({ key: "inference.vllm_offline.model_path", value: mp });
    updates.push({ key: "inference.vllm_offline.gpu_memory_utilization", value: gu });
    updates.push({ key: "inference.vllm_offline.max_model_len", value: ml });
    updates.push({ key: "inference.vllm_offline.max_num_seqs", value: ns });
    updates.push({ key: "inference.vllm_offline.max_num_batched_tokens", value: bt });
    updates.push({ key: "inference.vllm_offline.max_tokens", value: mt });
    updates.push({ key: "inference.vllm_offline.temperature", value: tp });
    updates.push({ key: "inference.vllm_offline.top_p", value: topP });
    updates.push({ key: "inference.vllm_offline.top_k", value: topK });
    updates.push({ key: "inference.vllm_offline.repetition_penalty", value: rp });
    updates.push({ key: "inference.vllm_offline.dtype", value: dt });
    updates.push({ key: "inference.vllm_offline.max_images_per_prompt", value: mi });
    updates.push({ key: "inference.vllm_offline.image_min_pixels", value: minPx });
    updates.push({ key: "inference.vllm_offline.image_max_pixels", value: maxPx });
    updates.push({ key: "inference.vllm_offline.trust_remote_code", value: rc });
    updates.push({ key: "inference.vllm_offline.system_prompt", value: sp });
  } else if (activeBackend === "hf") {
    const mp = document.getElementById("cfg-form-hf-model")?.value.trim();
    const dev = document.getElementById("cfg-form-hf-device")?.value.trim() || "auto";
    const dt = document.getElementById("cfg-form-hf-dtype")?.value || "auto";
    const mt = parseInt(document.getElementById("cfg-form-hf-maxtokens")?.value || "1024", 10);
    const attn = document.getElementById("cfg-form-hf-attn")?.value || null;
    const ms = parseInt(document.getElementById("cfg-form-hf-maxside")?.value || "0", 10);
    const minPx = parseInt(document.getElementById("cfg-form-hf-imgminpixels")?.value || "1024", 10);
    const maxPx = parseInt(document.getElementById("cfg-form-hf-imgmaxpixels")?.value || "1048576", 10);
    const gr = document.getElementById("cfg-form-hf-greedy")?.checked ?? true;
    const sp = document.getElementById("cfg-form-hf-sysprompt")?.value || "";

    if (mp) updates.push({ key: "inference.hf.model_path", value: mp });
    updates.push({ key: "inference.hf.device", value: dev });
    updates.push({ key: "inference.hf.dtype", value: dt });
    updates.push({ key: "inference.hf.max_tokens", value: mt });
    if (attn) updates.push({ key: "inference.hf.attn_implementation", value: attn });
    updates.push({ key: "inference.hf.image_max_side", value: ms });
    updates.push({ key: "inference.hf.image_min_pixels", value: minPx });
    updates.push({ key: "inference.hf.image_max_pixels", value: maxPx });
    updates.push({ key: "inference.hf.greedy", value: gr });
    updates.push({ key: "inference.hf.system_prompt", value: sp });
  } else if (activeBackend === "llamacpp") {
    const md = document.getElementById("cfg-form-llamacpp-mode")?.value || "server";
    const bu = document.getElementById("cfg-form-llamacpp-baseurl")?.value.trim() || "http://127.0.0.1:8080/v1";
    const mp = document.getElementById("cfg-form-llamacpp-modelpath")?.value.trim();
    const mm = document.getElementById("cfg-form-llamacpp-mmprojpath")?.value.trim();
    const cb = document.getElementById("cfg-form-llamacpp-clibinary")?.value.trim();
    const mt = parseInt(document.getElementById("cfg-form-llamacpp-maxtokens")?.value || "512", 10);
    const tpVal = document.getElementById("cfg-form-llamacpp-temp")?.value.trim();
    const tp = tpVal === "" ? 0.0 : parseFloat(tpVal);
    const topPVal = document.getElementById("cfg-form-llamacpp-topp")?.value.trim();
    const topP = topPVal === "" ? 1.0 : parseFloat(topPVal);
    const rp = parseFloat(document.getElementById("cfg-form-llamacpp-reppenalty")?.value || "1.0");
    const cc = parseInt(document.getElementById("cfg-form-llamacpp-concurrency")?.value || "4", 10);
    const ms = parseInt(document.getElementById("cfg-form-llamacpp-maxside")?.value || "2048", 10);
    const sp = document.getElementById("cfg-form-llamacpp-sysprompt")?.value || "";

    updates.push({ key: "inference.llamacpp.mode", value: md });
    updates.push({ key: "inference.llamacpp.base_url", value: bu });
    if (mp) updates.push({ key: "inference.llamacpp.model_path", value: mp });
    if (mm) updates.push({ key: "inference.llamacpp.mmproj_path", value: mm });
    if (cb) updates.push({ key: "inference.llamacpp.cli_binary", value: cb });
    updates.push({ key: "inference.llamacpp.max_tokens", value: mt });
    updates.push({ key: "inference.llamacpp.temperature", value: tp });
    updates.push({ key: "inference.llamacpp.top_p", value: topP });
    updates.push({ key: "inference.llamacpp.repetition_penalty", value: rp });
    updates.push({ key: "inference.llamacpp.max_concurrency", value: cc });
    updates.push({ key: "inference.llamacpp.image_max_side", value: ms });
    updates.push({ key: "inference.llamacpp.system_prompt", value: sp });
  } else {
    const mdl = document.getElementById("cfg-form-openai-model")?.value.trim();
    const bu = document.getElementById("cfg-form-openai-baseurl")?.value.trim();
    const ake = document.getElementById("cfg-form-openai-apikeyenv")?.value.trim() || "OPENAI_API_KEY";
    const cc = parseInt(document.getElementById("cfg-form-openai-concurrency")?.value || "8", 10);
    const mt = parseInt(document.getElementById("cfg-form-openai-maxtokens")?.value || "512", 10);
    const tp = parseFloat(document.getElementById("cfg-form-openai-temp")?.value || "0");
    const topP = parseFloat(document.getElementById("cfg-form-openai-topp")?.value || "1.0");
    const to = parseFloat(document.getElementById("cfg-form-openai-timeout")?.value || "120.0");
    const mr = parseInt(document.getElementById("cfg-form-openai-maxretries")?.value || "3", 10);
    const id = document.getElementById("cfg-form-openai-imagedetail")?.value || "high";
    const sp = document.getElementById("cfg-form-openai-sysprompt")?.value || "";

    if (mdl) updates.push({ key: "inference.openai.model", value: mdl });
    if (bu) updates.push({ key: "inference.openai.base_url", value: bu });
    updates.push({ key: "inference.openai.api_key_env", value: ake });
    updates.push({ key: "inference.openai.max_concurrency", value: cc });
    updates.push({ key: "inference.openai.max_tokens", value: mt });
    updates.push({ key: "inference.openai.temperature", value: tp });
    updates.push({ key: "inference.openai.top_p", value: topP });
    updates.push({ key: "inference.openai.request_timeout", value: to });
    updates.push({ key: "inference.openai.max_retries", value: mr });
    updates.push({ key: "inference.openai.image_detail", value: id });
    updates.push({ key: "inference.openai.system_prompt", value: sp });
  }


  // 字段抽取
  const leMm = document.getElementById("cfg-form-le-matchmode")?.value;
  const leVp = document.getElementById("cfg-form-le-valuepath")?.value.trim();
  const leUrl = document.getElementById("cfg-form-le-url")?.value.trim();
  const leMdl = document.getElementById("cfg-form-le-model")?.value.trim();
  if (leMm) updates.push({ key: "label_extract.match_mode", value: leMm });
  if (leVp) updates.push({ key: "label_extract.value_path", value: leVp });
  if (leUrl) updates.push({ key: "label_extract.url", value: leUrl });
  if (leMdl) updates.push({ key: "label_extract.model", value: leMdl });

  // 基础数据
  const dtType = document.getElementById("cfg-form-data-type")?.value;
  const dtMr = document.getElementById("cfg-form-data-mediaroot")?.value.trim();
  if (dtType) updates.push({ key: "data.type", value: dtType });
  if (dtMr) updates.push({ key: "data.media_root", value: dtMr });

  // 字段映射
  const mpMsg = document.getElementById("cfg-form-map-messages")?.value.trim();
  const mpImg = document.getElementById("cfg-form-map-images")?.value.trim();
  const mpRole = document.getElementById("cfg-form-map-role")?.value.trim();
  const mpCnt = document.getElementById("cfg-form-map-content")?.value.trim();
  if (mpMsg) updates.push({ key: "data.mapping.messages", value: mpMsg });
  if (mpImg) updates.push({ key: "data.mapping.images", value: mpImg });
  if (mpRole) updates.push({ key: "data.mapping.role", value: mpRole });
  if (mpCnt) updates.push({ key: "data.mapping.content", value: mpCnt });

  // 切分
  const spTst = parseFloat(document.getElementById("cfg-form-split-test")?.value || "0.05");
  const spTrn = parseFloat(document.getElementById("cfg-form-split-train")?.value || "0.95");
  const spVal = parseFloat(document.getElementById("cfg-form-split-val")?.value || "0.0");
  const spSd = parseInt(document.getElementById("cfg-form-split-seed")?.value || "42", 10);
  updates.push({ key: "split.test", value: spTst });
  updates.push({ key: "split.train", value: spTrn });
  updates.push({ key: "split.val", value: spVal });
  updates.push({ key: "split.seed", value: spSd });

  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updates }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast("全部配置项已成功持久化保存并保留原注释！", "success");
    await loadConfig();
  } catch (err) {
    showToast(`保存配置失败: ${err.message}`, "error");
  }
}

async function saveSingleConfigKey(key, value) {
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updates: [{ key, value }] }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast(`配置项 ${key} 已成功持久化保存`, "success");
    await loadConfig();
  } catch (err) {
    showToast(`更新配置失败: ${err.message}`, "error");
  }
}

function onLlamaCppModeChange() {
  const mode = document.getElementById("cfg-form-llamacpp-mode")?.value || "server";
  const urlField = document.getElementById("cfg-field-llamacpp-baseurl");
  const cliField = document.getElementById("cfg-field-llamacpp-clibinary");
  if (urlField) urlField.style.display = (mode === "server") ? "block" : "none";
  if (cliField) cliField.style.display = (mode === "cli") ? "block" : "none";
}

function onLlamaCppModelSelectChange(val) {
  if (!val) return;
  const sel = document.getElementById("cfg-form-llamacpp-model-select");
  const opt = sel?.options[sel.selectedIndex];
  const mmproj = opt?.getAttribute("data-mmproj") || "";
  const modelPathInput = document.getElementById("cfg-form-llamacpp-modelpath");
  const mmprojPathInput = document.getElementById("cfg-form-llamacpp-mmprojpath");
  if (modelPathInput) modelPathInput.value = val;
  if (mmprojPathInput && mmproj) mmprojPathInput.value = mmproj;
  markConfigDirty();
}

// --------------------------------------------------------------------------
// 任务启动模态框配置 (消除黑盒)
// --------------------------------------------------------------------------
function openJobModal(type, prefill = null) {
  state.jobModal.type = type;
  const modal = document.getElementById("job-launch-modal");
  const titleEl = document.getElementById("job-modal-title");
  const iconEl = document.getElementById("job-modal-type-icon");
  const dsSelect = document.getElementById("job-modal-dataset");
  const feParams = document.getElementById("job-modal-field-eval-params");
  const evalParams = document.getElementById("job-modal-eval-params");

  // 标题与图标定制
  const typeMap = {
    "eval": { title: "配置并启动 对话评测任务 (eval)", icon: "🚀" },
    "field-eval": { title: "配置并启动 逐字段抽取评测任务 (field-eval)", icon: "🏷️" },
    "pred": { title: "配置并启动 模型批量推理 (pred)", icon: "⚡" },
    "score": { title: "配置并启动 离线指标打分 (score)", icon: "🎯" },
    "sweep": { title: "配置并启动 跨数据集扫描 (sweep)", icon: "🌐" },
  };
  const info = typeMap[type] || { title: `配置并启动 ${type} 任务`, icon: "⚙️" };
  if (titleEl) titleEl.textContent = info.title;
  if (iconEl) iconEl.textContent = info.icon;

  // 填充数据集选择下拉框
  if (dsSelect) {
    if (type === "sweep") {
      dsSelect.innerHTML = `<option value="">(扫描当前工作区全部数据集)</option>`;
    } else {
      dsSelect.innerHTML = state.datasets
        .map((d) => `<option value="${escapeHtml(d.name)}" ${d.name === (prefill?.dataset || state.currentDataset) ? "selected" : ""}>${escapeHtml(d.name)}</option>`)
        .join("");
    }
  }

  // 专用参数面板显示隐藏
  if (feParams) feParams.style.display = type === "field-eval" ? "block" : "none";
  if (evalParams) evalParams.style.display = (type === "eval" || type === "score") ? "block" : "none";

  // 重置通用输入项默认值
  const backendSelect = document.getElementById("job-modal-backend");
  const modelInput = document.getElementById("job-modal-model");
  const modelSelect = document.getElementById("job-modal-model-select");
  const limitInput = document.getElementById("job-modal-limit");
  const failfastCb = document.getElementById("job-modal-failfast");
  const matchModeSelect = document.getElementById("job-modal-matchmode");
  const targetsSelect = document.getElementById("job-modal-targets");
  const targetsCustom = document.getElementById("job-modal-targets-custom");
  const evalTargetsSelect = document.getElementById("job-modal-eval-targets");
  const evalTargetsCustom = document.getElementById("job-modal-eval-targets-custom");
  const overwriteCb = document.getElementById("job-modal-overwrite");
  const scorerSelect = document.getElementById("job-modal-scorer");

  if (backendSelect) backendSelect.value = "";
  if (modelInput) modelInput.value = "";
  if (modelSelect) modelSelect.value = "";
  if (limitInput) limitInput.value = "";
  if (failfastCb) failfastCb.checked = false;
  if (matchModeSelect) matchModeSelect.value = "";
  if (targetsSelect) targetsSelect.value = "";
  if (targetsCustom) { targetsCustom.value = ""; targetsCustom.style.display = "none"; }
  if (evalTargetsSelect) evalTargetsSelect.value = "";
  if (evalTargetsCustom) { evalTargetsCustom.value = ""; evalTargetsCustom.style.display = "none"; }
  if (overwriteCb) overwriteCb.checked = false;
  if (scorerSelect) scorerSelect.value = "";

  // 确保弹窗内模型列表已渲染最新数据
  if ((!state.models.hf_models || state.models.hf_models.length === 0) &&
      (!state.models.mnn_models || state.models.mnn_models.length === 0)) {
    loadDetectedModels();
  } else {
    renderModelSelects();
  }

  // 应用预填充参数（针对一键重跑）
  if (prefill) {
    if (prefill.dataset && dsSelect) dsSelect.value = prefill.dataset;
    if (prefill.backend && backendSelect) backendSelect.value = prefill.backend;
    if (prefill.model) {
      if (modelInput) modelInput.value = prefill.model;
      if (modelSelect) {
        modelSelect.value = prefill.model;
        if (modelSelect.value !== prefill.model) {
          modelSelect.value = "__custom__";
        }
      }
    }
  }

  onJobModalBackendChange();
  updateJobCommandPreview();
  if (modal) modal.showModal();
}

function closeJobModal() {
  const modal = document.getElementById("job-launch-modal");
  if (modal) modal.close();
}

function onJobModalTargetsChange() {
  const sel = document.getElementById("job-modal-targets");
  const customInput = document.getElementById("job-modal-targets-custom");
  if (!sel || !customInput) return;
  if (sel.value === "__custom__") {
    customInput.style.display = "block";
    customInput.focus();
  } else {
    customInput.style.display = "none";
  }
  updateJobCommandPreview();
}

function onJobModalEvalTargetsChange() {
  const sel = document.getElementById("job-modal-eval-targets");
  const customInput = document.getElementById("job-modal-eval-targets-custom");
  if (!sel || !customInput) return;
  if (sel.value === "__custom__") {
    customInput.style.display = "block";
    customInput.focus();
  } else {
    customInput.style.display = "none";
  }
  updateJobCommandPreview();
}

function getJobModalTargets(type) {
  if (type === "field-eval") {
    const sel = document.getElementById("job-modal-targets");
    if (!sel || !sel.value) return "";
    if (sel.value === "__custom__") {
      return document.getElementById("job-modal-targets-custom")?.value?.trim() || "";
    }
    return sel.value;
  }
  const sel = document.getElementById("job-modal-eval-targets");
  if (!sel || !sel.value) return "";
  if (sel.value === "__custom__") {
    return document.getElementById("job-modal-eval-targets-custom")?.value?.trim() || "";
  }
  return sel.value;
}

function buildJobModalArgs() {
  const type = state.jobModal.type;
  const dsSelect = document.getElementById("job-modal-dataset");
  const dataset = dsSelect ? dsSelect.value : state.currentDataset;
  const pathSpan = document.getElementById("job-modal-dataset-path");

  // 更新数据集物理路径提示
  if (pathSpan) {
    const dsObj = state.datasets.find((d) => d.name === dataset);
    pathSpan.textContent = dsObj ? `路径: ${dsObj.path}` : "";
  }

  const backend = document.getElementById("job-modal-backend")?.value;
  const modelInput = document.getElementById("job-modal-model");
  const model = modelInput?.value?.trim();
  const limit = document.getElementById("job-modal-limit")?.value?.trim();
  const failfast = document.getElementById("job-modal-failfast")?.checked;

  const params = {};
  const parts = ["python", "-m", "eval_vlm", type];
  if (dataset) parts.push("-d", dataset);

  if (backend) {
    params.backend = backend;
    parts.push("--backend", backend);
  }
  if (model) {
    params.model = model;
    parts.push("--model", model.includes(" ") ? `"${model}"` : model);
  }
  if (limit) {
    const limInt = parseInt(limit, 10);
    if (!isNaN(limInt) && limInt > 0) {
      params.limit = limInt;
      parts.push("--limit", String(limInt));
    }
  }
  if (failfast) {
    params.fail_fast = true;
    parts.push("--fail-fast");
  }

  if (type === "field-eval") {
    const matchMode = document.getElementById("job-modal-matchmode")?.value;
    const targets = getJobModalTargets("field-eval");
    const overwrite = document.getElementById("job-modal-overwrite")?.checked;
    if (matchMode) {
      params.match_mode = matchMode;
      parts.push("--match-mode", matchMode);
    }
    if (targets) {
      params.targets = targets;
      parts.push("--targets", targets);
    }
    if (overwrite) {
      params.overwrite = true;
      parts.push("--overwrite");
    }
  } else if (type === "eval" || type === "score") {
    const scorer = document.getElementById("job-modal-scorer")?.value;
    const targets = getJobModalTargets("eval");
    if (scorer) {
      params.scorer = scorer;
      parts.push("--scorer", scorer);
    }
    if (targets) {
      params.targets = targets;
      parts.push("--targets", targets);
    }
  }

  return { type, dataset, params, parts, cmdStr: parts.join(" ") };
}

function updateJobCommandPreview() {
  const { cmdStr } = buildJobModalArgs();
  const previewEl = document.getElementById("job-modal-cmd-preview");
  if (previewEl) previewEl.textContent = cmdStr;

  const logDest = document.getElementById("job-modal-log-destination");
  if (logDest) {
    logDest.textContent = `.../_webui/jobs/<job_id>/log.txt (工作区状态目录)`;
  }
}

async function confirmLaunchJob() {
  const { type, dataset, params } = buildJobModalArgs();
  try {
    const url = dataset ? `/api/datasets/${encodeURIComponent(dataset)}/jobs` : "/api/jobs";
    const res = await apiFetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, dataset, params }),
    });
    const job = await res.json();
    closeJobModal();
    showToast(`任务已提交入队: ${job.id}`, "success");
    await loadJobs();
    openTerminal(job.id);
  } catch (err) {
    showToast(`启动任务失败: ${err.message}`, "error");
  }
}

function onJobModalModelSelectChange() {
  const sel = document.getElementById("job-modal-model-select");
  const input = document.getElementById("job-modal-model");
  const backendSel = document.getElementById("job-modal-backend");
  if (!sel || !input) return;
  const val = sel.value;
  if (!val) {
    updateJobCommandPreview();
    return;
  }
  if (val === "__custom__") {
    input.value = "";
    input.focus();
    updateJobCommandPreview();
    return;
  }
  const opt = sel.options[sel.selectedIndex];
  const modelType = opt?.getAttribute("data-type");
  const modelName = opt?.getAttribute("data-modelname");
  const mmprojPath = opt?.getAttribute("data-mmproj");

  if (modelType === "llamacpp") {
    input.value = val;
    input.setAttribute("data-modelpath", val);
    if (mmprojPath) input.setAttribute("data-mmproj", mmprojPath);
    else input.removeAttribute("data-mmproj");
    if (backendSel) backendSel.value = "llamacpp";
  } else {
    input.value = val;
    input.removeAttribute("data-modelpath");
    input.removeAttribute("data-mmproj");
    if (modelType === "mnn") {
      if (backendSel) backendSel.value = "mnn";
    } else if (modelType === "hf") {
      if (backendSel && (!backendSel.value || backendSel.value === "mnn")) {
        backendSel.value = "vllm_offline";
      }
    }
  }
  onJobModalBackendChange();
  updateJobCommandPreview();
}

function onJobModalBackendChange() {
  const backend = document.getElementById("job-modal-backend")?.value || "";
  const labelEl = document.getElementById("job-modal-model-label");
  const badgeEl = document.getElementById("job-modal-model-badge");
  let flagText = "--model";
  let descText = "目标模型 / 权重目录 (--model)";
  if (backend === "hf") {
    descText = "HF 权重目录 / 模型名 (--model)";
  } else if (backend === "vllm_offline") {
    descText = "vLLM 离线权重目录 (--model)";
  } else if (backend === "mnn") {
    descText = "MNN 权重目录 / 配置 (--model)";
  } else if (backend === "llamacpp") {
    descText = "GGUF 模型文件 / 权重目录 (--model)";
  }
  if (labelEl) labelEl.textContent = descText;
  if (badgeEl) badgeEl.textContent = flagText;
}

// --------------------------------------------------------------------------
// 任务面板内联启动控制台 (Inline Launch Console)
// --------------------------------------------------------------------------
let inlineJobType = "field-eval";

function initInlineJobConsole() {
  const dsSelect = document.getElementById("job-inline-dataset");
  if (dsSelect) {
    dsSelect.innerHTML = (state.datasets || [])
      .map((d) => `<option value="${escapeHtml(d.name)}" ${d.name === state.currentDataset ? "selected" : ""}>${escapeHtml(d.name)}</option>`)
      .join("");
  }
  updateInlineJobPreview();
}

function switchInlineJobType(type) {
  inlineJobType = type;
  const types = ["field-eval", "eval", "pred", "score"];
  types.forEach((t) => {
    const btn = document.getElementById(`btn-type-${t}`);
    if (btn) btn.classList.toggle("active", t === type);
  });

  const feParams = document.getElementById("job-inline-field-eval-params");
  const evalParams = document.getElementById("job-inline-eval-params");
  if (feParams) feParams.style.display = type === "field-eval" ? "block" : "none";
  if (evalParams) evalParams.style.display = (type === "eval" || type === "score") ? "block" : "none";

  const dsSelect = document.getElementById("job-inline-dataset");
  if (dsSelect) {
    dsSelect.innerHTML = (state.datasets || [])
      .map((d) => `<option value="${escapeHtml(d.name)}" ${d.name === state.currentDataset ? "selected" : ""}>${escapeHtml(d.name)}</option>`)
      .join("");
  }

  updateInlineJobPreview();
}

function onJobInlineModelSelectChange() {
  const sel = document.getElementById("job-inline-model-select");
  const input = document.getElementById("job-inline-model");
  const backendSel = document.getElementById("job-inline-backend");
  if (!sel || !input) return;
  const val = sel.value;
  if (!val) {
    updateInlineJobPreview();
    return;
  }
  if (val === "__custom__") {
    input.value = "";
    input.focus();
    updateInlineJobPreview();
    return;
  }
  const opt = sel.options[sel.selectedIndex];
  const modelType = opt?.getAttribute("data-type");
  const modelName = opt?.getAttribute("data-modelname");
  const mmprojPath = opt?.getAttribute("data-mmproj");

  if (modelType === "llamacpp") {
    input.value = val;
    input.setAttribute("data-modelpath", val);
    if (mmprojPath) input.setAttribute("data-mmproj", mmprojPath);
    else input.removeAttribute("data-mmproj");
    if (backendSel) backendSel.value = "llamacpp";
  } else {
    input.value = val;
    input.removeAttribute("data-modelpath");
    input.removeAttribute("data-mmproj");
    if (modelType === "mnn") {
      if (backendSel) backendSel.value = "mnn";
    } else if (modelType === "hf") {
      if (backendSel && (!backendSel.value || backendSel.value === "mnn")) {
        backendSel.value = "vllm_offline";
      }
    }
  }
  onJobInlineBackendChange();
  updateInlineJobPreview();
}

function onJobInlineBackendChange() {
  const backend = document.getElementById("job-inline-backend")?.value || "";
  const labelEl = document.getElementById("job-inline-model-label");
  const badgeEl = document.getElementById("job-inline-model-badge");
  let flagText = "--model";
  let descText = "目标模型 / 权重目录 (--model)";
  if (backend === "hf") {
    descText = "HF 权重目录 / 模型名 (--model)";
  } else if (backend === "vllm_offline") {
    descText = "vLLM 离线权重目录 (--model)";
  } else if (backend === "mnn") {
    descText = "MNN 权重目录 / 配置 (--model)";
  } else if (backend === "llamacpp") {
    descText = "GGUF 模型文件 / 权重目录 (--model)";
  }
  if (labelEl) labelEl.textContent = descText;
  if (badgeEl) badgeEl.textContent = flagText;
  updateInlineJobPreview();
}

function onJobInlineTargetsChange() {
  const sel = document.getElementById("job-inline-targets");
  const customInput = document.getElementById("job-inline-targets-custom");
  if (!sel || !customInput) return;
  if (sel.value === "__custom__") {
    customInput.style.display = "block";
    customInput.focus();
  } else {
    customInput.style.display = "none";
  }
  updateInlineJobPreview();
}

function onJobInlineEvalTargetsChange() {
  const sel = document.getElementById("job-inline-eval-targets");
  const customInput = document.getElementById("job-inline-eval-targets-custom");
  if (!sel || !customInput) return;
  if (sel.value === "__custom__") {
    customInput.style.display = "block";
    customInput.focus();
  } else {
    customInput.style.display = "none";
  }
  updateInlineJobPreview();
}

function getInlineTargets(type) {
  if (type === "field-eval") {
    const sel = document.getElementById("job-inline-targets");
    if (!sel || !sel.value) return "";
    if (sel.value === "__custom__") {
      return document.getElementById("job-inline-targets-custom")?.value?.trim() || "";
    }
    return sel.value;
  }
  const sel = document.getElementById("job-inline-eval-targets");
  if (!sel || !sel.value) return "";
  if (sel.value === "__custom__") {
    return document.getElementById("job-inline-eval-targets-custom")?.value?.trim() || "";
  }
  return sel.value;
}

function buildInlineJobArgs() {
  const type = inlineJobType;
  const dsSelect = document.getElementById("job-inline-dataset");
  const dataset = dsSelect ? dsSelect.value : state.currentDataset;

  const backend = document.getElementById("job-inline-backend")?.value;
  const modelInput = document.getElementById("job-inline-model");
  const model = modelInput?.value?.trim();
  const limit = document.getElementById("job-inline-limit")?.value?.trim();
  const failfast = document.getElementById("job-inline-failfast")?.checked;

  const params = {};
  const parts = ["python", "-m", "eval_vlm", type];
  if (dataset) parts.push("-d", dataset);

  if (backend) {
    params.backend = backend;
    parts.push("--backend", backend);
  }
  if (model) {
    params.model = model;
    parts.push("--model", model.includes(" ") ? `"${model}"` : model);
  }
  if (limit) {
    const limInt = parseInt(limit, 10);
    if (!isNaN(limInt) && limInt > 0) {
      params.limit = limInt;
      parts.push("--limit", String(limInt));
    }
  }
  if (failfast) {
    params.fail_fast = true;
    parts.push("--fail-fast");
  }

  if (type === "field-eval") {
    const matchMode = document.getElementById("job-inline-matchmode")?.value;
    const targets = getInlineTargets("field-eval");
    const overwrite = document.getElementById("job-inline-overwrite")?.checked;
    if (matchMode) {
      params.match_mode = matchMode;
      parts.push("--match-mode", matchMode);
    }
    if (targets) {
      params.targets = targets;
      parts.push("--targets", targets);
    }
    if (overwrite) {
      params.overwrite = true;
      parts.push("--overwrite");
    }
  } else if (type === "eval" || type === "score") {
    const scorer = document.getElementById("job-inline-scorer")?.value;
    const targets = getInlineTargets("eval");
    if (scorer) {
      params.scorer = scorer;
      parts.push("--scorer", scorer);
    }
    if (targets) {
      params.targets = targets;
      parts.push("--targets", targets);
    }
  }

  return { type, dataset, params, parts, cmdStr: parts.join(" ") };
}

function updateInlineJobPreview() {
  const { cmdStr } = buildInlineJobArgs();
  const previewEl = document.getElementById("job-inline-cmd-preview");
  if (previewEl) previewEl.textContent = cmdStr;

  const logDest = document.getElementById("job-inline-log-destination");
  if (logDest) {
    logDest.textContent = `.../_webui/jobs/<job_id>/log.txt (工作区状态目录)`;
  }
}

async function submitInlineJob() {
  const { type, dataset, params } = buildInlineJobArgs();
  try {
    const url = dataset ? `/api/datasets/${encodeURIComponent(dataset)}/jobs` : "/api/jobs";
    const res = await apiFetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, dataset, params }),
    });
    const job = await res.json();
    showToast(`任务已成功提交入队: ${job.id}`, "success");
    await loadJobs();
    openTerminal(job.id);
  } catch (err) {
    showToast(`提交任务失败: ${err.message}`, "error");
  }
}

// --------------------------------------------------------------------------
// 任务管理与 SSE (Jobs)
// --------------------------------------------------------------------------
async function loadJobs() {
  if (state.jobLoadPromise) return state.jobLoadPromise;
  state.jobLoadPromise = (async () => {
    try {
      const res = await apiFetch("/api/jobs");
      state.jobs = await res.json();
      renderJobs();
      updateJobPolling();
      return state.jobs;
    } catch (err) {
      console.error(err);
      showToast(`获取任务队列失败: ${err.message}`, "error");
      throw err;
    } finally {
      state.jobLoadPromise = null;
    }
  })();
  return state.jobLoadPromise;
}

function refreshJobs() {
  return loadJobs().catch(() => state.jobs);
}

function hasActiveJobs() {
  return state.jobs.some((job) => job.status === "queued" || job.status === "running");
}

function updateJobPolling() {
  const needed = state.activeTab === "jobs" || state.logDrawerOpen || hasActiveJobs();
  if (needed && !state.jobPollTimer) {
    state.jobPollTimer = window.setInterval(() => { refreshJobs(); }, 2000);
  } else if (!needed && state.jobPollTimer) {
    window.clearInterval(state.jobPollTimer);
    state.jobPollTimer = null;
  }
}

function jobDatasetLabel(dataset) {
  if (!dataset) return "全量";
  const items = String(dataset).split(",").map((item) => item.trim()).filter(Boolean);
  return items.length > 1 ? `共 ${items.length} 个数据集` : items[0] || "全量";
}

function renderJobs() {
  const tbody = document.getElementById("jobs-tbody");
  if (!tbody) return;

  if (!state.jobs.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:3rem; color:var(--text-dim);">暂无运行任务</td></tr>`;
    return;
  }

  tbody.innerHTML = state.jobs
    .map((j) => {
      let statusBadge = `<span class="role-badge">${escapeHtml(j.status)}</span>`;
      if (j.status === "succeeded") {
        statusBadge = `<span class="role-badge" style="background:var(--emerald-bg); border-color:var(--emerald-border); color:var(--emerald-500);">成功</span>`;
      } else if (j.status === "failed") {
        statusBadge = `<span class="role-badge" style="background:var(--rose-bg); border-color:var(--rose-border); color:var(--rose-500);">失败</span>`;
      } else if (j.status === "running") {
        statusBadge = `<span class="role-badge" style="background:rgba(99,102,241,0.2); border-color:rgba(99,102,241,0.4); color:#a5b4fc;">运行中</span>`;
      }

      const cmdText = (j.command && j.command.length) ? j.command.join(" ") : (j.params ? JSON.stringify(j.params) : "—");

      return `
      <tr>
        <td class="job-id-cell" style="font-family: var(--font-mono); font-weight: 600; font-size: 0.8rem;" title="${escapeHtml(j.id)}"><div class="job-id-text">${escapeHtml(j.id)}</div></td>
        <td><span class="role-badge" style="background:rgba(6,182,212,0.15); color:var(--cyan-500);">${escapeHtml(j.type)}</span></td>
        <td title="${escapeHtml(j.dataset || "全量")}"><strong style="color: var(--text-main);">${escapeHtml(jobDatasetLabel(j.dataset))}</strong></td>
        <td class="job-command-cell" style="max-width: 320px; font-size: 0.76rem; font-family: var(--font-mono); color: var(--text-dim); word-break: break-all;" title="${escapeHtml(cmdText)}">
          <div class="job-command-text">${escapeHtml(cmdText)}</div>
        </td>
        <td>${escapeHtml(j.user)}</td>
        <td>${statusBadge}</td>
        <td style="font-size: 0.78rem; color: var(--text-dim);">${new Date(j.created_at).toLocaleString()}</td>
        <td>
          <div class="job-actions" style="display: flex; gap: 0.35rem;">
            <button class="btn btn-sm btn-primary" onclick="openTerminal('${escapeHtml(j.id)}')">🖥️ 实时日志</button>
            ${
              j.status === "running" || j.status === "queued"
                ? `<button class="btn btn-sm btn-outline-danger" onclick="cancelJob('${escapeHtml(j.id)}')">停止</button>`
                : ""
            }
            ${
              j.status === "failed" || j.status === "interrupted" || j.status === "canceled"
                ? `<button class="btn btn-sm" onclick="resumeJob('${escapeHtml(j.id)}')">续跑</button>`
                : ""
            }
            ${
              j.status !== "running" && j.status !== "queued"
                ? `<button class="btn btn-sm btn-outline-danger" onclick="deleteJob('${escapeHtml(j.id)}')">删除</button>`
                : ""
            }
          </div>
        </td>
      </tr>
    `;
    })
    .join("");
}

async function openTerminal(jobId) {
  const session = ++state.terminalSession;
  stopTerminalConnection();
  state.currentJobId = jobId;
  state.terminalLogs = "";
  state.logDrawerOpen = true;

  const drawer = document.getElementById("terminal-drawer");
  const jobLabel = document.getElementById("terminal-job-label");
  const dsLabel = document.getElementById("terminal-dataset-label");
  const cmdDisplay = document.getElementById("terminal-cmd-display");
  const logDisplay = document.getElementById("terminal-log-display");
  const pre = document.getElementById("terminal-pre");

  if (drawer) {
    drawer.classList.remove("hidden");
    drawer.setAttribute("aria-expanded", "true");
  }
  if (jobLabel) jobLabel.textContent = `任务: ${jobId}`;
  if (pre) pre.textContent = "正在连接进程日志输出流...\n";

  // The stream replays the persisted log file, including for completed jobs.
  // Do not make that replay depend on the separate status endpoint: a slow
  // status query previously left this drawer permanently at the placeholder.
  connectTerminalStream(jobId, session, 0);
  void refreshTerminalSnapshot(jobId, session);
  updateJobPolling();
}

async function getJobSnapshot(jobId) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 3000);
  try {
    const response = await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, { signal: controller.signal });
    return await response.json();
  } finally {
    window.clearTimeout(timeout);
  }
}

function renderTerminalSnapshot(job, { showQueueMessage = false } = {}) {
  const dsLabel = document.getElementById("terminal-dataset-label");
  const cmdDisplay = document.getElementById("terminal-cmd-display");
  const logDisplay = document.getElementById("terminal-log-display");
  const pre = document.getElementById("terminal-pre");
  if (dsLabel) {
    dsLabel.textContent = `数据集: ${jobDatasetLabel(job.dataset)}`;
    dsLabel.title = job.dataset || "全量";
  }
  if (cmdDisplay) cmdDisplay.textContent = job.command?.length ? job.command.join(" ") : "—";
  if (logDisplay) logDisplay.textContent = job.log_file || "—";
  if (pre && showQueueMessage && job.status === "queued" && !state.terminalLogs) {
    pre.textContent = `[调度队列] 任务已进入等待执行队列 (ID: ${job.id})\n即将执行: ${job.command?.length ? job.command.join(" ") : "—"}\n正在连接调度器并等待拉起进程...\n\n`;
  }
}

async function refreshTerminalSnapshot(jobId, session) {
  try {
    const job = await getJobSnapshot(jobId);
    if (session !== state.terminalSession || !state.logDrawerOpen || state.currentJobId !== jobId) return;
    const snapshotIndex = state.jobs.findIndex((item) => item.id === jobId);
    if (snapshotIndex >= 0) state.jobs[snapshotIndex] = job;
    else state.jobs.unshift(job);
    renderJobs();
    // Only a still-empty terminal may show the queue hint. Never overwrite
    // log text that has already been replayed from the stream.
    renderTerminalSnapshot(job, { showQueueMessage: true });
  } catch (err) {
    if (session !== state.terminalSession || !state.logDrawerOpen || state.currentJobId !== jobId) return;
    // The stream remains the source of truth for log replay. A status failure
    // is intentionally non-fatal and must not replace terminal output.
    const detail = err?.name === "AbortError" ? "状态同步超时" : "状态同步失败";
    if (state.eventSource) setTerminalConnection(`${detail}，日志流仍在显示`);
  }
}

function isTerminalJob(status) {
  return ["succeeded", "failed", "canceled", "interrupted", "deleted"].includes(status);
}

function setTerminalConnection(message) {
  const el = document.getElementById("terminal-connection-status");
  if (el) el.textContent = message;
}

function stopTerminalConnection() {
  if (state.terminalRetryTimer) {
    window.clearTimeout(state.terminalRetryTimer);
    state.terminalRetryTimer = null;
  }
  if (state.eventSource) state.eventSource.close();
  state.eventSource = null;
}

function connectTerminalStream(jobId, session, attempt) {
  if (session !== state.terminalSession || !state.logDrawerOpen || state.currentJobId !== jobId) return;
  setTerminalConnection(attempt ? `正在重连日志流（第 ${attempt} 次）...` : "正在连接日志流...");
  const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream`);
  const cmdDisplay = document.getElementById("terminal-cmd-display");
  const logDisplay = document.getElementById("terminal-log-display");
  const pre = document.getElementById("terminal-pre");
  let terminalReceived = false;
  state.eventSource = source;
  const isCurrent = () => session === state.terminalSession && state.eventSource === source && state.currentJobId === jobId && state.logDrawerOpen;
  const sessionIsCurrent = () => session === state.terminalSession && state.currentJobId === jobId && state.logDrawerOpen;

  source.onopen = () => { if (isCurrent()) setTerminalConnection("日志流已连接"); };

  source.addEventListener("started", (e) => {
    if (!isCurrent()) return;
    try {
      const data = JSON.parse(e.data);
      if (cmdDisplay && data.command) cmdDisplay.textContent = data.command.join(" ");
      if (logDisplay && data.log_file) logDisplay.textContent = data.log_file;
      refreshJobs();
    } catch (_) {}
  });

  source.addEventListener("log", (e) => {
    if (!isCurrent()) return;
    try {
      const line = JSON.parse(e.data);
      state.terminalLogs += line;
      if (pre) pre.textContent = state.terminalLogs;
      if (state.autoScrollLogs && pre) {
        pre.scrollTop = pre.scrollHeight;
      }
    } catch (_) {}
  });

  source.addEventListener("status", (e) => {
    if (!isCurrent()) return;
    try {
      const statusData = JSON.parse(e.data);
      const exitInfo = statusData.exit_code !== undefined && statusData.exit_code !== null ? ` (exit_code=${statusData.exit_code})` : "";
      state.terminalLogs += `\n[系统状态更新: ${statusData.status}${exitInfo}]\n`;
      if (pre) pre.textContent = state.terminalLogs;
      if (isTerminalJob(statusData.status)) {
        terminalReceived = true;
        setTerminalConnection(`任务已结束：${statusData.status}`);
        source.close();
        if (state.eventSource === source) state.eventSource = null;
      }
      refreshJobs();
    } catch (_) {}
  });

  source.onerror = async () => {
    if (terminalReceived || !isCurrent()) return;
    source.close();
    let latest = null;
    try {
      latest = await getJobSnapshot(jobId);
    } catch (_) {}
    if (!sessionIsCurrent() || state.eventSource !== source) return;
    if (latest && isTerminalJob(latest.status)) {
      if (state.eventSource === source) state.eventSource = null;
      setTerminalConnection(`任务已结束：${latest.status}`);
      await refreshJobs();
      return;
    }
    if (state.eventSource === source) state.eventSource = null;
    const nextAttempt = Math.min(attempt + 1, 6);
    const delay = Math.min(8000, 500 * (2 ** Math.min(nextAttempt, 4)));
    setTerminalConnection(`日志流暂时断开，将在 ${(delay / 1000).toFixed(1)} 秒后重连...`);
    state.terminalRetryTimer = window.setTimeout(() => {
      state.terminalRetryTimer = null;
      connectTerminalStream(jobId, session, nextAttempt);
    }, delay);
  };
}

function closeTerminal() {
  state.terminalSession += 1;
  state.logDrawerOpen = false;
  const drawer = document.getElementById("terminal-drawer");
  if (drawer) {
    drawer.classList.add("hidden");
    drawer.classList.remove("terminal-fullscreen");
    drawer.setAttribute("aria-expanded", "false");
    const fullscreenButton = document.getElementById("terminal-fullscreen-btn");
    if (fullscreenButton) {
      fullscreenButton.textContent = "⛶ 全屏";
      fullscreenButton.setAttribute("aria-label", "终端全屏");
    }
  }
  stopTerminalConnection();
  updateJobPolling();
}

function toggleTerminalFullscreen() {
  const drawer = document.getElementById("terminal-drawer");
  if (!drawer) return;
  const fullscreen = drawer.classList.toggle("terminal-fullscreen");
  const button = document.getElementById("terminal-fullscreen-btn");
  if (button) {
    button.textContent = fullscreen ? "⛶ 还原" : "⛶ 全屏";
    button.setAttribute("aria-label", fullscreen ? "还原终端大小" : "终端全屏");
  }
}

async function cancelJob(jobId) {
  if (!confirm(`确认终止任务 ${jobId} 吗？`)) return;
  try {
    const res = await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    showToast("已发送终止指令", "warning");
    await loadJobs();
  } catch (err) {
    showToast(`终止失败: ${err.message}`, "error");
  }
}

async function resumeJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/resume`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const newJob = await res.json();
    showToast(`已提交断点续跑任务: ${newJob.id}`, "success");
    await loadJobs();
    openTerminal(newJob.id);
  } catch (err) {
    showToast(`续跑失败: ${err.message}`, "error");
  }
}

async function deleteJob(jobId) {
  if (!confirm(`确认删除任务 ${jobId} 及其落盘日志吗？此操作不可逆。`)) return;
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    showToast(`任务 ${jobId} 已删除`, "success");
    // 若当前正在查看该任务终端抽屉，则同步关闭
    if (state.currentJobId === jobId) {
      closeTerminal();
    }
    await loadJobs();
  } catch (err) {
    showToast(`删除失败: ${err.message}`, "error");
  }
}

async function clearFinishedJobs() {
  const finishedJobs = state.jobs.filter((j) => j.status !== "running" && j.status !== "queued");
  if (!finishedJobs.length) {
    showToast("当前没有可清理的已结束任务", "info");
    return;
  }
  if (!confirm(`确认清理所有已结束的 ${finishedJobs.length} 个历史任务记录及日志吗？此操作不可逆。`)) return;
  try {
    const res = await fetch("/api/jobs", { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    showToast(`成功清理 ${data.deleted_count || finishedJobs.length} 个已结束任务`, "success");
    if (finishedJobs.some((j) => j.id === state.currentJobId)) {
      closeTerminal();
    }
    await loadJobs();
  } catch (err) {
    showToast(`清理失败: ${err.message}`, "error");
  }
}

// --------------------------------------------------------------------------
// 评测结果与指标 (Runs) - 双模态全面支持 (field-eval / eval)
// --------------------------------------------------------------------------
async function loadRuns() {
  if (!state.currentDataset) {
    showRunsEmptyState("请先选择一个数据集");
    return;
  }

  // 更新 Runs 页面数据集下拉框
  const dsSelect = document.getElementById("runs-dataset-select");
  if (dsSelect && state.datasets.length) {
    dsSelect.innerHTML = state.datasets
      .map((d) => `<option value="${escapeHtml(d.name)}" ${d.name === state.currentDataset ? "selected" : ""}>${escapeHtml(d.name)}</option>`)
      .join("");
  }

  try {
    const res = await apiFetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/runs`);
    state.runs = await res.json();

    if (!state.runs.length) {
      showRunsEmptyState("当前数据集暂无已完成的评测结果");
      return;
    }

    // 默认选中第一个 Run (或保留当前选中的下标)
    if (state.selectedRunIndex >= state.runs.length) {
      state.selectedRunIndex = 0;
    }
    state.selectedRun = state.runs[state.selectedRunIndex];

    renderRunsSelector();
    updateRunMethodState();
    await renderActiveRunView();
  } catch (err) {
    showToast(`获取评测结果失败: ${err.message}`, "error");
  }
}

function showRunsEmptyState(msg) {
  const emptyContainer = document.getElementById("runs-empty-container");
  const fieldSec = document.getElementById("runs-field-eval-section");
  const evalSec = document.getElementById("runs-eval-section");
  const methodSwitch = document.getElementById("runs-method-switch");
  const runsSelect = document.getElementById("runs-select");
  const banner = document.getElementById("run-stale-banner");
  const failLink = document.getElementById("run-failures-link");
  const fieldLink = document.getElementById("run-field-html-link");

  if (fieldSec) fieldSec.style.display = "none";
  if (evalSec) evalSec.style.display = "none";
  if (methodSwitch) methodSwitch.style.display = "none";
  if (banner) banner.style.display = "none";
  if (failLink) failLink.style.display = "none";
  if (fieldLink) fieldLink.style.display = "none";
  if (runsSelect) runsSelect.innerHTML = `<option value="">(无已完成 Run)</option>`;

  if (emptyContainer) {
    emptyContainer.style.display = "block";
    emptyContainer.innerHTML = `
      <div class="empty-state-card">
        <div class="empty-state-icon">📊</div>
        <div class="empty-state-title">${escapeHtml(msg)}</div>
        <div class="empty-state-desc">
          尚未在该数据集下检测到 <code>field_metrics.json</code> 或 <code>metrics.json</code> 产物。<br>
          您可以前往【任务队列】或点击下方按钮直接启动评测：
        </div>
        <div style="display: flex; gap: 0.85rem; justify-content: center;">
          <button class="btn btn-primary" style="background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);" onclick="openJobModal('field-eval')">
            🏷️ 启动字段抽取评测 (field-eval)
          </button>
          <button class="btn btn-primary" onclick="openJobModal('eval')">
            🚀 启动对话打分评测 (eval)
          </button>
        </div>
      </div>`;
  }
}

function renderRunsSelector() {
  const select = document.getElementById("runs-select");
  const banner = document.getElementById("run-stale-banner");
  const emptyContainer = document.getElementById("runs-empty-container");
  if (emptyContainer) emptyContainer.style.display = "none";

  if (select) {
    select.innerHTML = state.runs
      .map((r, idx) => {
        let tag = "";
        if (r.has_field_eval && r.has_eval) tag = " [field-eval & eval]";
        else if (r.has_field_eval) tag = " [field-eval]";
        else if (r.has_eval) tag = " [eval]";
        if (r.is_stale) tag += " (已过期)";

        return `<option value="${idx}" ${idx === state.selectedRunIndex ? "selected" : ""}>
          ${escapeHtml(r.model)} / ${escapeHtml(r.backend)}${tag}
        </option>`;
      })
      .join("");
  }

  if (banner && state.selectedRun) {
    banner.style.display = state.selectedRun.is_stale ? "flex" : "none";
    const reasonEl = document.getElementById("run-stale-reason");
    if (reasonEl) reasonEl.textContent = state.selectedRun.stale_reason || "数据集已被修改，与该结果存在样本错位风险";
  }
}

function updateRunMethodState() {
  const r = state.selectedRun;
  const switchBox = document.getElementById("runs-method-switch");
  if (!r) return;

  // 根据当前 run 的实际产物，自适应显示/隐藏切换器
  if (r.has_field_eval && r.has_eval) {
    if (switchBox) switchBox.style.display = "inline-flex";
  } else {
    if (switchBox) switchBox.style.display = "none";
    if (r.has_field_eval) state.activeRunMethod = "field-eval";
    else if (r.has_eval) state.activeRunMethod = "eval";
  }

  // 更新分段按钮高亮
  const tabFE = document.getElementById("runs-tab-field-eval");
  const tabEval = document.getElementById("runs-tab-eval");
  if (tabFE) tabFE.classList.toggle("active", state.activeRunMethod === "field-eval");
  if (tabEval) tabEval.classList.toggle("active", state.activeRunMethod === "eval");
}

function switchRunMethod(method) {
  state.activeRunMethod = method;
  updateRunMethodState();
  renderActiveRunView();
}

async function onRunSelected(idxStr) {
  const idx = parseInt(idxStr, 10);
  if (isNaN(idx) || idx < 0 || idx >= state.runs.length) return;
  state.selectedRunIndex = idx;
  state.selectedRun = state.runs[idx];
  renderRunsSelector();
  updateRunMethodState();
  await renderActiveRunView();
}

async function renderActiveRunView() {
  const r = state.selectedRun;
  if (!r) return;

  const fieldSec = document.getElementById("runs-field-eval-section");
  const evalSec = document.getElementById("runs-eval-section");
  const failLink = document.getElementById("run-failures-link");
  const fieldLink = document.getElementById("run-field-html-link");

  if (state.activeRunMethod === "field-eval") {
    if (fieldSec) fieldSec.style.display = "block";
    if (evalSec) evalSec.style.display = "none";
    if (failLink) failLink.style.display = "none";

    // field-eval HTML 报告链接
    if (fieldLink) {
      fieldLink.style.display = r.has_field_mismatches_html ? "inline-flex" : "none";
      fieldLink.href = `/api/datasets/${encodeURIComponent(state.currentDataset)}/runs/${encodeURIComponent(r.model)}/${encodeURIComponent(r.backend)}/field-mismatches.html`;
    }

    await loadFieldMetrics();
    await loadFieldMismatches(0);
  } else {
    if (fieldSec) fieldSec.style.display = "none";
    if (evalSec) evalSec.style.display = "block";
    if (fieldLink) fieldLink.style.display = "none";

    // eval HTML 报告链接
    if (failLink) {
      failLink.style.display = r.has_failures_html ? "inline-flex" : "none";
      failLink.href = `/api/datasets/${encodeURIComponent(state.currentDataset)}/runs/${encodeURIComponent(r.model)}/${encodeURIComponent(r.backend)}/failures.html`;
    }

    await loadRunMetrics();
    await loadScored(0);
  }

  // 刷新当前数据集全部 HTML 报告统计
  updateDatasetHtmlCount();
}

function rerunCurrentRun(method = null) {
  const run = state.selectedRun;
  const targetMethod = method || state.activeRunMethod || (run?.has_field_eval ? "field-eval" : "eval");
  openJobModal(targetMethod, {
    dataset: state.currentDataset,
    model: run?.model || "",
    backend: run?.backend || "",
  });
}

async function updateDatasetHtmlCount() {
  const countEl = document.getElementById("dataset-html-count");
  if (!state.currentDataset) {
    if (countEl) countEl.textContent = "0";
    return;
  }
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/html-files`);
    if (res.ok) {
      const files = await res.json();
      if (countEl) countEl.textContent = String(files.length);
    }
  } catch (_) {}
}

async function openDatasetHtmlModal(datasetName = null) {
  const ds = datasetName || state.currentDataset;
  if (!ds) {
    showToast("请先选择一个活动数据集", "warning");
    return;
  }
  const modal = document.getElementById("dataset-html-modal");
  const subTitle = document.getElementById("dataset-html-modal-subtitle");
  const listEl = document.getElementById("dataset-html-list");
  const searchInput = document.getElementById("dataset-html-search");
  if (subTitle) subTitle.textContent = `当前数据集: ${ds}`;
  if (searchInput) searchInput.value = "";
  if (listEl) listEl.innerHTML = `<div style="color: var(--text-muted); padding: 1.5rem; text-align: center;">正在检索当前数据集目录下的 HTML 报告文件...</div>`;

  state.datasetHtmlFiles = [];
  state.datasetHtmlCategory = "all";
  state.datasetHtmlSearch = "";
  updateDatasetHtmlFilterTabs();

  closeHtmlPreview();
  if (modal) modal.showModal();

  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(ds)}/html-files`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const files = await res.json();
    state.datasetHtmlFiles = Array.isArray(files) ? files : [];

    const countEl = document.getElementById("dataset-html-count");
    if (countEl) countEl.textContent = String(state.datasetHtmlFiles.length);

    renderDatasetHtmlList();
  } catch (err) {
    if (listEl) listEl.innerHTML = `<div style="color: var(--rose-500); padding: 1rem;">加载 HTML 报告列表失败: ${escapeHtml(err.message)}</div>`;
  }
}

function updateDatasetHtmlFilterTabs() {
  const cat = state.datasetHtmlCategory || "all";
  ["all", "field", "eval"].forEach((c) => {
    const btn = document.getElementById(`dataset-html-flt-${c}`);
    if (btn) btn.classList.toggle("active", c === cat);
  });
}

function filterDatasetHtmlCategory(cat) {
  state.datasetHtmlCategory = cat;
  updateDatasetHtmlFilterTabs();
  renderDatasetHtmlList();
}

function filterDatasetHtmlFiles(query) {
  state.datasetHtmlSearch = (query || "").trim().toLowerCase();
  renderDatasetHtmlList();
}

function renderDatasetHtmlList() {
  const listEl = document.getElementById("dataset-html-list");
  if (!listEl) return;

  const files = state.datasetHtmlFiles || [];
  if (files.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state-card" style="padding: 1.75rem 1rem; margin: 0;">
        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📄</div>
        <div style="color: #cbd5e1; font-weight: 600; font-size: 1rem; margin-bottom: 0.35rem;">当前数据集暂未生成任何 HTML 报告</div>
        <div style="font-size: 0.82rem; color: var(--text-muted); max-width: 460px; margin: 0 auto 1.25rem; line-height: 1.5;">
          运行 <code>field-eval</code> 任务将生成 <code>field_mismatches.html</code> 可视化失配清单；运行 <code>eval</code> 对话打分任务若有未命中样本将生成 <code>failures.html</code> 报告。
        </div>
        <div style="display: flex; gap: 0.85rem; justify-content: center; flex-wrap: wrap;">
          <button class="btn btn-sm btn-primary" onclick="closeDatasetHtmlModal(); openJobModal('field-eval')">🏷️ 发起 field-eval 评测</button>
          <button class="btn btn-sm btn-danger" onclick="closeDatasetHtmlModal(); openJobModal('eval')">💬 发起 eval 评测</button>
        </div>
      </div>
    `;
    return;
  }

  const cat = state.datasetHtmlCategory || "all";
  const search = state.datasetHtmlSearch || "";

  const filtered = files.filter((f) => {
    const nameLower = (f.name || "").toLowerCase();
    const pathLower = (f.path || "").toLowerCase();

    // 分类筛选
    if (cat === "field" && !nameLower.includes("mismatch")) return false;
    if (cat === "eval" && !nameLower.includes("failure")) return false;

    // 关键词搜索
    if (search && !pathLower.includes(search) && !nameLower.includes(search)) {
      return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    listEl.innerHTML = `
      <div style="text-align: center; padding: 2rem 1rem; color: var(--text-muted); background: rgba(0,0,0,0.2); border-radius: var(--radius-md); border: 1px dashed var(--border-subtle);">
        <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🔍</div>
        <div>未找到符合筛选条件的 HTML 报告</div>
        <button class="btn btn-sm" style="margin-top: 0.75rem;" onclick="filterDatasetHtmlCategory('all'); document.getElementById('dataset-html-search').value=''; filterDatasetHtmlFiles('');">重置筛选</button>
      </div>
    `;
    return;
  }

  listEl.innerHTML = filtered
    .map((f) => {
      const sizeKb = (f.size / 1024).toFixed(1);
      const mtimeStr = f.modified_at ? new Date(f.modified_at).toLocaleString() : "未知";
      const nameLower = (f.name || "").toLowerCase();

      let badge = '<span class="html-report-badge badge-generic">📄 报告</span>';
      let icon = '📄';
      if (nameLower.includes("mismatch")) {
        badge = '<span class="html-report-badge badge-mismatch">🏷️ 字段失配清单</span>';
        icon = '🏷️';
      } else if (nameLower.includes("failure")) {
        badge = '<span class="html-report-badge badge-failures">💬 对话未命中报告</span>';
        icon = '💬';
      }

      return `
        <div class="html-report-card">
          <div style="display: flex; align-items: center; gap: 0.85rem; min-width: 0; flex: 1;">
            <span style="font-size: 1.35rem; flex-shrink: 0;">${icon}</span>
            <div style="min-width: 0; flex: 1;">
              <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 4px;">
                ${badge}
                <span style="font-weight: 600; color: #fff; font-size: 0.86rem; font-family: var(--font-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(f.path)}">
                  ${escapeHtml(f.path)}
                </span>
              </div>
              <div style="font-size: 0.74rem; color: var(--text-dim); display: flex; gap: 1.15rem; align-items: center; flex-wrap: wrap;">
                <span>大小: <strong style="color: var(--text-secondary);">${sizeKb} KB</strong></span>
                <span>生成时间: <strong style="color: var(--text-secondary);">${mtimeStr}</strong></span>
              </div>
            </div>
          </div>
          <div style="display: flex; gap: 0.5rem; align-items: center; flex-shrink: 0; margin-left: 0.5rem;">
            <button class="btn btn-sm" onclick="previewDatasetHtml('${escapeHtml(f.url)}', '${escapeHtml(f.path)}')">
              👁️ 内嵌预览
            </button>
            <a href="${escapeHtml(f.url)}" target="_blank" class="btn btn-sm btn-primary" style="text-decoration: none;">
              新窗口打开 ↗
            </a>
          </div>
        </div>
      `;
    })
    .join("");
}

function closeDatasetHtmlModal() {
  const modal = document.getElementById("dataset-html-modal");
  closeHtmlPreview();
  if (modal) modal.close();
}

function previewDatasetHtml(url, name) {
  const previewBox = document.getElementById("dataset-html-preview-box");
  const iframe = document.getElementById("dataset-html-preview-iframe");
  const title = document.getElementById("dataset-html-preview-title");
  const extLink = document.getElementById("dataset-html-preview-external");

  if (previewBox && iframe) {
    previewBox.style.display = "block";
    iframe.src = url;
    if (title) title.textContent = `预览报告: ${name}`;
    if (extLink) extLink.href = url;
    previewBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function closeHtmlPreview() {
  const previewBox = document.getElementById("dataset-html-preview-box");
  const iframe = document.getElementById("dataset-html-preview-iframe");
  if (previewBox) previewBox.style.display = "none";
  if (iframe) iframe.src = "about:blank";
}

// --------------------------------------------------------------------------
// field-eval 专属数据加载与看板渲染
// --------------------------------------------------------------------------
async function loadFieldMetrics() {
  if (!state.selectedRun) return;
  const { model, backend } = state.selectedRun;
  try {
    const res = await fetch(
      `/api/datasets/${encodeURIComponent(state.currentDataset)}/runs/${encodeURIComponent(model)}/${encodeURIComponent(backend)}/field-metrics`
    );
    if (res.ok) {
      state.fieldMetrics = await res.json();
      renderFieldMetrics();
    }
  } catch (_) {}
}

function renderFieldMetrics() {
  const m = state.fieldMetrics;
  if (!m) return;

  const ov = m.overall || {};
  const microAcc = ov.micro_accuracy ?? m.micro_accuracy;
  const macroAcc = ov.macro_accuracy ?? m.macro_accuracy;
  const exactMatchRate = ov.exact_match_rate ?? m.exact_match_rate ?? m.exact_match_ratio;
  const numScored = m.num_scored ?? m.evaluated_samples ?? m.total_samples ?? 0;
  const numMissing = m.num_pred_missing ?? 0;

  const microEl = document.getElementById("field-metric-micro");
  const macroEl = document.getElementById("field-metric-macro");
  const exactEl = document.getElementById("field-metric-exact");
  const countEl = document.getElementById("field-metric-count");

  if (microEl) microEl.textContent = microAcc !== undefined && microAcc !== null ? `${(microAcc * 100).toFixed(1)}%` : "—";
  if (macroEl) macroEl.textContent = macroAcc !== undefined && macroAcc !== null ? `${(macroAcc * 100).toFixed(1)}%` : "—";
  if (exactEl) exactEl.textContent = exactMatchRate !== undefined && exactMatchRate !== null ? `${(exactMatchRate * 100).toFixed(1)}%` : "—";
  if (countEl) countEl.textContent = `${numScored} (未输出: ${numMissing})`;

  // 渲染逐字段细分看板 (Per-Field Grid)
  const grid = document.getElementById("field-metrics-grid");
  if (!grid) return;

  const perField = m.per_field || (m.fields && !Array.isArray(m.fields) ? m.fields : {});
  const fieldList = Array.isArray(m.fields) ? m.fields : Object.keys(perField);

  grid.innerHTML = fieldList
    .map((f) => {
      const data = perField[f] || { accuracy: 0, correct: 0, total: 0 };
      const correct = data.correct ?? data.match ?? 0;
      const total = data.total ?? 0;
      const acc = data.accuracy !== undefined ? data.accuracy : (total ? correct / total : 0);
      const accPct = (acc * 100).toFixed(1);
      const emptyCount = data.empty_count ?? 0;
      const nonEmptyCount = data.non_empty_count ?? total;
      const overallAcc = data.overall_accuracy !== undefined ? (data.overall_accuracy * 100).toFixed(1) : accPct;
      let barColor = "var(--emerald-500)";
      if (acc < 0.7) barColor = "var(--rose-500)";
      else if (acc < 0.9) barColor = "var(--amber-500)";

      return `
      <div class="field-metric-card">
        <div class="field-metric-header">
          <span class="field-name-title">${escapeHtml(f)}</span>
          <span class="field-acc-pct" style="color: ${barColor};" title="非空准确率">${accPct}%</span>
        </div>
        <div class="field-progress-track">
          <div class="field-progress-bar" style="width: ${accPct}%; background: ${barColor};"></div>
        </div>
        <div class="field-metric-footer" style="display: flex; flex-direction: column; gap: 4px; font-size: 12px;">
          <div style="display: flex; justify-content: space-between;">
            <span>非空: <strong>${correct}</strong> / ${nonEmptyCount}</span>
            <span>空样本: <strong>${emptyCount}</strong></span>
          </div>
          <div style="display: flex; justify-content: space-between; color: var(--text-dim, #64748b);">
            <span>总体准确率: <strong>${overallAcc}%</strong></span>
            <span>非空失配: <strong>${nonEmptyCount - correct}</strong></span>
          </div>
        </div>
      </div>
    `;
    })
    .join("");
}

async function loadFieldMismatches(offset = 0) {
  if (!state.selectedRun) return;
  state.fieldMismatchesOffset = offset;
  const { model, backend } = state.selectedRun;

  const q = new URLSearchParams({
    offset: state.fieldMismatchesOffset,
    limit: state.fieldMismatchesLimit,
  });
  if (state.fieldMismatchesFilter) {
    q.append("filter_state", state.fieldMismatchesFilter);
  }

  const tbody = document.getElementById("field-mismatches-tbody");
  if (tbody) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:2.5rem; color:var(--text-dim);">加载逐字段失配记录中...</td></tr>`;
  }

  try {
    const res = await fetch(
      `/api/datasets/${encodeURIComponent(state.currentDataset)}/runs/${encodeURIComponent(model)}/${encodeURIComponent(backend)}/field-mismatches?${q.toString()}`
    );
    if (res.ok) {
      const data = await res.json();
      state.fieldMismatches = data.records;
      state.fieldMismatchesTotal = data.total;
      renderFieldMismatches();
    }
  } catch (_) {}
}

function renderFieldMismatches() {
  const tbody = document.getElementById("field-mismatches-tbody");
  if (!tbody) return;

  if (!state.fieldMismatches.length) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:3rem; color:var(--emerald-500);">🎉 该条件下无任何失配记录（全部准确命中）</td></tr>`;
    return;
  }

  tbody.innerHTML = state.fieldMismatches
    .map((row) => {
      let stateBadge = `<span class="role-badge" style="background:var(--rose-bg); color:var(--rose-500); border-color:var(--rose-border);">失配</span>`;
      if (row.state === "pred_missing") {
        stateBadge = `<span class="role-badge" style="background:rgba(245,158,11,0.15); color:var(--amber-500); border-color:var(--amber-border);">未产出描述</span>`;
      }

      // 比对字段详情
      const fieldsHtml = (row.fields || [])
        .map((f) => {
          const isMatch = f.correct;
          const refStr = (f.ref && f.ref.length) ? f.ref.join("、") : "(无)";
          const predStr = (f.pred && f.pred.length) ? f.pred.join("、") : (row.state === "pred_missing" ? "(未输出)" : "(无)");
          return `
          <div class="diff-tag-row ${isMatch ? "match" : "mismatch"}">
            <span class="diff-field-name">${escapeHtml(f.field)}:</span>
            <span class="diff-ref">标准[${escapeHtml(refStr)}]</span>
            <span style="color:var(--text-dim); margin: 0 2px;">vs</span>
            <span class="diff-pred" style="color: ${isMatch ? "var(--emerald-500)" : "#fb7185"};">
              模型[${escapeHtml(predStr)}] ${isMatch ? "✓" : "✗"}
            </span>
          </div>`;
        })
        .join("");

      return `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 600; font-size: 0.8rem;">${escapeHtml(row.id)}</td>
        <td>${stateBadge}</td>
        <td style="font-size: 0.82rem; line-height: 1.5; color: var(--text-secondary); word-break: break-word; max-width: 350px;">
          ${escapeHtml(row.pred_desc || "—")}
        </td>
        <td>
          <div class="diff-tag-group">
            ${fieldsHtml}
          </div>
        </td>
      </tr>
    `;
    })
    .join("");
}

// --------------------------------------------------------------------------
// eval 对话打分专属数据加载与看板渲染
// --------------------------------------------------------------------------
async function loadRunMetrics() {
  if (!state.selectedRun) return;
  const { model, backend } = state.selectedRun;
  try {
    const res = await fetch(
      `/api/datasets/${encodeURIComponent(state.currentDataset)}/runs/${encodeURIComponent(model)}/${encodeURIComponent(backend)}/metrics`
    );
    if (res.ok) {
      state.runMetrics = await res.json();
      renderMetrics();
    }
  } catch (_) {}
}

function renderMetrics() {
  const m = state.runMetrics;
  if (!m) return;
  const scoreEl = document.getElementById("metric-score");
  const countEl = document.getElementById("metric-count");
  const failedEl = document.getElementById("metric-failed");
  const scorerEl = document.getElementById("metric-scorer");

  if (scoreEl) scoreEl.textContent = m.overall_mean_score !== undefined ? m.overall_mean_score : "—";
  if (countEl) countEl.textContent = m.num_samples || 0;
  if (failedEl) failedEl.textContent = m.num_failed_targets || 0;
  if (scorerEl) scorerEl.textContent = m.scorer || "—";
}

async function loadScored(offset = 0) {
  if (!state.selectedRun) return;
  state.scoredOffset = offset;
  const { model, backend } = state.selectedRun;
  const q = new URLSearchParams({
    offset: state.scoredOffset,
    limit: state.scoredLimit,
    order: state.scoredOrder,
  });

  try {
    const res = await fetch(
      `/api/datasets/${encodeURIComponent(state.currentDataset)}/runs/${encodeURIComponent(model)}/${encodeURIComponent(backend)}/scored?${q.toString()}`
    );
    if (res.ok) {
      const data = await res.json();
      state.scoredRecords = data.records;
      state.scoredTotal = data.total;
      renderScored();
    }
  } catch (_) {}
}

function renderScored() {
  const tbody = document.getElementById("scored-tbody");
  if (!tbody) return;

  if (!state.scoredRecords.length) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding:3rem; color:var(--text-dim);">无评测得分记录</td></tr>`;
    return;
  }

  tbody.innerHTML = state.scoredRecords
    .map((r) => {
      const isGood = r.score >= 1.0;
      const scoreColor = isGood ? "var(--emerald-500)" : "var(--rose-500)";
      return `
      <tr>
        <td style="font-family: var(--font-mono); font-size: 0.78rem;">${escapeHtml(r.id)}</td>
        <td>${r.turn}</td>
        <td><strong style="color: ${scoreColor};">${r.score}</strong></td>
        <td style="max-width: 320px; font-size: 0.8rem; word-break: break-all;">${escapeHtml(r.prediction)}</td>
        <td style="max-width: 320px; font-size: 0.8rem; word-break: break-all; color: var(--text-muted);">${escapeHtml(r.reference)}</td>
      </tr>
    `;
    })
    .join("");
}

// --------------------------------------------------------------------------
// 健康体检 (Health)
// --------------------------------------------------------------------------
async function loadHealth() {
  if (!state.currentDataset) {
    const container = document.getElementById("health-content");
    if (container) {
      container.innerHTML = `
        <div class="empty-state-card">
          <div class="empty-state-icon">🩺</div>
          <div class="empty-state-title">未选定数据集</div>
          <div class="empty-state-desc">请先在【数据集】页面选择目标数据集以执行深度体检。</div>
          <button class="btn btn-primary" onclick="switchTab('datasets')">去选择数据集</button>
        </div>`;
    }
    return;
  }
  const container = document.getElementById("health-content");
  if (container) {
    container.innerHTML = `<div style="text-align:center; padding:4rem; color:var(--text-dim);">正在深度体检中...</div>`;
  }
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.healthReport = await res.json();
    renderHealth();
  } catch (err) {
    showToast("获取健康诊断失败", "error");
  }
}

function renderHealth() {
  const container = document.getElementById("health-content");
  const rep = state.healthReport;
  if (!container || !rep) return;

  let bannerHtml = "";
  if (rep.is_healthy) {
    bannerHtml = `
      <div style="background: var(--emerald-bg); border: 1px solid var(--emerald-border); color: var(--emerald-500); padding: 1.25rem; border-radius: var(--radius-md); font-weight: 600; display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
        <span style="font-size: 1.4rem;">✅</span>
        <span>数据集健康状态完美：未发现任何缺失图片、&lt;image&gt; 占位符失配或重复 ID 异常！</span>
      </div>`;
  } else {
    bannerHtml = `
      <div style="background: var(--rose-bg); border: 1px solid var(--rose-border); color: #fb7185; padding: 1.25rem; border-radius: var(--radius-md); font-weight: 600; display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
        <span style="font-size: 1.4rem;">⚠️</span>
        <span>数据集发现异常，请根据以下清单核对清理坏样本！</span>
      </div>`;
  }

  let missingImgsHtml = `<p style="color: var(--text-dim); font-size: 0.85rem;">无缺失图片。</p>`;
  if (rep.missing_images && rep.missing_images.length > 0) {
    missingImgsHtml = `
      <ul style="padding-left: 1.25rem; font-size: 0.85rem; color: #fb7185;">
        ${rep.missing_images
          .map(
            (m) => `
          <li style="margin-bottom: 0.35rem;">
            样本 <strong>${escapeHtml(m.sample_id)}</strong> (位置 #${m.position}): <code>${escapeHtml(m.ref)}</code>
          </li>`
          )
          .join("")}
      </ul>`;
  }

  let placeholderHtml = `<p style="color: var(--text-dim); font-size: 0.85rem;">无占位符失配。</p>`;
  if (rep.placeholder_mismatches && rep.placeholder_mismatches.length > 0) {
    placeholderHtml = `
      <ul style="padding-left: 1.25rem; font-size: 0.85rem; color: var(--amber-500);">
        ${rep.placeholder_mismatches
          .map(
            (p) => `
          <li style="margin-bottom: 0.35rem;">
            样本 <strong>${escapeHtml(p.sample_id)}</strong>: 对话内占位符 <strong>${p.placeholder_count}</strong> 个 vs 实际图片 <strong>${p.images_count}</strong> 张
          </li>`
          )
          .join("")}
      </ul>`;
  }

  container.innerHTML = `
    ${bannerHtml}
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
      <div class="dataset-card">
        <h3 style="color: #fb7185; margin-bottom: 0.75rem; font-size: 1.05rem;">❌ 磁盘缺失图片引用 (${rep.missing_images_count || 0} 张)</h3>
        ${missingImgsHtml}
      </div>
      <div class="dataset-card">
        <h3 style="color: var(--amber-500); margin-bottom: 0.75rem; font-size: 1.05rem;">⚠️ &lt;image&gt; 占位符失配 (${rep.placeholder_mismatches_count || 0} 条)</h3>
        ${placeholderHtml}
      </div>
    </div>
  `;
}

// --------------------------------------------------------------------------
// HuggingFace 转 GGUF 可视化工坊 (GGUF Conversion Workshop)
// --------------------------------------------------------------------------
function updateGGUFConvertHfOptions() {
  const sel = document.getElementById("gguf-conv-hf-select");
  if (!sel) return;
  const cur = sel.value;
  const opts = (state.models.hf_models || [])
    .map((m) => `<option value="${escapeHtml(m.path)}" data-name="${escapeHtml(m.name)}">🤗 ${escapeHtml(m.name)}</option>`)
    .join("");
  sel.innerHTML = `<option value="">(从已探测 HF 模型中快速选择)</option>` + opts;
  if (cur) sel.value = cur;
}

function openGGUFConvertModal() {
  const modal = document.getElementById("gguf-convert-modal");
  if (!modal) return;
  updateGGUFConvertHfOptions();
  updateGGUFConvertPreview();
  modal.showModal();
}

function closeGGUFConvertModal() {
  const modal = document.getElementById("gguf-convert-modal");
  if (modal) modal.close();
}

function onGGUFConvertHfSelect(val) {
  const pathInput = document.getElementById("gguf-conv-hf-path");
  const nameInput = document.getElementById("gguf-conv-name");
  if (pathInput) pathInput.value = val || "";
  if (val && nameInput && !nameInput.value.trim()) {
    const sel = document.getElementById("gguf-conv-hf-select");
    const opt = sel?.options[sel.selectedIndex];
    const detectedName = opt?.getAttribute("data-name");
    if (detectedName) {
      nameInput.value = detectedName;
    } else {
      const parts = val.replace(/\\/g, "/").replace(/\/+$/, "").split("/");
      nameInput.value = parts[parts.length - 1] || "";
    }
  }
  updateGGUFConvertPreview();
}

function onGGUFConvertHfInput() {
  const pathInput = document.getElementById("gguf-conv-hf-path");
  const nameInput = document.getElementById("gguf-conv-name");
  const val = pathInput?.value.trim() || "";
  if (val && nameInput && !nameInput.value.trim()) {
    const parts = val.replace(/\\/g, "/").replace(/\/+$/, "").split("/");
    nameInput.value = parts[parts.length - 1] || "";
  }
  updateGGUFConvertPreview();
}

function toggleGGUFConvertMMProj(enabled) {
  const panel = document.getElementById("gguf-conv-mmproj-panel");
  if (panel) panel.style.display = enabled ? "flex" : "none";
  updateGGUFConvertPreview();
}

function updateGGUFConvertPreview() {
  const hfPath = document.getElementById("gguf-conv-hf-path")?.value.trim() || "";
  const name = document.getElementById("gguf-conv-name")?.value.trim() || "";
  const outtype = document.getElementById("gguf-conv-outtype")?.value || "bf16";
  const mmprojEnable = document.getElementById("gguf-conv-mmproj-enable")?.checked ?? true;
  const mmprojType = document.getElementById("gguf-conv-mmproj-type")?.value || "auto";
  const mmprojOuttype = document.getElementById("gguf-conv-mmproj-outtype")?.value || "f16";
  const quant = document.getElementById("gguf-conv-quant")?.value || "";
  const llamaCppDir = document.getElementById("gguf-conv-llamacpp-dir")?.value.trim() || "";
  const cleanInter = document.getElementById("gguf-conv-clean-intermediate")?.checked ?? false;

  const modelName = name || (hfPath ? hfPath.replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop() : "A");

  // 更新文件名只读展示
  const mmprojFileEl = document.getElementById("gguf-conv-mmproj-filename");
  if (mmprojFileEl) {
    mmprojFileEl.value = `${modelName}_${mmprojOuttype}_mmproj.gguf`;
  }

  const destHint = document.getElementById("gguf-conv-dest-hint");
  if (destHint) {
    destHint.textContent = `落地: <llamacpp_models_dir>/${modelName}/`;
  }

  const parts = ["python", "-m", "eval_vlm", "convert-gguf"];
  if (hfPath) parts.push("--hf-path", hfPath.includes(" ") ? `"${hfPath}"` : hfPath);
  if (name) parts.push("--name", name.includes(" ") ? `"${name}"` : name);
  if (outtype) parts.push("--outtype", outtype);
  if (!mmprojEnable) {
    parts.push("--no-mmproj");
  } else {
    if (mmprojType && mmprojType !== "auto") parts.push("--mmproj-type", mmprojType);
    if (mmprojOuttype) parts.push("--mmproj-outtype", mmprojOuttype);
  }
  if (quant) parts.push("--quantize", quant);
  if (cleanInter) parts.push("--clean-intermediate");
  if (llamaCppDir) parts.push("--llama-cpp-dir", llamaCppDir.includes(" ") ? `"${llamaCppDir}"` : llamaCppDir);

  const previewEl = document.getElementById("gguf-conv-cmd-preview");
  if (previewEl) previewEl.textContent = parts.join(" ");
}

async function submitGGUFConvertTask() {
  const hfPath = document.getElementById("gguf-conv-hf-path")?.value.trim();
  if (!hfPath) {
    showToast("请指定源 HuggingFace 模型权重目录", "warning");
    return;
  }

  const name = document.getElementById("gguf-conv-name")?.value.trim() || null;
  const outtype = document.getElementById("gguf-conv-outtype")?.value || "bf16";
  const isMultimodal = document.getElementById("gguf-conv-mmproj-enable")?.checked ?? true;
  const mmprojType = document.getElementById("gguf-conv-mmproj-type")?.value || "auto";
  const mmprojOuttype = document.getElementById("gguf-conv-mmproj-outtype")?.value || "f16";
  const quantize = document.getElementById("gguf-conv-quant")?.value || null;
  const llamaCppDir = document.getElementById("gguf-conv-llamacpp-dir")?.value.trim() || null;
  const cleanIntermediate = document.getElementById("gguf-conv-clean-intermediate")?.checked ?? false;

  const payload = {
    hf_path: hfPath,
    name: name,
    outtype: outtype,
    is_multimodal: isMultimodal,
    mmproj_outtype: mmprojOuttype,
    mmproj_type: (mmprojType && mmprojType !== "auto") ? mmprojType : null,
    quantize: quantize,
    clean_intermediate: cleanIntermediate,
    llama_cpp_dir: llamaCppDir,
  };

  try {
    const res = await fetch("/api/tools/convert-gguf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const job = await res.json();
    closeGGUFConvertModal();
    showToast(`GGUF 转换任务已成功提交入队: ${job.id}`, "success");
    await loadJobs();
    openTerminal(job.id);
  } catch (err) {
    showToast(`提交转换任务失败: ${err.message}`, "error");
  }
}

// --------------------------------------------------------------------------
// 浅色 / 暗黑主题系统
// --------------------------------------------------------------------------
function initTheme() {
  const savedTheme = localStorage.getItem("eval_vlm_theme") ||
    (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  applyTheme(savedTheme);
}

function applyTheme(theme) {
  if (theme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  const newTheme = currentTheme === "light" ? "dark" : "light";
  applyTheme(newTheme);
  localStorage.setItem("eval_vlm_theme", newTheme);
  showToast(`已切换至${newTheme === "light" ? "简洁明亮浅色" : "暗黑科技深色"}主题`, "info");
}

// --------------------------------------------------------------------------
// 页面初始化
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  // 初始化主题
  initTheme();

  // 绑定路由与监听
  window.addEventListener("hashchange", handleHash);
  await Promise.all([loadDatasets(), loadSettings(), loadModels()]);
  handleHash();
});

// 显式挂载到 window 供控制台调试及内联事件统一调用
window.state = state;
window.switchTab = switchTab;
window.handleHash = handleHash;
window.loadDatasets = loadDatasets;
window.loadSamples = loadSamples;
window.loadConfig = loadConfig;
window.loadJobs = loadJobs;
window.loadRuns = loadRuns;
window.loadTrash = loadTrash;
window.loadHealth = loadHealth;
window.selectDatasetAndNavigate = selectDatasetAndNavigate;
window.openLightbox = openLightbox;
window.closeLightbox = closeLightbox;
window.promptDelete = promptDelete;
window.closeDeleteModal = closeDeleteModal;
window.confirmDelete = confirmDelete;
window.saveSingleConfigKey = saveSingleConfigKey;
window.openJobModal = openJobModal;
window.closeJobModal = closeJobModal;
window.updateJobCommandPreview = updateJobCommandPreview;
window.confirmLaunchJob = confirmLaunchJob;
window.cancelJob = cancelJob;
window.resumeJob = resumeJob;
window.openTerminal = openTerminal;
window.closeTerminal = closeTerminal;
window.restoreTrashItem = restoreTrashItem;
window.onRunSelected = onRunSelected;
window.switchRunMethod = switchRunMethod;
window.loadScored = loadScored;
window.loadFieldMismatches = loadFieldMismatches;
window.showToast = showToast;
window.initInlineJobConsole = initInlineJobConsole;
window.switchInlineJobType = switchInlineJobType;
window.updateInlineJobPreview = updateInlineJobPreview;
window.submitInlineJob = submitInlineJob;

// 全局数据集下拉
window.toggleDatasetDropdown = toggleDatasetDropdown;
window.selectGlobalDataset = selectGlobalDataset;
window.filterDatasetDropdown = filterDatasetDropdown;
window.updateHeaderDatasetDropdown = updateHeaderDatasetDropdown;
window.updateHeaderDatasetPill = updateHeaderDatasetPill;

// Sweep 批量评测
window.loadSweepData = loadSweepData;
window.toggleSweepDataset = toggleSweepDataset;
window.toggleAllSweepDatasets = toggleAllSweepDatasets;
window.invertSweepDatasets = invertSweepDatasets;
window.filterSweepDatasets = filterSweepDatasets;
window.onSweepBackendChange = onSweepBackendChange;
window.onSweepModelSelect = onSweepModelSelect;
window.updateSweepCmdPreview = updateSweepCmdPreview;
window.submitSweepJob = submitSweepJob;

// 全局设置与模型
window.loadSettings = loadSettings;
window.saveSettings = saveSettings;
window.loadModels = loadModels;

// 可视化配置工坊
window.switchConfigSubTab = switchConfigSubTab;
window.setConfigBackend = setConfigBackend;
window.markConfigDirty = markConfigDirty;
window.onConfigEvalTargetsChange = onConfigEvalTargetsChange;
window.saveAllConfigChanges = saveAllConfigChanges;
window.onJobInlineModelSelectChange = onJobInlineModelSelectChange;
window.onJobInlineBackendChange = onJobInlineBackendChange;
window.onJobInlineTargetsChange = onJobInlineTargetsChange;
window.onJobInlineEvalTargetsChange = onJobInlineEvalTargetsChange;
window.onJobModalModelSelectChange = onJobModalModelSelectChange;
window.onJobModalBackendChange = onJobModalBackendChange;
window.onJobModalTargetsChange = onJobModalTargetsChange;
window.onJobModalEvalTargetsChange = onJobModalEvalTargetsChange;
window.onLlamaCppModeChange = onLlamaCppModeChange;
window.onLlamaCppModelSelectChange = onLlamaCppModelSelectChange;

// 主题切换
window.initTheme = initTheme;
window.applyTheme = applyTheme;
window.toggleTheme = toggleTheme;

// GGUF 转换工坊
window.openGGUFConvertModal = openGGUFConvertModal;
window.closeGGUFConvertModal = closeGGUFConvertModal;
window.updateGGUFConvertHfOptions = updateGGUFConvertHfOptions;
window.onGGUFConvertHfSelect = onGGUFConvertHfSelect;
window.onGGUFConvertHfInput = onGGUFConvertHfInput;
window.toggleGGUFConvertMMProj = toggleGGUFConvertMMProj;
window.updateGGUFConvertPreview = updateGGUFConvertPreview;
window.submitGGUFConvertTask = submitGGUFConvertTask;

// 重跑与 HTML 报告
window.rerunCurrentRun = rerunCurrentRun;
window.openDatasetHtmlModal = openDatasetHtmlModal;
window.closeDatasetHtmlModal = closeDatasetHtmlModal;
window.previewDatasetHtml = previewDatasetHtml;
window.closeHtmlPreview = closeHtmlPreview;
window.updateDatasetHtmlCount = updateDatasetHtmlCount;
window.filterDatasetHtmlCategory = filterDatasetHtmlCategory;
window.filterDatasetHtmlFiles = filterDatasetHtmlFiles;


// ==========================================================================
// Sweep 批量评测结果全自动探测与可视化看板 (Sweep Results Studio)
// ==========================================================================

async function loadSweepResultsData() {
  // 自动从后端探测工作目录下的 _sweep 记录。
  // loadSweepRunsList also selects the newest available run when needed.
  await loadSweepRunsList(false);
}

async function loadSweepRunsList(forceRefresh = false) {
  if (state.sweepResults.runsLoadPromise) return state.sweepResults.runsLoadPromise;

  const select = document.getElementById("sr-run-select");
  if (select) select.innerHTML = `<option value="">(正在扫描工作区 _sweep/ ...)</option>`;
  const requestId = ++state.sweepResults.runsRequestId;
  const request = (async () => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 5000);
    try {
      const res = await apiFetch("/api/sweep/runs", { signal: controller.signal });
      const runs = await res.json();
      if (requestId !== state.sweepResults.runsRequestId) return;

      state.sweepResults.runsList = Array.isArray(runs) ? runs : [];
      const activeExists = state.sweepResults.runsList.some((run) => run.path === state.sweepResults.activeRunPath);
      const selectedPath = activeExists ? state.sweepResults.activeRunPath : state.sweepResults.runsList[0]?.path || "";
      if (select) {
        if (!state.sweepResults.runsList.length) {
          select.innerHTML = `<option value="">(未在工作区 _sweep/ 下检测到评测记录)</option>`;
        } else {
          select.innerHTML = state.sweepResults.runsList.map((r) => {
            const timeStr = r.mtime ? new Date(r.mtime).toLocaleString("zh-CN") : "未知时间";
            const label = `${r.model || "默认模型"} [${r.backend || "默认后端"}] - ${timeStr} (${r.datasets_count || 0} 个数据集)`;
            return `<option value="${escapeHtml(r.path)}" ${r.path === selectedPath ? "selected" : ""}>${escapeHtml(label)}</option>`;
          }).join("");
        }
      }

      if (!selectedPath) {
        clearSweepResultsState();
        if (forceRefresh) showToast("工作目录下暂无 _sweep 记录", "info");
        return;
      }
      if (forceRefresh || !activeExists || !state.sweepResults.data) {
        await onSweepRunSelect(selectedPath);
      } else {
        renderSweepResultsDashboard();
      }
      if (forceRefresh) showToast(`已成功扫描到 ${state.sweepResults.runsList.length} 个 Sweep 运行记录`, "success");
    } catch (err) {
      if (requestId !== state.sweepResults.runsRequestId) return;
      console.warn("加载本地 sweep 列表失败:", err);
      clearSweepResultsState();
      if (select) {
        const message = err.name === "AbortError" ? "扫描超时" : `扫描失败: ${err.message}`;
        select.innerHTML = `<option value="">(${escapeHtml(message)}，请点击“重新扫描”重试)</option>`;
      }
      showToast("Sweep 结果扫描失败，请点击“重新扫描”重试", "error");
    } finally {
      window.clearTimeout(timeout);
    }
  })();
  state.sweepResults.runsLoadPromise = request;
  try {
    return await request;
  } finally {
    if (state.sweepResults.runsLoadPromise === request) {
      state.sweepResults.runsLoadPromise = null;
    }
  }
}

async function onSweepRunSelect(path) {
  if (!path) {
    clearSweepResultsState();
    return;
  }
  const requestId = ++state.sweepResults.selectionRequestId;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 5000);
  try {
    const res = await apiFetch(`/api/sweep/summary?path=${encodeURIComponent(path)}`, { signal: controller.signal });
    const data = await res.json();
    if (requestId !== state.sweepResults.selectionRequestId) return;
    parseAndSetSweepData(data, path);
    const select = document.getElementById("sr-run-select");
    if (select) select.value = path;
  } catch (err) {
    if (requestId !== state.sweepResults.selectionRequestId) return;
    clearSweepResultsState();
    const message = err.name === "AbortError" ? "读取 Sweep 记录超时" : `读取 Sweep 记录失败: ${err.message}`;
    showToast(`${message}，请点击“重新扫描”重试`, "error");
  } finally {
    window.clearTimeout(timeout);
  }
}

function clearSweepResultsState() {
  // Invalidate a late summary response before removing its associated view.
  state.sweepResults.selectionRequestId++;
  state.sweepResults.data = null;
  state.sweepResults.activeRunPath = "";
  state.sweepResults.selectedDatasetName = null;
  showSweepEmptyState();
}

function showSweepEmptyState() {
  const dashboard = document.getElementById("sr-dashboard");
  const emptyState = document.getElementById("sr-empty-state");
  const wsPathEl = document.getElementById("sr-empty-ws-path");
  if (dashboard) dashboard.style.display = "none";
  if (emptyState) emptyState.style.display = "block";
  if (wsPathEl && state.settings.workspace) {
    wsPathEl.textContent = `${state.settings.workspace}/_sweep/`;
  }
}

function parseAndSetSweepData(json, runPath = "") {
  state.sweepResults.data = json;
  state.sweepResults.activeRunPath = runPath;

  const results = json.results || [];
  if (results.length > 0 && (!state.sweepResults.selectedDatasetName || !results.some(r => r.dataset === state.sweepResults.selectedDatasetName))) {
    state.sweepResults.selectedDatasetName = results[0].dataset;
  }

  const emptyState = document.getElementById("sr-empty-state");
  const dashboard = document.getElementById("sr-dashboard");
  if (emptyState) emptyState.style.display = "none";
  if (dashboard) dashboard.style.display = "flex";

  renderSweepResultsDashboard();
}

function renderSweepResultsDashboard() {
  const data = state.sweepResults.data;
  if (!data) return;

  renderSweepHeroKpis(data);
  renderSweepOverviewTable(data);
  renderSweepSelectedDatasetDetail();
}

function renderSweepHeroKpis(data) {
  const results = data.results || [];
  const datasets = data.datasets || results.map(r => r.dataset);
  const first = results[0] || {};

  const modelName = first.model || "未知模型";
  const backend = first.backend || "未知后端";

  const totalDatasets = datasets.length || results.length;
  const numOk = typeof data.num_ok === "number" ? data.num_ok : results.filter(r => r.status === "ok").length;
  const numErr = typeof data.num_error === "number" ? data.num_error : results.filter(r => r.status !== "ok").length;

  let totalSamples = 0;
  let accSum = 0;
  let accCount = 0;
  let emSum = 0;
  let emCount = 0;

  for (const r of results) {
    const m = r.metrics || {};
    const nSamples = m.num_samples || m.overall_total || 0;
    totalSamples += nSamples;

    if (r.method === "field-eval") {
      const overall = m.overall || {};
      const acc = typeof overall.micro_accuracy === "number" ? overall.micro_accuracy : (m.overall_accuracy || 0);
      accSum += acc;
      accCount++;
      if (typeof overall.exact_match_rate === "number") {
        emSum += overall.exact_match_rate;
        emCount++;
      }
    } else {
      const score = typeof m.overall_mean_score === "number" ? m.overall_mean_score : 0;
      accSum += score;
      accCount++;
    }
  }

  const avgAcc = accCount ? (accSum / accCount) : 0;
  const avgEm = emCount ? (emSum / emCount) : 0;

  const heroModel = document.getElementById("sr-hero-model-name");
  const heroBackend = document.getElementById("sr-hero-backend-badge");
  const heroDsCount = document.getElementById("sr-hero-datasets-count");
  const heroStatus = document.getElementById("sr-hero-status-pill");
  const heroTotalSamples = document.getElementById("sr-hero-total-samples");
  const heroAvgAcc = document.getElementById("sr-hero-avg-accuracy");
  const heroAvgEm = document.getElementById("sr-hero-avg-em");
  const heroMeta = document.getElementById("sr-hero-meta-desc");

  if (heroModel) heroModel.textContent = modelName;
  if (heroBackend) heroBackend.textContent = backend;
  if (heroDsCount) heroDsCount.textContent = totalDatasets;
  if (heroStatus) {
    heroStatus.textContent = `${numOk} 正常 / ${numErr} 异常`;
    heroStatus.style.color = numErr > 0 ? "var(--rose-500)" : "var(--emerald-500)";
  }
  if (heroTotalSamples) heroTotalSamples.textContent = totalSamples.toLocaleString();
  if (heroAvgAcc) heroAvgAcc.textContent = `${(avgAcc * 100).toFixed(1)}%`;
  if (heroAvgEm) heroAvgEm.textContent = emCount ? `${(avgEm * 100).toFixed(1)}%` : "—";
  if (heroMeta && state.sweepResults.activeRunPath) {
    heroMeta.textContent = `文件路径: ${state.sweepResults.activeRunPath}`;
  }
}

function onSweepFilterChange() {
  const searchInput = document.getElementById("sr-filter-search");
  const methodSelect = document.getElementById("sr-filter-method");
  const statusSelect = document.getElementById("sr-filter-status");
  const sortSelect = document.getElementById("sr-filter-sort");

  state.sweepResults.filterSearch = searchInput ? searchInput.value.trim().toLowerCase() : "";
  state.sweepResults.filterMethod = methodSelect ? methodSelect.value : "all";
  state.sweepResults.filterStatus = statusSelect ? statusSelect.value : "all";
  state.sweepResults.sortBy = sortSelect ? sortSelect.value : "default";

  if (state.sweepResults.data) {
    renderSweepOverviewTable(state.sweepResults.data);
  }
}

function getSweepResultScore(r) {
  const m = r.metrics || {};
  if (r.method === "field-eval") {
    return getSweepResultOverallAccuracy(r) ?? m.overall?.micro_accuracy ?? 0;
  }
  return m.overall_mean_score ?? 0;
}

function getSweepResultOverallAccuracy(result) {
  const metrics = result?.metrics || {};
  const overall = metrics.overall || {};
  const value = [
    metrics.overall_accuracy,
    overall.overall_accuracy,
    overall.accuracy,
  ].find((candidate) => typeof candidate === "number");
  return value === undefined ? null : value;
}

function renderSweepOverviewTable(data) {
  const tbody = document.getElementById("sr-overview-tbody");
  const countBadge = document.getElementById("sr-table-count-badge");
  if (!tbody) return;

  const rawResults = data.results || [];
  let filtered = rawResults.filter(r => {
    if (state.sweepResults.filterMethod !== "all" && r.method !== state.sweepResults.filterMethod) return false;
    if (state.sweepResults.filterStatus !== "all" && r.status !== state.sweepResults.filterStatus) return false;
    if (state.sweepResults.filterSearch) {
      const q = state.sweepResults.filterSearch;
      const ds = (r.dataset || "").toLowerCase();
      const model = (r.model || "").toLowerCase();
      if (!ds.includes(q) && !model.includes(q)) return false;
    }
    return true;
  });

  if (state.sweepResults.sortBy === "score_desc") {
    filtered.sort((a, b) => getSweepResultScore(b) - getSweepResultScore(a));
  } else if (state.sweepResults.sortBy === "score_asc") {
    filtered.sort((a, b) => getSweepResultScore(a) - getSweepResultScore(b));
  } else if (state.sweepResults.sortBy === "samples_desc") {
    filtered.sort((a, b) => (b.metrics?.num_samples || 0) - (a.metrics?.num_samples || 0));
  } else if (state.sweepResults.sortBy === "name_asc") {
    filtered.sort((a, b) => (a.dataset || "").localeCompare(b.dataset || ""));
  }

  if (countBadge) countBadge.textContent = `${filtered.length} / ${rawResults.length} 个数据集`;

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 2rem; color: var(--text-dim);">没有符合当前筛选条件的数据集</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map((r, idx) => {
    const isSelected = r.dataset === state.sweepResults.selectedDatasetName;
    const m = r.metrics || {};
    const samples = m.num_samples ?? m.overall_total ?? "—";
    const isFieldEval = r.method === "field-eval";

    const score = getSweepResultScore(r);
    const scorePct = (score * 100).toFixed(1) + "%";
    const overallAccuracy = getSweepResultOverallAccuracy(r);
    const overallAccuracyText = overallAccuracy === null ? "—" : `${(overallAccuracy * 100).toFixed(1)}%`;
    let barColor = "var(--emerald-500)";
    if (score < 0.5) barColor = "var(--rose-500)";
    else if (score < 0.8) barColor = "var(--amber-500)";

    const emRate = isFieldEval && typeof m.overall?.exact_match_rate === "number"
      ? (m.overall.exact_match_rate * 100).toFixed(1) + "%"
      : "—";

    let fieldsOrTurns = "";
    if (isFieldEval) {
      const fCount = m.fields ? m.fields.length : Object.keys(m.per_field || {}).length;
      fieldsOrTurns = `<span title="${escapeHtml((m.fields || []).join(', '))}">${fCount} 个字段</span>`;
    } else {
      const tCount = m.per_turn ? Object.keys(m.per_turn).length : "—";
      fieldsOrTurns = `<span>${tCount} 轮对话</span>`;
    }

    const statusBadge = r.status === "ok"
      ? `<span class="badge badge-success">OK</span>`
      : `<span class="badge badge-danger">Error</span>`;

    return `
      <tr class="${isSelected ? 'selected' : ''}" onclick="selectSweepDataset('${escapeHtml(r.dataset)}')">
        <td style="text-align: center; color: var(--text-dim); font-size: 0.75rem;">${idx + 1}</td>
        <td>
          <strong style="color: #000; font-size: 0.86rem;">${escapeHtml(r.dataset)}</strong>
        </td>
        <td>
          <span class="role-badge" style="${isFieldEval ? 'background: rgba(99,102,241,0.2); color: #a5b4fc;' : 'background: rgba(6,182,212,0.2); color: var(--cyan-500);'}">
            ${isFieldEval ? 'field-eval' : 'eval'}
          </span>
        </td>
        <td style="text-align: right; font-family: var(--font-mono);">${samples}</td>
        <td>
          <div class="sr-progress-bar-wrap" title="准确率/综合得分: ${scorePct}">
            <div class="sr-progress-bar-fill" style="width: ${score * 100}%; background: ${barColor};"></div>
            <span class="sr-progress-bar-text">${scorePct}</span>
          </div>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.2rem;">总体准确率 overall_accuracy: ${overallAccuracyText}</div>
        </td>
        <td style="text-align: right; font-family: var(--font-mono); font-weight: 600; color: ${emRate !== '—' ? 'var(--text-main)' : 'var(--text-dim)'};">${emRate}</td>
        <td style="font-size: 0.76rem; color: var(--text-muted);">${fieldsOrTurns}</td>
        <td style="text-align: center;">${statusBadge}</td>
        <td style="text-align: center;">
          <button class="btn btn-sm ${isSelected ? 'btn-primary' : ''}" onclick="event.stopPropagation(); selectSweepDataset('${escapeHtml(r.dataset)}')">
            审查
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

function selectSweepDataset(datasetName) {
  state.sweepResults.selectedDatasetName = datasetName;
  state.sweepResults.activeHeatmapField = null;

  renderSweepOverviewTable(state.sweepResults.data);
  renderSweepSelectedDatasetDetail();

  const drillPanel = document.getElementById("sr-drilldown-panel");
  if (drillPanel) {
    drillPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function navigateSweepDetail(direction) {
  const results = state.sweepResults.data?.results || [];
  if (!results.length) return;
  const currentIdx = results.findIndex(r => r.dataset === state.sweepResults.selectedDatasetName);
  if (currentIdx === -1) return;

  let newIdx = currentIdx + direction;
  if (newIdx < 0) newIdx = results.length - 1;
  if (newIdx >= results.length) newIdx = 0;

  selectSweepDataset(results[newIdx].dataset);
}

function openSelectedRunReport() {
  const results = state.sweepResults.data?.results || [];
  const current = results.find(r => r.dataset === state.sweepResults.selectedDatasetName);
  if (!current || !current.report) {
    showToast("该数据集未生成独立报告文件", "info");
    return;
  }
  showToast(`报告落盘路径: ${current.report}`, "info");
}

function switchSweepSubView(subView) {
  state.sweepResults.subView = subView;
  const cmBtn = document.getElementById("sr-subtab-cm-btn");
  const pvBtn = document.getElementById("sr-subtab-pv-btn");
  const cmView = document.getElementById("sr-subview-cm");
  const pvView = document.getElementById("sr-subview-pv");

  if (cmBtn) cmBtn.classList.toggle("active", subView === "cm");
  if (pvBtn) pvBtn.classList.toggle("active", subView === "pv");
  if (cmView) cmView.style.display = subView === "cm" ? "block" : "none";
  if (pvView) pvView.style.display = subView === "pv" ? "block" : "none";
}

function renderSweepSelectedDatasetDetail() {
  const drillPanel = document.getElementById("sr-drilldown-panel");
  const results = state.sweepResults.data?.results || [];
  if (!results.length) {
    if (drillPanel) drillPanel.style.display = "none";
    return;
  }

  const current = results.find(r => r.dataset === state.sweepResults.selectedDatasetName) || results[0];
  state.sweepResults.selectedDatasetName = current.dataset;

  if (drillPanel) drillPanel.style.display = "block";

  const dsIdx = results.findIndex(r => r.dataset === current.dataset);
  const idxCounter = document.getElementById("sr-detail-idx-counter");
  if (idxCounter) idxCounter.textContent = `${dsIdx + 1} / ${results.length}`;

  const titleEl = document.getElementById("sr-detail-ds-name");
  const methodBadge = document.getElementById("sr-detail-method-badge");
  const matchmodeBadge = document.getElementById("sr-detail-matchmode-badge");
  const submetaEl = document.getElementById("sr-detail-submeta");
  const reportBtn = document.getElementById("sr-detail-report-btn");

  if (titleEl) titleEl.textContent = current.dataset;
  if (methodBadge) methodBadge.textContent = current.method;
  if (matchmodeBadge) {
    if (current.metrics?.match_mode) {
      matchmodeBadge.style.display = "inline-block";
      matchmodeBadge.textContent = `${current.metrics.match_mode} 匹配模式`;
    } else {
      matchmodeBadge.style.display = "none";
    }
  }
  if (submetaEl) {
    const scoredAt = current.metrics?.scored_at ? new Date(current.metrics.scored_at).toLocaleString("zh-CN") : "未知时间";
    const samples = current.metrics?.num_samples ?? current.metrics?.overall_total ?? "—";
    submetaEl.textContent = `评测时间: ${scoredAt} · 样本量: ${samples} · 状态: ${current.status || "ok"}`;
  }
  if (reportBtn) {
    reportBtn.style.display = current.report ? "inline-block" : "none";
  }

  const fieldContent = document.getElementById("sr-field-eval-content");
  const evalContent = document.getElementById("sr-eval-content");

  if (current.method === "field-eval") {
    if (fieldContent) fieldContent.style.display = "block";
    if (evalContent) evalContent.style.display = "none";
    renderSweepFieldEvalDetail(current);
  } else {
    if (fieldContent) fieldContent.style.display = "none";
    if (evalContent) evalContent.style.display = "block";
    renderSweepEvalDetail(current);
  }
}

function renderSweepFieldEvalDetail(result) {
  const m = result.metrics || {};

  // 1. 抽取统计条
  const extractBar = document.getElementById("sr-extract-stats-bar");
  if (extractBar) {
    const ref = m.ref_extract || {};
    const pred = m.pred_extract || {};
    extractBar.innerHTML = `
      <span style="font-size: 0.8rem; font-weight: 700; color: var(--cyan-500); margin-right: 0.5rem;">⚡ 抽取缓存状态:</span>
      <div class="sr-extract-pill">
        <span>标准参考 Ref:</span>
        <strong>${ref.total || 0}</strong> 条 (复用 <strong>${ref.skipped_already_done || 0}</strong>, 新增 <strong>${ref.newly_completed || 0}</strong>, 异常 <strong>${ref.errors || 0}</strong>)
      </div>
      <div style="width: 1px; height: 16px; background: var(--border-subtle);"></div>
      <div class="sr-extract-pill">
        <span>模型预测 Pred:</span>
        <strong>${pred.total || 0}</strong> 条 (复用 <strong>${pred.skipped_already_done || 0}</strong>, 新增 <strong>${pred.newly_completed || 0}</strong>, 异常 <strong>${pred.errors || 0}</strong>)
      </div>
    `;
  }

  // 2. 整体指标卡片组
  const overallPills = document.getElementById("sr-field-overall-pills");
  if (overallPills) {
    const ov = m.overall || {};
    const overallAccuracy = getSweepResultOverallAccuracy(result);
    const microAcc = ov.micro_accuracy ?? 0;
    const macroAcc = ov.macro_accuracy ?? microAcc;
    const emRate = ov.exact_match_rate ?? 0;
    const strictEmRate = ov.strict_exact_match_rate ?? 0;

    overallPills.innerHTML = `
      <div class="sr-metric-pill">
        <div class="title">总体准确率 (overall_accuracy)</div>
        <div class="val" style="color: var(--emerald-500);">${overallAccuracy === null ? "—" : `${(overallAccuracy * 100).toFixed(2)}%`}</div>
      </div>
      <div class="sr-metric-pill">
        <div class="title">综合微准确率 (Micro Acc)</div>
        <div class="val" style="color: var(--emerald-500);">${(microAcc * 100).toFixed(2)}%</div>
      </div>
      <div class="sr-metric-pill">
        <div class="title">宏平均准确率 (Macro Acc)</div>
        <div class="val" style="color: #a5b4fc;">${(macroAcc * 100).toFixed(2)}%</div>
      </div>
      <div class="sr-metric-pill">
        <div class="title">完全一致率 (Exact Match)</div>
        <div class="val" style="color: var(--cyan-500);">${(emRate * 100).toFixed(2)}%</div>
        <div style="font-size: 0.7rem; color: var(--text-dim); margin-top: 2px;">${ov.exact_match_samples || 0} 样本完全命中</div>
      </div>
      <div class="sr-metric-pill">
        <div class="title">严格一致率 (Strict EM)</div>
        <div class="val" style="color: var(--amber-500);">${(strictEmRate * 100).toFixed(2)}%</div>
        <div style="font-size: 0.7rem; color: var(--text-dim); margin-top: 2px;">${ov.strict_exact_match_samples || 0} 样本</div>
      </div>
      <div class="sr-metric-pill">
        <div class="title">已评测样本总数</div>
        <div class="val" style="color: #fff;">${m.num_scored || m.num_samples || 0}</div>
        <div style="font-size: 0.7rem; color: var(--text-dim); margin-top: 2px;">缺失样本: ${m.num_pred_missing || 0}</div>
      </div>
    `;
  }

  // 3. 逐字段卡片网格
  const perFieldGrid = document.getElementById("sr-per-field-grid");
  if (perFieldGrid) {
    const perField = m.per_field || {};
    const fieldNames = Object.keys(perField);
    perFieldGrid.innerHTML = fieldNames.map(fName => {
      const f = perField[fName];
      const acc = f.accuracy ?? f.overall_accuracy ?? 0;
      const accPct = (acc * 100).toFixed(2) + "%";
      let barCol = "var(--emerald-500)";
      if (acc < 0.5) barCol = "var(--rose-500)";
      else if (acc < 0.8) barCol = "var(--amber-500)";

      return `
        <div class="sr-field-card">
          <div class="sr-field-card-header">
            <span class="sr-field-card-title">${escapeHtml(fName)}</span>
            <span style="font-family: var(--font-mono); font-size: 1.05rem; font-weight: 700; color: ${barCol};">${accPct}</span>
          </div>
          <div class="sr-progress-bar-wrap" style="height: 10px;">
            <div class="sr-progress-bar-fill" style="width: ${acc * 100}%; background: ${barCol};"></div>
          </div>
          <div class="sr-field-meta-line">
            <span>支持样本: <strong>${f.correct ?? f.overall_correct ?? 0} / ${f.total ?? f.overall_total ?? 0}</strong></span>
            <span>非空准确率: <strong>${((f.non_empty_accuracy ?? 0) * 100).toFixed(1)}%</strong></span>
          </div>
          ${f.empty_count > 0 ? `
          <div class="sr-field-meta-line" style="color: var(--amber-500); font-size: 0.72rem;">
            <span>空值样本: ${f.empty_count} 个</span>
            <span>空值判定命中率: ${((f.empty_accuracy ?? 0) * 100).toFixed(1)}%</span>
          </div>` : ''}
        </div>
      `;
    }).join("");
  }

  // 4. 设置混淆矩阵字段切换 pills
  const cmPillsContainer = document.getElementById("sr-cm-field-pills");
  const cms = m.confusion_matrices || {};
  const cmFields = Object.keys(cms);

  if (!state.sweepResults.activeHeatmapField || !cms[state.sweepResults.activeHeatmapField]) {
    state.sweepResults.activeHeatmapField = cmFields[0] || null;
  }

  if (cmPillsContainer) {
    if (!cmFields.length) {
      cmPillsContainer.innerHTML = `<span style="font-size: 0.8rem; color: var(--text-dim);">该字段抽取无混淆矩阵数据</span>`;
    } else {
      cmPillsContainer.innerHTML = cmFields.map(fn => {
        const isActive = fn === state.sweepResults.activeHeatmapField;
        return `
          <button class="sr-field-pill-btn ${isActive ? 'active' : ''}" onclick="onSelectCmField('${escapeHtml(fn)}')">
            ${escapeHtml(fn)}
          </button>
        `;
      }).join("");
    }
  }

  // 渲染混淆矩阵
  if (state.sweepResults.activeHeatmapField && cms[state.sweepResults.activeHeatmapField]) {
    renderConfusionMatrixHeatmap(cms[state.sweepResults.activeHeatmapField], state.sweepResults.activeHeatmapField);
  }

  // 渲染逐取值统计
  renderPerValueBreakdown(m.per_value || {});
}

function onSelectCmField(fieldName) {
  state.sweepResults.activeHeatmapField = fieldName;
  const cms = state.sweepResults.data?.results?.find(r => r.dataset === state.sweepResults.selectedDatasetName)?.metrics?.confusion_matrices || {};
  if (cms[fieldName]) {
    renderConfusionMatrixHeatmap(cms[fieldName], fieldName);
  }
  const pills = document.querySelectorAll(".sr-field-pill-btn");
  pills.forEach(p => p.classList.toggle("active", p.textContent.trim() === fieldName));
}

function getConfusionMatrixCellAlpha(value, maxValue, maxAlpha) {
  const ratio = maxValue > 0 ? Math.min(1, Math.max(0, value / maxValue)) : 0;
  const minAlpha = 0.08;
  return minAlpha + ratio * (maxAlpha - minAlpha);
}

function renderConfusionMatrixHeatmap(cm, fieldName) {
  const activeFieldTitle = document.getElementById("sr-cm-active-field-name");
  const tableContainer = document.getElementById("sr-cm-table-container");
  const perclassContainer = document.getElementById("sr-cm-perclass-container");
  const macroBadge = document.getElementById("sr-cm-macro-avg-badge");

  if (activeFieldTitle) activeFieldTitle.textContent = fieldName;

  if (macroBadge) {
    const macro = cm.macro_avg || {};
    const f1 = typeof macro.f1 === "number" ? macro.f1.toFixed(3) : "—";
    const prec = typeof macro.precision === "number" ? macro.precision.toFixed(3) : "—";
    const recall = typeof macro.recall === "number" ? macro.recall.toFixed(3) : "—";
    macroBadge.textContent = `宏平均 F1: ${f1} (Prec: ${prec}, Rec: ${recall})`;
  }

  const predClasses = cm.classes || [];
  const refClasses = cm.ref_classes || predClasses;
  const matrix = cm.matrix || [];

  let maxVal = 1;
  matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

  let html = `
    <table class="confusion-matrix-table">
      <thead>
        <tr>
          <th class="cm-corner" title="行: 真实标签 / 列: 模型预测">真实＼预测</th>
          ${predClasses.map(cls => `<th title="预测类别: ${escapeHtml(cls)}">${escapeHtml(cls)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
  `;

  refClasses.forEach((refCls, rowIdx) => {
    html += `<tr><td class="cm-row-label" title="真实类别: ${escapeHtml(refCls)}">${escapeHtml(refCls)}</td>`;
    const row = matrix[rowIdx] || [];

    predClasses.forEach((predCls, colIdx) => {
      const val = row[colIdx] || 0;
      const isDiag = (refCls === predCls);

      let cellClass = "cm-cell";
      let cellStyle = "";

      if (val === 0) {
        cellClass += " cm-zero";
      } else if (isDiag) {
        cellClass += " cm-diag";
        const alpha = getConfusionMatrixCellAlpha(val, maxVal, 0.85);
        cellStyle = `background: rgba(16, 185, 129, ${alpha.toFixed(2)}); border: 1px solid var(--emerald-500);`;
      } else {
        cellClass += " cm-error";
        const alpha = getConfusionMatrixCellAlpha(val, maxVal, 0.8);
        cellStyle = `background: rgba(244, 63, 94, ${alpha.toFixed(2)}); border: 1px solid var(--rose-500);`;
      }

      const tooltip = `真实: ${escapeHtml(refCls)}\n预测: ${escapeHtml(predCls)}\n样本数: ${val}`;
      html += `<td class="${cellClass}" style="${cellStyle}" title="${tooltip}">${val}</td>`;
    });

    html += `</tr>`;
  });

  html += `</tbody></table>`;
  if (tableContainer) tableContainer.innerHTML = html;

  const perClass = cm.per_class || {};
  let pcHtml = `
    <table class="sr-perclass-table">
      <thead>
        <tr>
          <th>类别名称</th>
          <th>精确率 P</th>
          <th>召回率 R</th>
          <th>F1 值</th>
          <th>支持样本</th>
        </tr>
      </thead>
      <tbody>
  `;

  const pcKeys = Object.keys(perClass);
  if (!pcKeys.length) {
    pcHtml += `<tr><td colspan="5" style="text-align: center; color: var(--text-dim);">无逐类指标数据</td></tr>`;
  } else {
    pcKeys.forEach(k => {
      const item = perClass[k];
      const p = typeof item.precision === "number" ? (item.precision * 100).toFixed(1) + "%" : "—";
      const r = typeof item.recall === "number" ? (item.recall * 100).toFixed(1) + "%" : "—";
      const f1 = typeof item.f1 === "number" ? item.f1.toFixed(3) : "—";
      const sup = item.support ?? "—";

      pcHtml += `
        <tr>
          <td title="${escapeHtml(k)}">${escapeHtml(k)}</td>
          <td>${p}</td>
          <td>${r}</td>
          <td style="font-weight: 700; color: #a5b4fc;">${f1}</td>
          <td>${sup}</td>
        </tr>
      `;
    });
  }

  pcHtml += `</tbody></table>`;
  if (perclassContainer) perclassContainer.innerHTML = pcHtml;
}

function renderPerValueBreakdown(perValue) {
  const container = document.getElementById("sr-per-value-content");
  if (!container) return;

  const fields = Object.keys(perValue);
  if (!fields.length) {
    container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--text-dim); padding: 2rem;">无逐取值支持度统计数据</div>`;
    return;
  }

  container.innerHTML = fields.map(fName => {
    const valuesObj = perValue[fName] || {};
    const valNames = Object.keys(valuesObj);

    return `
      <div class="sr-pv-field-card">
        <div class="sr-pv-field-title">
          <span>🏷️ ${escapeHtml(fName)}</span>
          <span style="font-size: 0.75rem; color: var(--text-dim); font-weight: normal;">${valNames.length} 个可能取值</span>
        </div>
        <div class="sr-pv-list">
          ${valNames.map(vn => {
            const item = valuesObj[vn];
            const acc = typeof item.accuracy === "number" ? item.accuracy : 0;
            const accPct = (acc * 100).toFixed(1) + "%";
            let bColor = "var(--emerald-500)";
            if (acc < 0.5) bColor = "var(--rose-500)";
            else if (acc < 0.8) bColor = "var(--amber-500)";

            return `
              <div class="sr-pv-item">
                <div class="sr-pv-item-head">
                  <span class="sr-pv-val-name" title="${escapeHtml(vn)}">${escapeHtml(vn)}</span>
                  <span class="sr-pv-val-stats">
                    ${item.correct || 0} / ${item.support || 0} (${accPct})
                  </span>
                </div>
                <div class="sr-pv-bar">
                  <div class="sr-pv-bar-fill" style="width: ${acc * 100}%; background: ${bColor};"></div>
                </div>
              </div>
            `;
          }).join("")}
        </div>
      </div>
    `;
  }).join("");
}

function renderSweepEvalDetail(result) {
  const m = result.metrics || {};

  const overallPills = document.getElementById("sr-eval-overall-pills");
  if (overallPills) {
    const meanScore = typeof m.overall_mean_score === "number" ? m.overall_mean_score : 0;
    overallPills.innerHTML = `
      <div class="sr-metric-pill">
        <div class="title">综合平均得分 (Overall Mean)</div>
        <div class="val" style="color: var(--emerald-500);">${(meanScore * 100).toFixed(2)}%</div>
      </div>
      <div class="sr-metric-pill">
        <div class="title">已评测样本数 (Samples)</div>
        <div class="val" style="color: var(--cyan-500);">${m.num_samples || 0}</div>
      </div>
      <div class="sr-metric-pill">
        <div class="title">评测目标总轮次 (Targets)</div>
        <div class="val" style="color: #a5b4fc;">${m.num_targets || 0}</div>
      </div>
      <div class="sr-metric-pill">
        <div class="title">未达标样本数 (Failed Samples)</div>
        <div class="val" style="color: ${m.num_failed_samples > 0 ? 'var(--rose-500)' : 'var(--emerald-500)'};">${m.num_failed_samples || 0}</div>
      </div>
    `;
  }

  const turnsContainer = document.getElementById("sr-eval-turns-container");
  if (turnsContainer) {
    const perTurn = m.per_turn || {};
    const turnKeys = Object.keys(perTurn);

    if (!turnKeys.length) {
      turnsContainer.innerHTML = `<div style="color: var(--text-dim); text-align: center; padding: 1.5rem;">未提供轮次拆解得分</div>`;
    } else {
      turnsContainer.innerHTML = turnKeys.map((tk, idx) => {
        const t = perTurn[tk];
        const scorer = t.scorer || "scorer";

        let metricDetails = "";
        if (typeof t.accuracy === "number") {
          metricDetails += `<div class="sr-metric-pill"><div class="title">准确率 Accuracy</div><div class="val" style="color: var(--emerald-500);">${(t.accuracy * 100).toFixed(2)}%</div></div>`;
        }
        if (typeof t.f1 === "number") {
          metricDetails += `<div class="sr-metric-pill"><div class="title">Token F1</div><div class="val" style="color: var(--cyan-500);">${(t.f1 * 100).toFixed(2)}%</div></div>`;
        }
        if (typeof t.precision === "number") {
          metricDetails += `<div class="sr-metric-pill"><div class="title">精确率 Precision</div><div class="val">${(t.precision * 100).toFixed(2)}%</div></div>`;
        }
        if (typeof t.recall === "number") {
          metricDetails += `<div class="sr-metric-pill"><div class="title">召回率 Recall</div><div class="val">${(t.recall * 100).toFixed(2)}%</div></div>`;
        }

        let cmHtml = "";
        if (t.confusion_matrix) {
          const cm = t.confusion_matrix;
          const predClasses = cm.classes || [];
          const refClasses = cm.ref_classes || predClasses;
          const matrix = cm.matrix || [];
          let maxVal = 1;
          matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

          cmHtml = `
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-subtle);">
              <div style="font-weight: 600; font-size: 0.84rem; color: #fff; margin-bottom: 0.5rem;">🔥 第 ${idx + 1} 轮分类混淆矩阵</div>
              <div style="overflow-x: auto; max-height: 350px;">
                <table class="confusion-matrix-table">
                  <thead>
                    <tr>
                      <th class="cm-corner">真实＼预测</th>
                      ${predClasses.map(c => `<th>${escapeHtml(c)}</th>`).join("")}
                    </tr>
                  </thead>
                  <tbody>
                    ${refClasses.map((rc, rIdx) => `
                      <tr>
                        <td class="cm-row-label">${escapeHtml(rc)}</td>
                        ${predClasses.map((pc, cIdx) => {
                          const val = (matrix[rIdx] || [])[cIdx] || 0;
                          const isDiag = (rc === pc);
                          let cStyle = "";
                          if (val === 0) cStyle = "background: rgba(255,255,255,0.02); color: var(--text-dim);";
                          else if (isDiag) {
                            const a = getConfusionMatrixCellAlpha(val, maxVal, 0.85);
                            cStyle = `background: rgba(16, 185, 129, ${a.toFixed(2)}); border: 1px solid var(--emerald-500); color: #fff;`;
                          } else {
                            const a = getConfusionMatrixCellAlpha(val, maxVal, 0.8);
                            cStyle = `background: rgba(244, 63, 94, ${a.toFixed(2)}); border: 1px solid var(--rose-500); color: #fff;`;
                          }
                          return `<td class="cm-cell" style="${cStyle}">${val}</td>`;
                        }).join("")}
                      </tr>
                    `).join("")}
                  </tbody>
                </table>
              </div>
            </div>
          `;
        }

        return `
          <div class="dataset-card" style="padding: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
              <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="role-badge" style="background: rgba(99,102,241,0.2); color: #a5b4fc; font-weight: 700;">
                  轮次: ${escapeHtml(tk)}
                </span>
                <span class="config-key-badge">${escapeHtml(scorer)}</span>
              </div>
              <span style="font-size: 0.75rem; color: var(--text-dim);">样本量: ${t.num_scored || t.num_total || 0}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem;">
              ${metricDetails}
            </div>
            ${cmHtml}
          </div>
        `;
      }).join("");
    }
  }

  const failBox = document.getElementById("sr-eval-failures-box");
  if (failBox) {
    if (m.failures_html_path || m.failures_path) {
      failBox.style.display = "block";
      failBox.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
          <div>
            <strong style="color: var(--rose-500);">⚠️ 失败与坏例诊断报告</strong>
            <div style="font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem;">${escapeHtml(m.failures_path || m.failures_html_path)}</div>
          </div>
          <button class="btn btn-sm" onclick="showToast('失败分析报告路径: ${escapeHtml(m.failures_path || '')}', 'info')">
            查看详情
          </button>
        </div>
      `;
    } else {
      failBox.style.display = "none";
    }
  }
}

function openSweepRawJsonModal() {
  const modal = document.getElementById("sr-raw-json-modal");
  const pre = document.getElementById("sr-raw-json-pre");
  if (!state.sweepResults.data) {
    showToast("当前未加载任何 Sweep 评测数据", "warning");
    return;
  }
  if (pre) {
    pre.textContent = JSON.stringify(state.sweepResults.data, null, 2);
  }
  if (modal) modal.showModal();
}

function closeSweepRawJsonModal() {
  const modal = document.getElementById("sr-raw-json-modal");
  if (modal) modal.close();
}

function copySweepRawJson() {
  if (!state.sweepResults.data) return;
  const text = JSON.stringify(state.sweepResults.data, null, 2);
  navigator.clipboard.writeText(text).then(() => {
    showToast("已复制原始 JSON 到剪贴板", "success");
  }).catch(() => {
    showToast("复制失败，请手动全选复制", "error");
  });
}

function exportSweepMarkdown() {
  const data = state.sweepResults.data;
  if (!data) {
    showToast("无可用评测数据", "warning");
    return;
  }
  const results = data.results || [];
  let md = `# Sweep 批量评测结果汇总报告\n\n`;
  md += `- **评估模型**: ${results[0]?.model || "默认模型"}\n`;
  md += `- **推理后端**: ${results[0]?.backend || "默认后端"}\n`;
  md += `- **覆盖数据集数**: ${results.length}\n`;
  md += `- **生成时间**: ${new Date().toLocaleString("zh-CN")}\n\n`;
  md += `| 数据集 | 评测模式 | 样本量 | 准确率 / 综合得分 | 严格精确匹配 (EM) | 状态 |\n`;
  md += `| :--- | :--- | :---: | :---: | :---: | :---: |\n`;

  results.forEach(r => {
    const isField = r.method === "field-eval";
    const m = r.metrics || {};
    const samples = m.num_samples ?? m.overall_total ?? "—";
    const score = (getSweepResultScore(r) * 100).toFixed(2) + "%";
    const em = isField && typeof m.overall?.exact_match_rate === "number" ? (m.overall.exact_match_rate * 100).toFixed(2) + "%" : "—";
    md += `| ${r.dataset} | ${r.method} | ${samples} | ${score} | ${em} | ${r.status} |\n`;
  });

  const blob = new Blob([md], { type: "text/markdown;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sweep_summary_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("已导出 Markdown 汇总报告", "success");
}

function exportSweepCsv() {
  const data = state.sweepResults.data;
  if (!data) {
    showToast("无可用评测数据", "warning");
    return;
  }
  const results = data.results || [];
  let csv = `\uFEFFDataset,Method,Model,Backend,NumSamples,OverallScore,ExactMatchRate,Status\n`;
  results.forEach(r => {
    const isField = r.method === "field-eval";
    const m = r.metrics || {};
    const samples = m.num_samples ?? m.overall_total ?? 0;
    const score = getSweepResultScore(r).toFixed(4);
    const em = isField && typeof m.overall?.exact_match_rate === "number" ? m.overall.exact_match_rate.toFixed(4) : "";
    csv += `"${r.dataset}","${r.method}","${r.model || ''}","${r.backend || ''}",${samples},${score},${em},"${r.status || 'ok'}"\n`;
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sweep_metrics_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("已导出 CSV 指标表", "success");
}

// 导出到全局 window 对象
window.loadSweepResultsData = loadSweepResultsData;
window.loadSweepRunsList = loadSweepRunsList;
window.onSweepRunSelect = onSweepRunSelect;
window.openSweepRawJsonModal = openSweepRawJsonModal;
window.closeSweepRawJsonModal = closeSweepRawJsonModal;
window.copySweepRawJson = copySweepRawJson;
window.exportSweepMarkdown = exportSweepMarkdown;
window.exportSweepCsv = exportSweepCsv;
window.onSweepFilterChange = onSweepFilterChange;
window.selectSweepDataset = selectSweepDataset;
window.navigateSweepDetail = navigateSweepDetail;
window.openSelectedRunReport = openSelectedRunReport;
window.switchSweepSubView = switchSweepSubView;
window.onSelectCmField = onSelectCmField;
