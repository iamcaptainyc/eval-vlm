/**
 * eval_vlm Web UI - Modern Vanilla JS Reactive Engine
 * Zero CDN dependencies, 100% offline & local network compatible.
 * Full support for both standard eval and field-eval, with transparent Job Queue.
 */

// 全局应用状态
const state = {
  activeTab: "datasets",
  user: { username: "loading...", role: "viewer" },
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

  // 配置
  configData: { config: {}, raw: "", settableDoc: "" },

  // 任务队列
  jobs: [],
  currentJobId: null,
  terminalLogs: "",
  logDrawerOpen: false,
  autoScrollLogs: true,
  eventSource: null,

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

// --------------------------------------------------------------------------
// 路由与 Tab 切换
// --------------------------------------------------------------------------
function switchTab(tab, dataset = null, updateHash = true) {
  state.activeTab = tab;
  if (dataset) state.currentDataset = dataset;

  if (updateHash) {
    if (state.currentDataset) {
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

  updateHeaderDatasetPill();

  // 触发对应 Tab 数据获取
  if (tab === "datasets") loadDatasets();
  if (tab === "gallery") loadSamples(0);
  if (tab === "config") loadConfig();
  if (tab === "jobs") loadJobs();
  if (tab === "runs") loadRuns();
  if (tab === "trash") loadTrash();
  if (tab === "health") loadHealth();
}

function handleHash() {
  const hash = window.location.hash || "#/datasets";
  const parts = hash.replace("#/", "").split("/");
  const tab = parts[0] || "datasets";
  const ds = parts[1] ? decodeURIComponent(parts[1]) : state.currentDataset;
  if (["datasets", "gallery", "config", "jobs", "runs", "trash", "health"].includes(tab)) {
    switchTab(tab, ds, false);
  }
}

function updateHeaderDatasetPill() {
  const pill = document.getElementById("header-dataset-pill");
  const nameEl = document.getElementById("header-dataset-name");
  const shaEl = document.getElementById("header-dataset-sha");
  if (!pill || !nameEl || !shaEl) return;

  if (state.currentDataset) {
    pill.style.display = "flex";
    nameEl.textContent = state.currentDataset;
    shaEl.textContent = state.testSha ? `#${state.testSha.slice(0, 8)}` : "";
  } else {
    pill.style.display = "none";
  }
}

// --------------------------------------------------------------------------
// 数据集 (Datasets)
// --------------------------------------------------------------------------
async function loadDatasets() {
  try {
    const res = await fetch("/api/datasets");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.datasets = await res.json();
    if (!state.currentDataset && state.datasets.length > 0) {
      state.currentDataset = state.datasets[0].name;
      updateHeaderDatasetPill();
    }
    renderDatasets();
  } catch (err) {
    showToast("获取数据集列表失败", "error");
  }
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
  updateHeaderDatasetPill();
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
    updateHeaderDatasetPill();
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
              <div class="image-tile">
                <div class="missing-image-box">
                  <span style="font-size: 1.1rem;">⚠️</span>
                  <span style="font-size: 0.72rem; font-weight: 700; margin-top: 2px;">图片丢失</span>
                  <span style="font-size: 0.65rem; word-break: break-all; opacity: 0.8;">${escapeHtml(img.ref)}</span>
                </div>
              </div>`;
            }
            return `
            <div class="image-tile">
              <img src="${escapeHtml(img.url)}&thumb=1" loading="lazy" alt="${escapeHtml(img.ref)}" onclick="openLightbox('${escapeHtml(img.url)}', '${escapeHtml(img.ref)}')">
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
          <div>
            <span class="sample-index-badge">#${s.position}</span>
            <span class="sample-id-code">${escapeHtml(s.id)}</span>
          </div>
          <button class="btn btn-sm btn-outline-danger" onclick="promptDelete('${escapeHtml(s.id)}', 'record')">
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
// 配置编辑 (Config)
// --------------------------------------------------------------------------
async function loadConfig() {
  if (!state.currentDataset) {
    const rawPre = document.getElementById("config-raw-pre");
    if (rawPre) rawPre.textContent = "# 未选定数据集，请先进入数据集页面选择";
    return;
  }
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/config`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.configData = await res.json();
    renderConfig();
  } catch (err) {
    showToast("读取配置文件失败", "error");
  }
}

function renderConfig() {
  const cfg = state.configData.config || {};
  const rawPre = document.getElementById("config-raw-pre");
  if (rawPre) rawPre.textContent = state.configData.raw || "";

  const backendSelect = document.getElementById("cfg-backend");
  const modelInput = document.getElementById("cfg-model");
  const urlInput = document.getElementById("cfg-base-url");
  const scorerSelect = document.getElementById("cfg-scorer");
  const targetsSelect = document.getElementById("cfg-targets");

  if (backendSelect && cfg.inference?.backend) backendSelect.value = cfg.inference.backend;
  if (modelInput && cfg.inference?.openai?.model) modelInput.value = cfg.inference.openai.model;
  if (urlInput && cfg.inference?.openai?.base_url) urlInput.value = cfg.inference.openai.base_url;
  if (scorerSelect && cfg.scoring?.scorer) scorerSelect.value = cfg.scoring.scorer;
  if (targetsSelect && cfg.eval?.targets) targetsSelect.value = cfg.eval.targets;
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

// --------------------------------------------------------------------------
// 任务启动模态框配置 (消除黑盒)
// --------------------------------------------------------------------------
function openJobModal(type) {
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
        .map((d) => `<option value="${escapeHtml(d.name)}" ${d.name === state.currentDataset ? "selected" : ""}>${escapeHtml(d.name)}</option>`)
        .join("");
    }
  }

  // 专用参数面板显示隐藏
  if (feParams) feParams.style.display = type === "field-eval" ? "block" : "none";
  if (evalParams) evalParams.style.display = (type === "eval" || type === "score") ? "block" : "none";

  // 重置通用输入项默认值
  const backendSelect = document.getElementById("job-modal-backend");
  const modelInput = document.getElementById("job-modal-model");
  const limitInput = document.getElementById("job-modal-limit");
  const failfastCb = document.getElementById("job-modal-failfast");
  if (backendSelect) backendSelect.value = "";
  if (modelInput) modelInput.value = "";
  if (limitInput) limitInput.value = "";
  if (failfastCb) failfastCb.checked = false;

  updateJobCommandPreview();
  if (modal) modal.showModal();
}

function closeJobModal() {
  const modal = document.getElementById("job-launch-modal");
  if (modal) modal.close();
}

function updateJobCommandPreview() {
  const type = state.jobModal.type;
  const dsSelect = document.getElementById("job-modal-dataset");
  const dsName = dsSelect ? dsSelect.value : state.currentDataset;
  const pathSpan = document.getElementById("job-modal-dataset-path");

  // 更新数据集物理路径提示
  if (pathSpan) {
    const dsObj = state.datasets.find((d) => d.name === dsName);
    pathSpan.textContent = dsObj ? `路径: ${dsObj.path}` : "";
  }

  const backend = document.getElementById("job-modal-backend")?.value;
  const model = document.getElementById("job-modal-model")?.value.trim();
  const limit = document.getElementById("job-modal-limit")?.value.trim();
  const failfast = document.getElementById("job-modal-failfast")?.checked;

  const parts = ["python", "-m", "eval_vlm", type];
  if (dsName) parts.push("-d", dsName);
  if (backend) parts.push("--backend", backend);
  if (model) parts.push("--model", model);
  if (limit) parts.push("--limit", limit);
  if (failfast) parts.push("--fail-fast");

  if (type === "field-eval") {
    const matchMode = document.getElementById("job-modal-matchmode")?.value;
    const targets = document.getElementById("job-modal-targets")?.value;
    const overwrite = document.getElementById("job-modal-overwrite")?.checked;
    if (matchMode && matchMode !== "exact") parts.push("--match-mode", matchMode);
    if (targets && targets !== "first") parts.push("--targets", targets);
    if (overwrite) parts.push("--overwrite");
  } else if (type === "eval" || type === "score") {
    const scorer = document.getElementById("job-modal-scorer")?.value;
    const targets = document.getElementById("job-modal-eval-targets")?.value;
    if (scorer) parts.push("--scorer", scorer);
    if (targets) parts.push("--targets", targets);
  }

  const cmdStr = parts.join(" ");
  const previewEl = document.getElementById("job-modal-cmd-preview");
  if (previewEl) previewEl.textContent = cmdStr;

  const logDest = document.getElementById("job-modal-log-destination");
  if (logDest) {
    logDest.textContent = `.../_webui/jobs/<job_id>/log.txt (工作区状态目录)`;
  }
}

async function confirmLaunchJob() {
  const type = state.jobModal.type;
  const dsSelect = document.getElementById("job-modal-dataset");
  const dataset = dsSelect ? dsSelect.value : state.currentDataset;

  const backend = document.getElementById("job-modal-backend")?.value;
  const model = document.getElementById("job-modal-model")?.value.trim();
  const limit = document.getElementById("job-modal-limit")?.value.trim();
  const failfast = document.getElementById("job-modal-failfast")?.checked;

  const params = {};
  if (backend) params.backend = backend;
  if (model) params.model = model;
  if (limit) params.limit = parseInt(limit, 10);
  if (failfast) params.fail_fast = true;

  if (type === "field-eval") {
    const matchMode = document.getElementById("job-modal-matchmode")?.value;
    const targets = document.getElementById("job-modal-targets")?.value;
    const overwrite = document.getElementById("job-modal-overwrite")?.checked;
    if (matchMode) params.match_mode = matchMode;
    if (targets) params.targets = targets;
    if (overwrite) params.overwrite = true;
  } else if (type === "eval" || type === "score") {
    const scorer = document.getElementById("job-modal-scorer")?.value;
    const targets = document.getElementById("job-modal-eval-targets")?.value;
    if (scorer) params.scorer = scorer;
    if (targets) params.targets = targets;
  }

  try {
    const url = dataset ? `/api/datasets/${encodeURIComponent(dataset)}/jobs` : "/api/jobs";
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, dataset, params }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const job = await res.json();
    closeJobModal();
    showToast(`任务已提交入队: ${job.id}`, "success");
    await loadJobs();
    openTerminal(job.id);
  } catch (err) {
    showToast(`启动任务失败: ${err.message}`, "error");
  }
}

// --------------------------------------------------------------------------
// 任务管理与 SSE (Jobs)
// --------------------------------------------------------------------------
async function loadJobs() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.jobs = await res.json();
    renderJobs();
  } catch (err) {
    console.error(err);
  }
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
        <td style="font-family: var(--font-mono); font-weight: 600; font-size: 0.8rem;">${escapeHtml(j.id)}</td>
        <td><span class="role-badge" style="background:rgba(6,182,212,0.15); color:var(--cyan-500);">${escapeHtml(j.type)}</span></td>
        <td><strong style="color: #fff;">${escapeHtml(j.dataset || "—")}</strong></td>
        <td style="max-width: 320px; font-size: 0.76rem; font-family: var(--font-mono); color: var(--text-dim); word-break: break-all;" title="${escapeHtml(cmdText)}">
          ${escapeHtml(cmdText)}
        </td>
        <td>${escapeHtml(j.user)}</td>
        <td>${statusBadge}</td>
        <td style="font-size: 0.78rem; color: var(--text-dim);">${new Date(j.created_at).toLocaleString()}</td>
        <td>
          <div style="display: flex; gap: 0.35rem;">
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
          </div>
        </td>
      </tr>
    `;
    })
    .join("");
}

function openTerminal(jobId) {
  state.currentJobId = jobId;
  state.terminalLogs = "";
  state.logDrawerOpen = true;

  const drawer = document.getElementById("terminal-drawer");
  const jobLabel = document.getElementById("terminal-job-label");
  const dsLabel = document.getElementById("terminal-dataset-label");
  const cmdDisplay = document.getElementById("terminal-cmd-display");
  const logDisplay = document.getElementById("terminal-log-display");
  const pre = document.getElementById("terminal-pre");

  if (drawer) drawer.classList.remove("hidden");
  if (pre) pre.textContent = "正在连接进程日志输出流...\\n";

  // 读取已缓存的 job 元信息
  const job = state.jobs.find((j) => j.id === jobId);
  if (job) {
    if (dsLabel) dsLabel.textContent = `数据集: ${job.dataset || "全量"}`;
    if (cmdDisplay) cmdDisplay.textContent = job.command?.length ? job.command.join(" ") : "—";
    if (logDisplay) logDisplay.textContent = job.log_file || "—";
  }

  if (state.eventSource) {
    state.eventSource.close();
  }

  state.eventSource = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream`);

  state.eventSource.addEventListener("started", (e) => {
    try {
      const data = JSON.parse(e.data);
      if (cmdDisplay && data.command) cmdDisplay.textContent = data.command.join(" ");
      if (logDisplay && data.log_file) logDisplay.textContent = data.log_file;
    } catch (_) {}
  });

  state.eventSource.addEventListener("log", (e) => {
    try {
      const line = JSON.parse(e.data);
      state.terminalLogs += line;
      if (pre) pre.textContent = state.terminalLogs;
      if (state.autoScrollLogs && pre) {
        pre.scrollTop = pre.scrollHeight;
      }
    } catch (_) {}
  });

  state.eventSource.addEventListener("status", (e) => {
    try {
      const statusData = JSON.parse(e.data);
      state.terminalLogs += `\n[系统状态更新: ${statusData.status}]\n`;
      if (pre) pre.textContent = state.terminalLogs;
      loadJobs();
    } catch (_) {}
  });

  state.eventSource.onerror = () => {
    state.eventSource.close();
  };
}

function closeTerminal() {
  state.logDrawerOpen = false;
  const drawer = document.getElementById("terminal-drawer");
  if (drawer) drawer.classList.add("hidden");
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

async function cancelJob(jobId) {
  if (!confirm(`确认终止任务 ${jobId} 吗？`)) return;
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
    const res = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/runs`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
    showToast("获取评测结果失败", "error");
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
      let barColor = "var(--emerald-500)";
      if (acc < 0.7) barColor = "var(--rose-500)";
      else if (acc < 0.9) barColor = "var(--amber-500)";

      return `
      <div class="field-metric-card">
        <div class="field-metric-header">
          <span class="field-name-title">${escapeHtml(f)}</span>
          <span class="field-acc-pct" style="color: ${barColor};">${accPct}%</span>
        </div>
        <div class="field-progress-track">
          <div class="field-progress-bar" style="width: ${accPct}%; background: ${barColor};"></div>
        </div>
        <div class="field-metric-footer">
          <span>正确: <strong>${correct}</strong> / ${total}</span>
          <span>失配: <strong>${total - correct}</strong></span>
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
// 页面初始化
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  // 检查鉴权
  try {
    const authRes = await fetch("/api/whoami");
    if (authRes.ok) {
      state.user = await authRes.json();
      const uEl = document.getElementById("header-username");
      const rEl = document.getElementById("header-role");
      if (uEl) uEl.textContent = state.user.username;
      if (rEl) rEl.textContent = state.user.role;
    }
  } catch (_) {}

  // 绑定路由与监听
  window.addEventListener("hashchange", handleHash);
  await loadDatasets();
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
