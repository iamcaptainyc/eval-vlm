/**
 * eval_vlm Web UI 前端交互驱动 (Alpine.js)
 */

document.addEventListener("alpine:init", () => {
  Alpine.data("webuiApp", () => ({
    // 全局状态
    activeTab: "datasets",
    user: { username: "loading...", role: "viewer" },
    datasets: [],
    currentDataset: "",
    datasetInfo: null,

    // 画廊状态
    samples: [],
    samplesTotal: 0,
    samplesOffset: 0,
    samplesLimit: 24,
    samplesFilter: "",
    testSha: "",
    selectedSampleIds: [],

    // 大图灯箱状态
    lightboxImg: null,

    // 删除弹窗状态
    deleteModal: {
      open: false,
      sampleId: "",
      mode: "record", // "record" | "image"
      imageIndex: null,
      reason: "",
    },

    // 配置编辑状态
    configData: {
      raw: "",
      config: {},
      settableDoc: "",
    },

    // 任务状态
    jobs: [],
    currentJobId: null,
    logDrawerOpen: false,
    terminalLogs: "",
    autoScrollLogs: true,
    eventSource: null,

    // 评测结果状态
    runs: [],
    selectedRun: null,
    runMetrics: null,
    scoredRecords: [],
    scoredTotal: 0,
    scoredOffset: 0,
    scoredLimit: 30,
    scoredFilterOrder: "default",

    // 回收站状态
    trashItems: [],

    // 健康诊断状态
    healthReport: null,

    // 初始化
    async init() {
      await this.checkAuth();
      await this.loadDatasets();
      this.handleHashChange();
      window.addEventListener("hashchange", () => this.handleHashChange());
    },

    async checkAuth() {
      try {
        const res = await fetch("/api/whoami");
        if (res.ok) {
          this.user = await res.json();
        }
      } catch (e) {
        console.warn("Auth check error:", e);
      }
    },

    handleHashChange() {
      const hash = window.location.hash || "#/datasets";
      const parts = hash.replace("#/", "").split("/");
      const section = parts[0] || "datasets";
      const dsName = parts[1];

      if (["datasets", "gallery", "config", "jobs", "runs", "trash", "health"].includes(section)) {
        this.activeTab = section;
      }
      if (dsName && dsName !== this.currentDataset) {
        this.selectDataset(decodeURIComponent(dsName), false);
      }
    },

    navigate(tab, dsName = null) {
      this.activeTab = tab;
      const targetDs = dsName || this.currentDataset;
      if (targetDs) {
        window.location.hash = `#/${tab}/${encodeURIComponent(targetDs)}`;
      } else {
        window.location.hash = `#/${tab}`;
      }
    },

    // -------------------------------------------------------------
    // 数据集逻辑
    // -------------------------------------------------------------
    async loadDatasets() {
      try {
        const res = await fetch("/api/datasets");
        if (res.ok) {
          this.datasets = await res.json();
          if (!this.currentDataset && this.datasets.length > 0) {
            this.selectDataset(this.datasets[0].name, false);
          }
        }
      } catch (e) {
        this.toast("加载数据集列表失败", "error");
      }
    },

    async selectDataset(name, updateUrl = true) {
      this.currentDataset = name;
      if (updateUrl) {
        window.location.hash = `#/${this.activeTab}/${encodeURIComponent(name)}`;
      }
      await this.loadDatasetDetails();
      if (this.activeTab === "gallery") {
        await this.loadSamples(0);
      } else if (this.activeTab === "config") {
        await this.loadConfig();
      } else if (this.activeTab === "runs") {
        await this.loadRuns();
      } else if (this.activeTab === "trash") {
        await this.loadTrash();
      } else if (this.activeTab === "health") {
        await this.loadHealth();
      }
    },

    async loadDatasetDetails() {
      if (!this.currentDataset) return;
      try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(this.currentDataset)}`);
        if (res.ok) {
          this.datasetInfo = await res.json();
          this.testSha = this.datasetInfo.test_sha256;
        }
      } catch (e) {
        console.error(e);
      }
    },

    // -------------------------------------------------------------
    // 样本画廊逻辑
    // -------------------------------------------------------------
    async loadSamples(offset = 0) {
      if (!this.currentDataset) return;
      this.samplesOffset = offset;
      const q = new URLSearchParams({
        offset: this.samplesOffset,
        limit: this.samplesLimit,
      });
      if (this.samplesFilter) {
        q.append("filter", this.samplesFilter);
      }

      try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(this.currentDataset)}/samples?${q.toString()}`);
        if (res.ok) {
          const data = await res.json();
          this.samples = data.samples;
          this.samplesTotal = data.total;
          this.testSha = data.test_sha256;
          this.selectedSampleIds = [];
        } else {
          this.toast("加载样本失败", "error");
        }
      } catch (e) {
        this.toast("网络错误", "error");
      }
    },

    openLightbox(img) {
      this.lightboxImg = img;
      const modal = document.getElementById("lightbox-dialog");
      if (modal) modal.showModal();
    },

    closeLightbox() {
      this.lightboxImg = null;
      const modal = document.getElementById("lightbox-dialog");
      if (modal) modal.close();
    },

    // -------------------------------------------------------------
    // 删除与回收站 (核心)
    // -------------------------------------------------------------
    promptDelete(sampleId, mode = "record", imageIndex = null) {
      this.deleteModal = {
        open: true,
        sampleId: sampleId,
        mode: mode,
        imageIndex: imageIndex,
        reason: "",
      };
      const dialog = document.getElementById("delete-dialog");
      if (dialog) dialog.showModal();
    },

    closeDeleteModal() {
      this.deleteModal.open = false;
      const dialog = document.getElementById("delete-dialog");
      if (dialog) dialog.close();
    },

    async executeDelete() {
      const { sampleId, mode, imageIndex, reason } = this.deleteModal;
      try {
        const res = await fetch(
          `/api/datasets/${encodeURIComponent(this.currentDataset)}/samples/${encodeURIComponent(sampleId)}`,
          {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              expected_sha256: this.testSha,
              reason: reason,
              mode: mode,
              image_index: imageIndex,
            }),
          }
        );

        if (res.status === 409) {
          this.toast("test.json 已被修改，正在刷新数据...", "warning");
          this.closeDeleteModal();
          await this.loadSamples(this.samplesOffset);
          await this.loadDatasetDetails();
          return;
        }

        if (res.ok) {
          const result = await res.json();
          this.closeDeleteModal();
          this.toast(`样本 ${sampleId} 已删除 (${result.mode})，已进入回收站`, "success");
          await this.loadSamples(this.samplesOffset);
          await this.loadDatasetDetails();
        } else {
          const err = await res.json();
          this.toast(`删除失败: ${err.detail || "未知错误"}`, "error");
        }
      } catch (e) {
        this.toast(`删除请求异常: ${e}`, "error");
      }
    },

    async loadTrash() {
      if (!this.currentDataset) return;
      try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(this.currentDataset)}/trash`);
        if (res.ok) {
          this.trashItems = await res.json();
        }
      } catch (e) {
        this.toast("加载回收站失败", "error");
      }
    },

    async restoreTrash(trashId) {
      if (!confirm("确认恢复该样本吗？恢复将插回原位置并同步索引。")) return;
      try {
        const res = await fetch(
          `/api/datasets/${encodeURIComponent(this.currentDataset)}/trash/${encodeURIComponent(trashId)}/restore`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ expected_sha256: this.testSha }),
          }
        );

        if (res.status === 409) {
          this.toast("test.json 在此期间已被修改，请刷新重试", "warning");
          await this.loadDatasetDetails();
          return;
        }

        if (res.ok) {
          this.toast("样本恢复成功！", "success");
          await this.loadTrash();
          await this.loadDatasetDetails();
        } else {
          const err = await res.json();
          this.toast(`恢复失败: ${err.detail || "未知错误"}`, "error");
        }
      } catch (e) {
        this.toast("网络异常", "error");
      }
    },

    // -------------------------------------------------------------
    // 配置编辑
    // -------------------------------------------------------------
    async loadConfig() {
      if (!this.currentDataset) return;
      try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(this.currentDataset)}/config`);
        if (res.ok) {
          this.configData = await res.json();
        }
      } catch (e) {
        this.toast("加载配置失败", "error");
      }
    },

    async saveConfigKey(key, value) {
      try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(this.currentDataset)}/config`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ updates: [{ key, value }] }),
        });
        if (res.ok) {
          this.toast(`配置项 ${key} 已更新`, "success");
          await this.loadConfig();
        } else {
          this.toast("更新配置失败", "error");
        }
      } catch (e) {
        this.toast("网络错误", "error");
      }
    },

    // -------------------------------------------------------------
    // 任务管理与 SSE
    // -------------------------------------------------------------
    async loadJobs() {
      try {
        const res = await fetch("/api/jobs");
        if (res.ok) {
          this.jobs = await res.json();
        }
      } catch (e) {
        console.error(e);
      }
    },

    async submitDatasetJob(type, params = {}) {
      if (!this.currentDataset && type !== "sweep") return;
      try {
        const url = type === "sweep" ? "/api/sweep/jobs" : `/api/datasets/${encodeURIComponent(this.currentDataset)}/jobs`;
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ type: type, params: params }),
        });
        if (res.ok) {
          const job = await res.json();
          this.toast(`任务已入队: ${job.id}`, "success");
          await this.loadJobs();
          this.openJobTerminal(job.id);
        } else {
          const err = await res.json();
          this.toast(`提交任务失败: ${err.detail || "错误"}`, "error");
        }
      } catch (e) {
        this.toast("提交任务异常", "error");
      }
    },

    openJobTerminal(jobId) {
      this.currentJobId = jobId;
      this.terminalLogs = "";
      this.logDrawerOpen = true;

      if (this.eventSource) {
        this.eventSource.close();
      }

      this.eventSource = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream`);

      this.eventSource.addEventListener("log", (e) => {
        try {
          const text = JSON.parse(e.data);
          this.terminalLogs += text;
          if (this.autoScrollLogs) {
            this.$nextTick(() => {
              const el = document.getElementById("terminal-output");
              if (el) el.scrollTop = el.scrollHeight;
            });
          }
        } catch (err) {}
      });

      this.eventSource.addEventListener("status", (e) => {
        try {
          const data = JSON.parse(e.data);
          this.terminalLogs += `\n[系统状态更新: ${data.status}]\n`;
          this.loadJobs();
        } catch (err) {}
      });

      this.eventSource.onerror = () => {
        this.eventSource.close();
      };
    },

    closeJobTerminal() {
      this.logDrawerOpen = false;
      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }
    },

    async cancelJob(jobId) {
      if (!confirm(`确认取消任务 ${jobId} 吗？`)) return;
      try {
        const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
        if (res.ok) {
          this.toast("已发送取消请求", "warning");
          await this.loadJobs();
        }
      } catch (e) {
        this.toast("取消请求失败", "error");
      }
    },

    async resumeJob(jobId) {
      try {
        const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/resume`, { method: "POST" });
        if (res.ok) {
          const job = await res.json();
          this.toast(`续跑任务已提交: ${job.id}`, "success");
          await this.loadJobs();
          this.openJobTerminal(job.id);
        }
      } catch (e) {
        this.toast("续跑请求异常", "error");
      }
    },

    // -------------------------------------------------------------
    // 评测结果与指标
    // -------------------------------------------------------------
    async loadRuns() {
      if (!this.currentDataset) return;
      try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(this.currentDataset)}/runs`);
        if (res.ok) {
          this.runs = await res.json();
          if (this.runs.length > 0 && !this.selectedRun) {
            this.selectRun(this.runs[0]);
          }
        }
      } catch (e) {
        this.toast("加载评测结果失败", "error");
      }
    },

    async selectRun(run) {
      this.selectedRun = run;
      await this.loadRunMetrics();
      await this.loadScored(0);
    },

    async loadRunMetrics() {
      if (!this.selectedRun) return;
      const { model, backend } = this.selectedRun;
      try {
        const res = await fetch(
          `/api/datasets/${encodeURIComponent(this.currentDataset)}/runs/${encodeURIComponent(model)}/${encodeURIComponent(backend)}/metrics`
        );
        if (res.ok) {
          this.runMetrics = await res.json();
        }
      } catch (e) {}
    },

    async loadScored(offset = 0) {
      if (!this.selectedRun) return;
      this.scoredOffset = offset;
      const { model, backend } = this.selectedRun;
      const q = new URLSearchParams({
        offset: this.scoredOffset,
        limit: this.scoredLimit,
        order: this.scoredFilterOrder,
      });

      try {
        const res = await fetch(
          `/api/datasets/${encodeURIComponent(this.currentDataset)}/runs/${encodeURIComponent(model)}/${encodeURIComponent(backend)}/scored?${q.toString()}`
        );
        if (res.ok) {
          const data = await res.json();
          this.scoredRecords = data.records;
          this.scoredTotal = data.total;
        }
      } catch (e) {
        this.toast("加载得分记录失败", "error");
      }
    },

    // -------------------------------------------------------------
    // 诊断与质检
    // -------------------------------------------------------------
    async loadHealth() {
      if (!this.currentDataset) return;
      try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(this.currentDataset)}/health`);
        if (res.ok) {
          this.healthReport = await res.json();
        }
      } catch (e) {
        this.toast("获取诊断报告失败", "error");
      }
    },

    // -------------------------------------------------------------
    // Toast 提示
    // -------------------------------------------------------------
    toast(msg, type = "info") {
      const container = document.getElementById("toast-container");
      if (!container) return;
      const el = document.createElement("div");
      el.className = `toast toast-${type}`;
      el.innerText = msg;
      container.appendChild(el);
      setTimeout(() => {
        el.style.opacity = "0";
        setTimeout(() => el.remove(), 300);
      }, 3500);
    },
  }));
});
