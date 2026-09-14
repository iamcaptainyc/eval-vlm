# eval_vlm Web UI — 实施方案

## Context(为什么做这件事)

`eval_vlm` 是一个**解耦的 VLM 测试集评测工具**:所有操作都通过 CLI(`eval-vlm split/pred/score/eval/field-eval/sweep/report ...`)在一个 workspace(`~/eval_vlm_workspace/<数据集>/`)上进行,阶段之间只靠文件产物(test.json → predictions.jsonl → scored.jsonl / metrics.json)连接。

当前**没有任何可视化层**:审查测试集与图片、编辑每个数据集的 config、切换模型/后端、跑 pred/eval/sweep、看指标,都得在终端敲命令,事后再打开自包含 HTML 报告。人工要发现"图不对/标注不对"的坏样本、删掉它们、再重跑,链路很长。

**目标**:开发一个 Web UI,让上述操作可视化,尤其是(用户明确的**最重要一点**)——把 `test.json` 与**真实图片**一起展示在网页里,人工在网页中**删除不符合要求的样本**,后端**同步修改 test.json**(可备份/回滚);此外把配置编辑、任务执行(pred/eval/sweep,实时日志)、结果查看,以及若干**自动化审核/排错**功能一并纳入。

## 已锁定的产品决策(来自用户)

1. **技术栈**:FastAPI 后端 + 免构建轻量前端(单页 HTML + CDN 引入 Alpine.js/htmx,无 npm 构建)。
2. **能执行任务**:UI 以**子进程调用 `eval-vlm` CLI**(复用现有 GPU/conda 环境)跑 pred/eval/sweep,**SSE 实时日志/进度**,可取消、可续跑。
3. **删除语义**:**删整条样本** + **自动时间戳备份 + 软删除(trash)可回滚**;多图样本可选"只删单图"(须同步移除一个 `<image>` 占位符)。
4. **部署**:**受信任网络内使用**——任务队列(GPU 重任务串行)、test.json 编辑加锁。
5. **删除后旧结果的处理**:**失效标记 + 提示重跑**(不改动 predictions/scored 文件,受影响 run 打 stale 标记,由用户显式重跑)。

## 关键约束(已核对源码,精确)

- `Sample.id = f"{index:06d}-{sha1(json.dumps(record, sort_keys=True, ensure_ascii=False))[:8]}"`(`loader.py:33-37`)——**id 内嵌样本在 test.json 中的位置**。物理删除一条记录会让其后所有记录 index 前移 → id 改变 → 已有 `predictions.jsonl`/`scored.jsonl`(按 `id+turn` 对齐)错位。删除定位/回滚逻辑必须**逐字复刻**该哈希(含 `ensure_ascii=False`),否则 id 对不上。
- `<image>` 不变式(`loader.py:116-122`):**仅当 `images` 非空时**校验 `sum(turn.content.count("<image>") for turn in turns) == len(images)`,且跨**所有**轮计数。单图删除必须同步删一个占位符。
- test.json 写盘格式(`splitter.py:88-93`):`json.dump(subset, f, ensure_ascii=False, indent=2)`——重写必须字节级一致。
- `split_meta.json`(`splitter.py:95-111`):`indices["test"]` 是**原始源文件**的索引且**已排序**(`splitter.py:60`),故 test.json 位置 `p` ↔ `indices["test"][p]`;`source_sha256` 是**原始源文件**的哈希,**不是** test.json 的哈希——**删 test 样本不会使 source_sha256 失效**。
- `predictions.jsonl`/`scored.jsonl` 按 `(id, turn)` 对齐(`store.py:26-45`,`evaluate.py:91-102`);`PredictionWriter.write` **每行 flush**(`store.py:104`)→ 硬中断也不丢已写结果(利于取消/续跑)。
- `discover_run_dirs(dataset_dir)`(`store.py:151`)返回含 `metrics.json/precision.json/run_meta.json/pred_meta.json` 任一的 `<model>/<backend>/` 目录。
- `run_eval_once`(`cli.py:359`)、`run_field_eval_once`(`cli.py:299`)、`run_sweep`(`sweep.py:100`)是 **`argparse.Namespace` 耦合**的 → **只经子进程调用,不 in-process 调**。
- CLI 入口 `eval-vlm = eval_vlm.cli:main` 且支持 `python -m eval_vlm`;`cli._force_utf8_stdout()`(`cli.py:864`)重置 UTF-8,子进程仍应设 `PYTHONIOENCODING=utf-8`(Windows 稳妥)。

## 架构总览

新增包 `src/eval_vlm/webui/`,**不改动**现有 CLI/核心逻辑;新增可选依赖 extra `[webui]`。后端两种调用方式:

- **轻/只读**(列数据集、读样本、解析图片、读 metrics、改 config)→ **直接 import** cfg 驱动 API(见"复用清单")。
- **重任务**(pred/eval/field-eval/sweep/split)→ **spawn `[sys.executable, "-m", "eval_vlm", <subcmd>, ...]`**(服务本身跑在 claudepy 解释器下,`sys.executable` 即 claudepy,天然复用 GPU/conda 环境、resume、UTF-8 处理),经统一**串行任务队列**,日志落盘 + SSE 推流。

### 建议模块布局

```
src/eval_vlm/webui/
  __init__.py
  __main__.py       # python -m eval_vlm.webui -> uvicorn.run(app)
  app.py            # FastAPI app factory:挂载路由 + StaticFiles(同源,无需 CORS)
  settings.py       # host/port、workspace 与运行目录解析
  audit.py          # 本地 WebUI 写操作审计
  deps.py           # get_global_cfg / 解析数据集夹 / get_dataset_cfg(缓存)
  locks.py          # 每数据集 async 锁 + 磁盘锁文件 + 乐观 sha 校验
  jobs.py           # JobManager:串行队列、子进程、SSE 广播、落盘、重启对账
  editing.py        # 删除/恢复样本、单图删除、split_meta 同步、trash/备份、下游失效标记  <-- 核心
  datasets.py       # 只读浏览:列表、详情、分页样本、图片解析
  configio.py       # 读 config.yaml(结构化)+ 经 set_dataset_value 写回;模板段 introspect
  runsio.py         # run 目录、metrics.json、scored.jsonl、report、failures.html 透传
  automation.py     # 最低分审核、健康检查、run diff
  models.py         # pydantic 请求/响应 schema
  routers/          # datasets.py config.py jobs.py editing.py results.py automation.py
  static/           # index.html + app.js + styles.css(Alpine/htmx via CDN)
```

放在**包内**,`python -m eval_vlm.webui` 可在任意 workspace 启动,且能直接 `from ..config import ...` / `from ..data.loader import ...`。

### 运行态目录

服务状态全部放 **`<workspace>/_webui/`**(与 sweep 的 `_sweep/` 同级,不入 git):

```
_webui/
  audit.log.jsonl               # 所有变更操作审计(who/what/when)
  locks/<dataset>.lock
  jobs/<job_id>/{meta.json, log.txt}
  trash/<dataset>/<ts>-<sampleid>/{record.json, manifest.json, test.json.bak}
```

## 访问范围与并发

- **访问范围**: 当前 WebUI 不提供鉴权或用户角色；仅在可信本机或受保护内网使用。写操作统一以 `local` 写入审计日志。
- **任务并发**:单一**全局串行队列**(GPU 重任务不能并行);多用户提交进同一队列,前端显示排队位。sweep 本身也串行跑多数据集,天然契合。
- **编辑并发**:test.json 删除/恢复走 `editing.py`,**每数据集 async 锁 + 磁盘锁文件**,读改写原子(写 `.tmp` 再 `replace`,复用 `store.write_json` 模式);**乐观 sha 校验**做第二道防线(见删除机制)。默认**当该数据集有任务正在跑时禁止删除**(避免 run 中途 test.json 变化污染该 run)。
- **只读**:样本浏览/图片流无锁,可并发。

## REST + SSE API(具体端点;`[V]`=viewer,`[E]`=editor)

**数据集与样本**
- `GET /api/datasets` `[V]` — 列 workspace 下含 `config.yaml` 的数据集(test 样本数、#runs via `discover_run_dirs`)。
- `GET /api/datasets/{name}` `[V]` — 详情:config 摘要、split_meta、test.json sha256、run 列表、健康摘要。
- `GET /api/datasets/{name}/samples?offset=&limit=&filter=` `[V]` — 分页样本(`load_raw_records` + `_parse_record`:id/turns/images/targets/meta + 每图 exists 标记);**不**在此返回 base64。
- `GET /api/datasets/{name}/image?ref=<urlenc>&thumb=1` `[V]` — 经 `resolve_image_path(ref, cfg)` 定位;**校验 ref 属于该数据集图片集合**(防目录穿越);`FileResponse`(原图)或 Pillow 缩略(thumb=1,复用 `report_assets.MAX_SIDE=768`);http/data ref 透传/302。

**配置编辑**
- `GET /api/datasets/{name}/config` `[V]` — 结构化 config + 原文 + 模板段 schema(供表单渲染)。
- `PUT /api/datasets/{name}/config` `[E]` — 提交 `[{dotted_key, value}]`,逐键 `workspace.set_dataset_value`(保留注释,任意嵌套如 `inference.backend`/`inference.openai.model`/`scoring.scorer`)。持锁执行。

**任务**
- `POST /api/datasets/{name}/jobs` `[E]` — body `{type: split|pred|eval|field-eval, params}`;入队返回 job_id + 排队位。
- `POST /api/sweep/jobs` `[E]` — `{datasets, method, backend, ...}` → `eval-vlm sweep --dataset a,b,c`。
- `GET /api/jobs` `[V]` / `GET /api/jobs/{id}` `[V]` — 队列/详情。
- `GET /api/jobs/{id}/stream` `[V]`(SSE) — 实时 `event: log` / `event: status` / `event: progress`;支持 `?offset=` 从 `log.txt` 续读。
- `POST /api/jobs/{id}/cancel` `[E]`,`POST /api/jobs/{id}/resume` `[E]`。

**样本删除/恢复(核心)**
- `DELETE /api/datasets/{name}/samples/{sample_id}` `[E]` — body `{expected_sha256, reason?, mode: "record"|"image", image_index?}`。软删除→trash。返回新 sha + 下游失效报告。
- `POST /api/datasets/{name}/trash/{trash_id}/restore` `[E]`,`GET /api/datasets/{name}/trash` `[V]`。

**结果**
- `GET /api/datasets/{name}/runs` `[V]` — `discover_run_dirs` + metrics/run_meta 摘要 + **stale 标记**(run 记录的 test sha 是否 == 当前 test.json sha)。
- `GET .../runs/{model}/{backend}/metrics` `[V]`、`/scored?offset=&limit=&min_score=&max_score=` `[V]`、`/failures.html` `[V]`(透传已有产物)、`/report` `[V]`。

**自动化**(见下)
- `GET /api/datasets/{name}/health` `[V]`、`.../review?order=lowest&limit=` `[V]`、`.../diff?a=&b=` `[V]`。

## 任务执行细节

- **argv**:`[sys.executable, "-m", "eval_vlm", "eval", "-d", <name>, "--workspace", <ws>]`;后端/模型/scorer 等在**开跑前**经 config 编辑写回,argv 保持精简(避免复刻全部旗标);可选透传 `--backend/--overwrite/--fail-fast/--method`。
- **env**:继承 `os.environ`,强制 `PYTHONIOENCODING=utf-8`、`PYTHONUNBUFFERED=1`,确保 `EVAL_VLM_CONFIG` 指向服务同一份全局配置。
- **Windows**:`creationflags=CREATE_NEW_PROCESS_GROUP`(取消时发 `CTRL_BREAK_EVENT` 优雅停,再 terminate/kill);`stdout=PIPE, stderr=STDOUT` 合并,text/utf-8。
- **串行队列**:单 `asyncio.Queue` + 单 worker 协程;状态 `queued→running→(succeeded|failed|canceled|interrupted)`;`meta.json` 记 `id/type/dataset/params/user/status/pid/时间戳/exit_code/log_path/queue_position`。
- **SSE**:worker 逐行读子进程输出,追加 `log.txt` 且推给内存广播;连线时先按 `?offset=` 回放再转直播;tqdm 用 `\r` 写进度,按 `\n` 和 `\r` 都切分保持日志可读。
- **进度信号(稳健、后端无关)**:开跑时算 `expected = sum(len(s.targets) for s in load_samples(cfg, source=cfg.test_path))`,worker 每秒数 `predictions.jsonl` 行数 / expected;比解析 tqdm 可靠。另把 `[run] 待推理 N 条`、`[eval] 完成 X` 这类行作人类可读状态。
- **取消**:发 `CTRL_BREAK_EVENT`→ 宽限 → kill;因逐行 flush,已完成 `(id,turn)` 已落盘,安全。
- **续跑**:重提同一 job 即续跑(`run_inference` 经 `load_prediction_keys` 跳过已成功样本,`runner.py`);"Resume" 按钮=重提。
- **重启对账**:`JobManager` 启动扫 `jobs/*/meta.json`,`running` 但 pid 已死 → 标 `interrupted` 供一键续;未起的 queued 重入队。

## 删除同步机制(核心,`editing.py`;策略=失效标记+重跑)

**危害**:id 位移使下游错位,且多用户可能在位置 id 上竞争。**均在数据集锁内完成**:

1. **乐观 sha 守卫**:重算当前 test.json 的 sha256,若 ≠ 客户端传来的 `expected_sha256` → **409**(前端强制刷新并警告)。
2. **内容校验定位**:`load_raw_records`,找到 `_stable_id(pos, rec) == sample_id`(逐字复刻含 `ensure_ascii=False` 的哈希),并在当前位置重算确认仍匹配 → 否则 409(防位置段漂移误删)。
3. **备份/trash(可逆)**:整份 `test.json` → `_webui/trash/<name>/<ts>-<id>/test.json.bak`;写 `record.json`(被删记录)+ `manifest.json`(`{sample_id, position, original_source_index, deleted_by, reason, ts, test_sha_before, mode}`)。
4. **删除**:
   - `mode="record"`:从数组移除该记录。
   - `mode="image"`(多图):移除 `images[image_index]`,并按文档序删**第 image_index 个** `<image>` 占位符(定位到对应轮 `str.replace("<image>","",1)`),保持不变式;单图退化为删整条。
5. **重写** test.json:`json.dump(records, f, ensure_ascii=False, indent=2)`(字节级对齐 `splitter.py:92`),算 `new_sha256`。
6. **split_meta.json 同步**:`load_split_meta` → 移除 `indices["test"][position]`(已排序,位置直映)、`counts["test"] -= 1`;**保持 `source_sha256`/`total_samples` 不变**(原始源未动);追加审计块 `webui_edits: [{ts, deleted_ids, deleted_original_indices, backup_path, user, mode}]` 并置 `test_modified_after_split: true`;`store.write_json` 原子写。
7. **下游失效(不改结果文件)**:因 id 位移,已有 run 与新 test.json 不再对齐。对 `discover_run_dirs` 出的每个 run 目录写 `dataset_dirty.json`(`{stale_since, reason, test_sha_before, test_sha_after, deleted_ids, by}`)。前端在这些 run 上显**红色 stale 标记"数据集已编辑,需重跑"**;重跑因 id 变化会全量重推(符合预期),**不做**结果文件的 id remap。
8. **审计**:追加 `_webui/audit.log.jsonl`。返回 `{new_sha256, shifted_ids_count, invalidated_runs}`。

**单图删除的额外点**:改了记录内容 → 即使无位置移动其 id/hash 也变;该样本的下游行必然失效 → 一律按失效处理(与整条删一致)。

**恢复**:读 `manifest.json`+`record.json`,重算当前 sha 守卫,按原 position 插回(位置失效则追加),重写 test.json + 反向修 split_meta 索引,并把受影响 run 标 stale;trash 条目转 `restored/` 留档。

**为何不改 `_stable_id`**:让 id 不随位置变需改 loader 语义,风险大、越界;失效标记方案零结果文件改动、最安全,契合用户选择。

## 前端(免构建)

单 `index.html` + CDN 的 **Alpine.js**/**htmx**;hash 路由(`#/datasets`、`#/datasets/{name}`、`#/datasets/{name}/samples`、`#/jobs`、`#/datasets/{name}/runs/...`);统一请求错误和 409 处理。

页面:
1. **数据集浏览**:表格/卡片(test 数、#runs、健康徽章)。
2. **数据集详情 + 配置编辑**:按模板段生成表单(data.mapping、inference.backend + 当前后端块、eval、scoring、split);模型/后端选择即改 `inference.*`;"保存 config"(PUT)后可"运行"(split/pred/eval/field-eval/sweep)建任务。
3. **样本画廊(核心)**:分页图片网格,每格缩略图走**流式** `/image?thumb=1`(而非 base64,浏览数百样本更快)、id、对话预览、meta;缺失图片(`exists=false`)醒目红标;每格**删除**(整条)+ 多图**删单图**;**批量多选删除**(一次持锁 = 一次备份);灯箱复用 `evaluate._render_failures_html` 的 CSS/交互。删除发 `expected_sha256`,成功后 toast "已删除(可恢复),N 个 run 已过期",409 则强制刷新。回收站抽屉支持恢复。
4. **任务面板**:提交/队列/历史 + `EventSource` 实时日志 + 进度条 + 取消/续跑。
5. **结果查看**:metrics 表 + scored 浏览(按分升序,一键"看最低分")+ 预测 vs 参考并排(分数徽章)+ 内嵌 `failures.html`。

## 附加自动化(优先级 → 底层产物)

| 优先级 | 功能 | 底层函数/产物 |
|---|---|---|
| P0 | **数据集健康检查**:图片缺失、`<image>`≠len(images)、media_root 孤儿图、http/data ref、重复 id | `load_raw_records`、`resolve_image_path`(`loader.py:145`)、占位符规则(`loader.py:116-122`) |
| P0 | **缺图自动标记**(画廊红标) | `resolve_image_path(...).exists()` |
| P0 | **最低分优先审核**(一键) | `scored.jsonl`(`score/detail/images`),升序;行 schema `evaluate.py:91-102` |
| P1 | **预测 vs 参考并排 + 分数徽章** | scored.jsonl;复用 `evaluate` 渲染样式 |
| P1 | **批量删除** | 删除 API,一次锁内批处理 |
| P1 | **pred 实时指标** | `predictions.jsonl` 计数 / expected(同进度信号) |
| P2 | **两 run diff**(一致率/逐样本差异/分差) | `discover_run_dirs`、`load_predictions`;参考 `precision`/`report` |
| P2 | **后端 config 预设**(openai/mnn/hf/vllm_offline) | `config.template.yaml` 各段;`set_dataset_value` 应用 |
| P2 | **stale 横幅** | run 记录 test sha vs 当前 test.json sha |
| P3 | **field-eval 失配查看** | `field_mismatches.json`/`field_metrics.json`(Config 属性) |
| P3 | **混淆矩阵** | `metrics.json` `per_turn[*].confusion_matrix` |

## 复用清单(现有函数,勿重造)

- 工作区/配置:`workspace.load_global_config`/`resolve_workspace`(`workspace.py:149,305`)、`resolve_dataset_dir`(`:337`)、`set_dataset_value`(`:533`)、`describe_settable_keys`(`:184`)、`render_template`(`:369`);`config.load_dataset_config`(`config.py:703`)、`Config` 路径属性(`config.py:497-650`)、`safe_model_dirname`(`config.py:23`)。
- 数据:`loader.load_raw_records`(`:40`)、`load_samples`(`:53`)、`_parse_record`(`:71`)、`resolve_image_path`(`:145`)、`_stable_id`(`:33`)。
- 划分:`splitter.load_split_meta`(`:121`);写格式(`:88-93`)。
- 结果 IO:`store.load_predictions`(`:72`)、`load_prediction_keys`(`:26`)、`write_json`(`:121`,原子)、`discover_run_dirs`(`:151`)。
- 图片:`report_assets.image_ref_to_html_src`(`:160`)、`batch_preload_images`(`:89`)、`MAX_SIDE`(`:24`)——缩略/报告导出;实时流式优先 `resolve_image_path` + FileResponse。
- HTML 模板参考:`evaluate.py:_render_failures_html`(约 398 行,含搜索/筛选/灯箱)。
- 子进程目标:`python -m eval_vlm {split,pred,eval,field-eval,sweep}`(`cli.py:616-859`);**不 in-process 调** `run_eval_once`/`run_field_eval_once`/`run_sweep`。

## 依赖变更

`pyproject.toml` 新增(核心依赖不动):

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0"]
webui = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.29",
  "python-multipart>=0.0.9",
  "sse-starlette>=1.8",     # 或手写 StreamingResponse
]

[project.scripts]
eval-vlm = "eval_vlm.cli:main"
eval-vlm-webui = "eval_vlm.webui.__main__:main"
```

安装:`C:/Users/chanceyu/AppData/Local/miniconda3/envs/claudepy/python.exe -m pip install -e ".[webui]"`。前端库走 CDN,不进 Python 依赖。`_webui/` 在 workspace(非 repo)内,无需改 `.gitignore`(仓库 `.gitignore` 已忽略 outputs/.omc/.claude)。

## 分期(Phasing)

- **Phase 0 — 骨架**:包 + app factory + `python -m eval_vlm.webui`;`GET /api/datasets`、`/{name}`;静态壳。冒烟:claudepy 下起服务、列出 fixture workspace。
- **Phase 1 — MVP(用户最看重)**:样本画廊(分页)+ 流式图片路由(缩略 + 缺图标记)+ **删除同步机制**(整条 + 单图)含 trash/恢复、split_meta 同步、失效标记、每数据集锁、乐观 sha 守卫。editing 全 pytest 覆盖。
- **Phase 2 — 配置 + 任务**:config 读写(`set_dataset_value`)+ JobManager(串行队列/子进程/SSE 日志+进度/取消/续跑/重启对账)+ 任务面板 UI + 跑任务时禁删。
- **Phase 3 — 结果 + 自动化**:runs/metrics/scored 查看、failures.html 透传、最低分审核、健康检查、run diff、预设与 stale 横幅。

## 验证(端到端)

解释器统一:`C:/Users/chanceyu/AppData/Local/miniconda3/envs/claudepy/python.exe`。

1. **单测(pytest,复用 `tests/test_workspace.py` 的 `EVAL_VLM_CONFIG` monkeypatch + fake 后端 fixture)**:
   - **删除同步**(`test_webui_editing.py`):临时 workspace `init_dataset`+`split_dataset`(fake)+ `run_inference`+`score_predictions` 造 predictions/scored。删中间一条 → 断言 test.json 重写字节一致(indent=2/ensure_ascii=False)、后续 id 按预期前移、`split_meta.counts["test"]` 减一且移除对应 `indices["test"]`、`source_sha256` 不变、受影响 run 写了 `dataset_dirty.json`;单图删除 → 一个 `<image>` 移除、`load_samples` 不抛 `DataFormatError`;恢复往返还原(除审计块);乐观守卫 → 旧 `expected_sha256` 触发 409。
   - **锁**(`test_webui_locks.py`):两并发删除协程串行、无交错;磁盘锁拒第二持有者。
   - **API**(`test_webui_api.py`,Starlette `TestClient`):无凭证 401;列表/详情/分页;图片路由拒非本数据集 ref(穿越守卫);config PUT 经 `set_dataset_value` 持久化且保留注释。
2. **任务 + SSE 冒烟**:用 fake 后端 `eval` 任务(无需 GPU/网络)→ 断言 `succeeded`、predictions 产出、SSE 有 `log/progress/status`;取消 → `canceled` 且已 flush 结果保留、续跑可完成;重启对账 → 伪造 pid 已死的 `running` meta 被标 `interrupted`。
3. **手动/集成**:`python -m eval_vlm.webui --host 127.0.0.1 --port 8080` → 浏览器:浏览→删一条(见 trash + run stale 标)→恢复;再对真实模型起 `eval` 看实时日志+进度。回归:`python -m pytest` 现有套件保持绿(除 pyproject 外未改核心)。
4. **lint**:`ruff check src/eval_vlm/webui`。

## 风险与注意

- **id 位移不可避免**:采用"失效标记 + 提示重跑"(用户选择),不改结果文件,最安全。
- **图片路径穿越**:`/image?ref=` 必须限定在 `media_root_path` 内,解析后校验 ref 属于该数据集图片集合。
- **多用户竞争**:文件锁 + 乐观 sha 守卫 + 内容哈希定位,防位置段漂移误删。
- **跑任务时编辑**:默认禁止对有活跃任务的数据集删样本,避免污染该 run。
- **子进程环境**:沿用 claudepy/GPU;队列串行避免显存冲突。
