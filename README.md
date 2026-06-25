# eval_vlm

解耦的 **VLM(视觉语言模型)测试集评测工具**。用于评估刚训练完的 VLM 在测试集上的表现。

数据源使用 **LlamaFactory 数据集格式**(sharegpt 风格 JSON),推理通过 **OpenAI 兼容 API** 调用(模型用 vLLM / SGLang / LlamaFactory `api` 部署成 HTTP 服务,**部署与评测彻底分离**)。

## 典型工作流

1. 用大模型按数据源生成**完整数据集**(LlamaFactory 格式 JSON);
2. `split` 把它**纯分割**成 `train.json` / `test.json`(`val.json` 可选)——三份都是**原样 LlamaFactory 格式**;
3. 拿 `train.json` 去 **LlamaFactory 训练**模型;
4. 训练好的模型用 **vLLM 部署成 OpenAI 兼容 API**;
5. 本项目 `pred --dataset` + `score`(或一键 `eval`)拿 `test.json` 评测该 API。

## 设计:三步解耦

三个步骤是**独立 CLI 命令**,步骤间只通过**文件产物**交接,可在不同机器、不同时间独立运行与重跑:

```
完整数据集 JSON ─split─▶ train.json / val.json / test.json ─pred --dataset─▶ predictions.jsonl ─score─▶ metrics.json
 (LlamaFactory)            (均为原样 LlamaFactory 格式)        (原始预测+原图地址,可断点续跑)  scored.jsonl / failures.md / summary.md
```

- **split 只做纯分割**:每条记录原样写出(答案、对话结构、**图片路径全不动**),`train.json` 直接能喂给 LlamaFactory 训练。
- `pred --dataset` 按 LlamaFactory 格式读 `test.json` → 调推理后端 → 拿预测;不关心数据怎么切的。
- `score` 把预测和标准答案(最后一个 assistant 轮)按 id 对齐打分;不关心预测怎么来的。
- 推理后端、评分器都**可插拔**:换后端改一行配置,加评分器只需新增一个文件。

### 多轮对话与逐轮评测(rollout)

数据常是**多轮对话**(例如:轮1 让模型描述图片,轮2 让模型从选项里选情绪标签)。
本工具支持**逐轮评测**:默认对**每个 assistant 轮**都打分(轮1描述 + 轮2标签),由 `eval.targets` 控制:

- `eval.targets: all`(默认)—— 每个 assistant 轮都评;`last` —— 仅最后一轮(只评标签)。
- `eval.context: rollout`(默认)—— 生成下一轮时,前面 assistant 轮用**模型自己生成**的内容作上下文(真·连续对话:给图→模型描述→再问→模型答标签,误差会累积);`gold` —— 用数据集标准前文(教师强制,各轮独立评测)。

一条样本因此会产生**多条预测**,用 `(id, turn)` 唯一标识;评分时按 `(id, turn)` 对齐。
标签/答案完全由数据决定,代码里不写死。

### 逐轮用不同 scorer

不同轮性质不同(描述是开放式、标签是分类),可在 `scoring.turn_scorers` 里**按目标顺序**分别指定,缺省回落 `scoring.scorer`:

```yaml
scoring:
  scorer: exact_match            # 默认
  turn_scorers: [token_f1, exact_match]   # 轮1描述用 token_f1(相似度),轮2标签用 exact_match
```

## 安装

```bash
pip install -r requirements.txt
# 或可编辑安装(提供 eval-vlm 命令):
pip install -e .
```

## 工作目录模型(自包含数据集文件夹)

为免去"每个数据集复制改一份 YAML、预测/评分还要回去改配置"的麻烦,本工具采用**工作目录模型**:

- **机器级设置**(工作目录 `workspace`、图片根 `media_root`、跨机前缀 `image_strip_prefix`)放**全局配置** `~/.eval_vlm/config.yaml`(`EVAL_VLM_CONFIG` 环境变量可改路径),所有数据集共享、只配一次。
- **每个数据集**是 `workspace/<数据集名>/` 一个**自包含文件夹**:`split` 时从内置模板自动生成该数据集的 `config.yaml`,连同 `split_meta.json`、`train/test.json` 落在里面(各模型共享)。
- **产物按模型分目录**:`pred/score/eval` 的结果落在 `workspace/<数据集>/<模型名>/`(组织为 **工作目录/数据集/模型**)。模型名按后端取:openai/vllm/fake 取 `inference.openai.model`,mnn 取 `inference.mnn.config_path` 所在目录名。换模型 = 换结果目录,**不同模型对同一数据集互不覆盖**;同一模型目录已存在则按断点续跑沿用/补齐。
- **`--dataset` 两种含义**:`split --dataset <源JSON>` 是**初始化**(在工作目录建同名文件夹);`pred/score/eval --dataset <名|路径>` 是**读取**已存在的数据集文件夹(自动找夹内 `config.yaml`)。
- **用户参数优先且持久化**:`--base-url/--model/--scorer/--backend/--mnn-config/--prompt` 等 CLI 覆盖会**永久写回**该数据集的 `config.yaml`(不再是临时),后续命令直接读到新值;当然也可直接手改 `config.yaml`。

### 首次设置(一次性)

```bash
eval-vlm config init                              # 生成 ~/.eval_vlm/config.yaml(带注释)
eval-vlm config keys                              # 列出所有可设置的键(类型/默认/说明)+ 哪些只能手改数据集 config.yaml
eval-vlm config set workspace /root/autodl-tmp/capt/eval_runs   # 所有数据集文件夹的父目录
eval-vlm config set media_root /root/autodl-tmp/capt/code/LlamaFactory   # 图片根
eval-vlm config show                              # 查看当前全局配置

# 可选:改 split 默认比例(不传 --train/--test 等时使用;命令行参数优先)
eval-vlm config set split.train 0.9               # 训练集默认比例
eval-vlm config set split.test 0.1                # 测试集默认比例
eval-vlm config set split.seed 7                  # 默认随机种子
```

> **split 比例的默认值从哪改**:不传 `--train/--test/--val/--seed/--stratify-by` 时,
> 取全局配置 `~/.eval_vlm/config.yaml` 的 `split:` 块(出厂默认 train 0.95 / test 0.05 /
> val 0.0 / seed 42)。优先级:**命令行参数 > 全局 `split.*` > 内置默认**。用
> `eval-vlm config set split.<键> <值>` 或直接手改该文件均可。

### 快速开始(用内置 fixture 跑通,不联网)

```bash
# 1) 初始化数据集文件夹(--dataset = 源 JSON;--train/--test 设置比例)
eval-vlm split --dataset tests/fixtures/llamafactory_demo.json --train 0.6 --test 0.4
# -> 在 workspace/llamafactory_demo/ 生成 config.yaml + train/test.json

# 2) 把该文件夹内 config.yaml 的 inference.backend 改成 fake(离线回显)

# 3) 一键预测 + 评分(--dataset = 数据集名;eval = pred + score)
eval-vlm eval --dataset llamafactory_demo
# 结果在 workspace/llamafactory_demo/
```

### 评测真实模型(OpenAI 兼容服务)

```bash
# 1) 初始化数据集(可一次性传入划分参数)
eval-vlm split --dataset /root/autodl-tmp/capt/data/emo_v4.json --train 0.95 --test 0.05

# 2) 拿 workspace/emo_v4/train.json 去 LlamaFactory 训练;再用 vLLM 部署成 HTTP 服务:
#    vllm serve /path/to/checkpoint --served-model-name trained-vlm --port 8000
#    并把 emo_v4/config.yaml 的 inference.openai.base_url / model 改好(或下面用命令行传入,会永久写回)

# 3) 一键预测 + 评分(eval = pred + score)
export OPENAI_API_KEY=EMPTY
eval-vlm eval --dataset emo_v4 --base-url http://localhost:8000/v1 --model trained-vlm
```

> **split 与预测/评分不连续**:split 之后要先训练 + 部署模型,故 `eval` 命令只连续执行 **pred + score**,不含 split。

### 命令一览

| 命令 | `--dataset` / `--datadir` 含义 | 作用 |
| --- | --- | --- |
| `config init / show / set <k> <v> / keys` | — | 管理全局配置;`keys` 列出全部可设置键(workspace/media_root/image_strip_prefix + split 默认比例 `split.*`)及不可全局设的数据集级项 |
| `split --dataset <源JSON>` | 源数据集 JSON 路径 | **初始化**:建文件夹 + 生成 config.yaml + 分割 |
| `pred --dataset <名\|路径>` | 已存在数据集 | 读 test.json → `<数据集>/<模型>/predictions.jsonl`(只预测,不评分;等价旧 `run`) |
| `pred --datadir <图片文件夹>` | 无标注图片文件夹 | **无标注图片描述**:逐张调 VLM,产物落 `workspace/<同名>/<模型>/`(不评分) |
| `score --dataset <名\|路径>` | 已存在数据集 | 评分 → `<数据集>/<模型>/` 下 metrics/scored/failures/summary |
| `eval --dataset <名\|路径>` | 已存在数据集 | 一键 **pred + score**(不含 split) |

> `pred` 用 `--dataset` 与 `--datadir` **二选一**(互斥,必填其一):前者预测已分割数据集的 `test.json`,后者描述一整个无标注图片文件夹。

### split 参数 / CLI 覆盖(永久写回 config.yaml)

> `--base-url/--model/--scorer/--backend/--mnn-config/--mnn-image-max-side/--prompt/--system-prompt` 会**永久写回**该数据集的 `config.yaml`(用户参数优先且持久化),后续命令直接读到。

| 参数 | 适用命令 | 作用 |
| --- | --- | --- |
| `--train / --test / --val` | split | 划分**比例**(如 `--train 0.8 --test 0.2`) |
| `--seed` / `--stratify-by` | split | 随机种子 / 分层抽样字段 |
| `--name NAME` | split / pred --datadir | 数据集/输出文件夹名(默认取源文件名或图片文件夹名) |
| `--force` | split / pred --datadir | 已存在时重建 config.yaml(split 还会重新分割) |
| `--train-out / --val-out / --test-out` | split | 把对应产物直接写到任意目录(如 LlamaFactory `data/`) |
| `--base-url / --model` | pred / eval | **写回** `inference.openai.base_url / model`;`model` 同时决定 openai 后端产物子目录名 |
| `--scorer` | score / eval | **写回** `scoring.scorer` |
| `--backend openai\|vllm\|mnn\|fake` | pred | **写回** `inference.backend`(openai/vllm=调 OpenAI 兼容 API;mnn=本地 pymnn 推理;fake=回显不联网,自检用) |
| `--mnn-config FILE` | pred | `--backend mnn` 时:转换产物目录里 `config.json` 的路径(**写回** `inference.mnn.config_path`;也据其所在目录名定产物子目录) |
| `--mnn-image-max-side N` | pred | mnn 后端图片最长边像素上限(超大图等比缩放防 native segfault;设 0 关闭)(**写回** `inference.mnn.image_max_side`) |
| `--prompt TEXT` | pred --datadir | **写回** `pred.prompt`(默认 `请描述图片`;设了多轮 `template` 时无效) |
| `--system-prompt TEXT` | pred --datadir | **写回** `pred.system_prompt` |
| `--overwrite` | pred --datadir | 无视已有结果整份重跑(覆盖 `predictions.jsonl`);默认断点续跑只补未完成 |
| `--workspace DIR` | split/pred/score/eval | 临时覆盖全局 workspace |

### pred --datadir(无标注图片描述)

`pred --dataset`/`score` 面向**带标注的测试集**(数据集 JSON → split → 评分)。当你只有**一堆无标注照片**、
想让模型逐张描述时,用 `pred --datadir`:它不需要 split、没有标准答案、**不评分**,每张图各起一段独立对话:

```
user:      <image>请描述图片
assistant: <模型生成的描述>
```

```bash
# 自检(不联网,回显验证全链路):产物落到 <workspace>/test_images/
eval-vlm pred --datadir test_images --backend fake

# 调真实部署的模型(OpenAI 兼容服务,vLLM/SGLang/LlamaFactory api)
eval-vlm pred --datadir test_images --base-url http://localhost:8000/v1 --model trained-vlm
# 自定义提示词 / 系统提示 / 输出文件夹名
eval-vlm pred --datadir ./photos --prompt "用一句话描述这张图" --system-prompt "你是图像描述助手" --name photos_desc

# 本地 MNN 推理(训练后转成 mnn 的模型,无需起服务)
eval-vlm pred --datadir ./photos --backend mnn --mnn-config /root/qwen2-vl-mnn/config.json
```

#### 本地 MNN 后端(`--backend mnn`)

训练完 VLM 后转成 MNN 格式(`llm.mnn` / `llm.mnn.weight` / `tokenizer.mtok` / `config.json` 等
放同一目录),即可用 **pymnn 在本机直接推理**,无需先把模型部署成 HTTP 服务——和
`openai`/`vllm` 后端(调远端 OpenAI 兼容 API)互补。`mnn-infer` 即 `pred --backend mnn`,
专用于**无标注图片文件夹**的推理,产物(`predictions.jsonl`/`failures.jsonl`/`pred_meta.json`)
与其它后端完全一致。

```bash
# 一次性指定 MNN 模型 config.json
eval-vlm pred --datadir ./photos --backend mnn --mnn-config /root/qwen2-vl-mnn/config.json
# 或在文件夹 config.yaml 的 inference.mnn.config_path 写死后直接:
eval-vlm pred --datadir ./photos --backend mnn
```

**前置依赖**:需安装**带 LLM 支持**的 pymnn(`MNN.llm` / `MNN.cv`)。一般需用
`-DMNN_BUILD_LLM=ON` 编译;多模态(VL)再加 `-DMNN_BUILD_LLM_OMNI=ON`。未安装时用
`--backend mnn` 会给出明确报错,不影响其它后端使用。

**工作原理**(对齐 MNN 官方 pymnn 接口):图片用 `MNN.cv.imread` 原生读为 Var(免 numpy 往返),
按 `{'text': '<img>image_0</img>…', 'images': [{'data': img, 'height': H, 'width': W}]}` 组织;
`height`/`width` 由图片自动推导。调用 `model.response(prompt, stream=False, max_new_tokens)`
**直接返回完整生成文本**(`max_new_tokens` 取 `inference.mnn.max_tokens`),无需手写解码循环。

**约束**:pymnn 的 LLM 无批量接口且单个模型对象有状态(KV cache),因此 MNN 后端**强制串行**;
仅支持**单图单轮**(`pred --datadir` 的默认场景)。mnn 块只含它真正会用到的设置
(`config_path` / `image_max_side` / `max_tokens`),没有 `base_url`/`model`/并发等无意义字段;
采样参数请在 MNN 自己的 `config.json` 里配。

#### 自定义 vLLM API 与对话组织(`config.yaml`)

`pred --datadir` 沿用**自包含文件夹模型**(同 `split`→`pred --dataset`):**首次运行**在 `<workspace>/<名>/` 生成一份
`config.yaml`,**再次运行**直接读它;描述产物按模型落 `<workspace>/<名>/<模型名>/`。想完整定制
**vLLM API** 或**每张图的对话怎么组织**,可手改这份 `config.yaml`(`--force` 可重新生成,覆盖手改),
或直接用 CLI flag —— **用户参数会永久写回 `config.yaml`**(用户参数优先且持久化)。
重新跑想覆盖旧结果用 `--overwrite`(默认断点续跑,只补未完成)。

> 所有命令(`split`/`pred`/`score`/`eval`)共用**同一个统一模板**,所以 pred 生成的
> `config.yaml` 也会带 `split` / `eval` / `scoring` 段——它们对 pred --datadir 是**惰性的**(每段都标了「谁用」),
> 留默认即可、无副作用;pred --datadir 只读 `data.media_root` / `inference` / `pred` 三处。

- **vLLM API**:`inference.openai:` 块全参可调 —— `base_url` / `model` / `api_key_env` / `max_tokens` /
  `temperature` / `max_concurrency` / `request_timeout` / `max_retries` / `image_detail` / `system_prompt`
  (`backend=mnn` 时改用 `inference.mnn:` 块的 `config_path` / `image_max_side` / `max_tokens`)。
- **对话组织**:`pred:` 块。两种写法二选一:
  - **单轮简写**:只设 `prompt`(+可选 `system_prompt`)。若 `prompt` 不含 `<image>`,自动在最前面加一个。
  - **多轮模板**:设 `template`(`role`/`content` 列表),**覆盖** `prompt`,可加纯文本 `assistant`/`user` 轮做 few-shot 引导。

```yaml
pred:
  system_prompt: 你是专业的图像描述助手        # 映射到当前后端块的 system_prompt(openai)
  template:                                    # 多轮:覆盖 prompt
    - {role: user, content: "我会给你一张图片,请用简体中文、200 字内客观描述。"}
    - {role: assistant, content: "好的,请提供图片。"}
    - {role: user, content: "<image>请描述这张图片"}
```

> 模板规则(否则构造时即报错):全部轮里 `<image>` **恰好出现 1 次**且**位于某个 `user` 轮**;**最后一轮必须是 `user`**(模型据此作答)。

产物组织为 `<workspace>/<图片文件夹名>/<模型名>/`(`config.yaml` 在其父级,各模型共享;
若图片文件夹路径正是输出文件夹本身会报错,用 `--name` 区分):

| 文件 | 位置 | 内容 |
| --- | --- | --- |
| `config.yaml` | `<名>/` | 该次 pred 的自包含配置(首次生成;重跑读取);CLI flag 会写回这里定制 vLLM API 与对话组织 |
| `predictions.jsonl` | `<名>/<模型>/` | 每行一条成功描述,**原样 LlamaFactory 格式**(`messages` + `images`,多轮模板会完整保留各轮),可直接当新数据集复用;额外带 `id`/`latency` 便于追溯。**追加写,支持断点续跑**(已成功图片自动跳过;`--overwrite` 整份重跑) |
| `failures.jsonl` | `<名>/<模型>/` | 每行一条失败记录(`id`/`image`/`error`),供排查与重跑(**仅反映本轮**:每次运行重写) |
| `pred_meta.json` | `<名>/<模型>/` | 运行元信息(模型/后端/对话结构/计数/时间) |

支持的图片扩展名:`.png/.jpg/.jpeg/.webp/.bmp/.gif/.tif/.tiff`(仅当前层,不递归)。

## 配置说明

配置分两层:

- **全局配置** `~/.eval_vlm/config.yaml`(机器级,一次性):`workspace` / `media_root` / `image_strip_prefix`,以及 `split:` 块(split 默认比例 train/test/val/seed/stratify_by)。用 `eval-vlm config set <k> <v>` 修改(嵌套用点号,如 `config set split.train 0.9`)。split 取值优先级:命令行参数 > 全局 `split.*` > 内置默认。
- **数据集配置** `<workspace>/<名>/config.yaml`(每个数据集一份,split 时从内置模板生成):下表各段。`configs/example.yaml` 是同结构的**可读参考**。

关键段:

| 段 | 作用 |
| --- | --- |
| `data` | 数据源路径、图片根目录、LlamaFactory 字段映射(对齐 `dataset_info.json`) |
| `split` | `ratio` 或 `count`、`seed`(确定性)、`stratify_by`(分层抽样) |
| `inference` | `backend` 选后端;`openai:` 块(base_url/model/并发/重试/超时等)与 `mnn:` 块(config_path/image_max_side/max_tokens)各自独立 |
| `eval` | `targets`(all/last)、`context`(rollout/gold):控制评测哪些轮、上下文来源 |
| `scoring` | `scorer`(默认,可被 `--scorer` 覆盖)、`turn_scorers`(逐轮指定) |

### LlamaFactory 两种格式

默认对齐 `mllm_demo`(`messages` + `role/content` + `user/assistant`)。若数据是通用 sharegpt(`conversations` + `from/value` + `human/gpt`),只改 `data.mapping` 即可,无需改代码:

```yaml
data:
  mapping:
    messages: conversations
    images: images
    tags: {role: from, content: value, user: human, assistant: gpt}
```

## 输出产物

数据集级(各模型共享)落 `<workspace>/<数据集名>/`;模型级(pred/score/eval 结果)落
`<workspace>/<数据集名>/<模型名>/` —— **不同模型对同一数据集互不覆盖**。

| 文件 | 位置 | 内容 |
| --- | --- | --- |
| `config.yaml` | `<数据集>/` | 该数据集的自包含配置(split 时从内置模板生成;pred/score/eval 直接读它;CLI flag 写回这里) |
| `train.json` / `test.json` / `val.json` | `<数据集>/` | 划分出的数据集,**均为原样 LlamaFactory 格式**(val 仅 `val>0` 时产出) |
| `split_meta.json` | `<数据集>/` | 划分元信息(seed/比例/计数/源哈希/原始下标),用于复现与审计 |
| `predictions.jsonl` | `<数据集>/<模型>/` | 每行一条预测(id/**turn**/prediction/**images**/latency/error),`images` 为原图地址可追溯回原图人工核查;多轮下每样本多行,**追加写,支持断点续跑** |
| `metrics.json` | `<数据集>/<模型>/` | 聚合指标(含 `per_turn` 逐轮分组指标 + `overall_mean_score` + `num_failed_samples`/`num_failed_targets`) |
| `scored.jsonl` | `<数据集>/<模型>/` | 逐(样本,轮)得分(id/turn/ordinal/scorer/score/**images**/...);机器可读全量数据 |
| `failures.md` | `<数据集>/<模型>/` | **exact_match 未命中清单(人类可读)**:仅纳入 exact_match 评分错误的样本,按 `id` 分组列出**全部对话轮**(模型输出 vs 标准答案 + ✓/✗ + 原图地址),供人工核查。非 exact_match(如 token_f1)不计入 |
| `summary.md` | `<数据集>/<模型>/` | 人类可读摘要(含未命中样本/目标轮数) |
| `run_meta.json` | `<数据集>/<模型>/` | 运行元信息(模型/配置/时间/计数),用于复现 |

**多模型对比**:同一数据集换不同模型(改 `config.yaml` 的 `inference.openai.model`、`--model`,或换 mnn 模型)重跑,结果各进
`<数据集>/<模型>/`,可并列对比、互不覆盖。

**断点续跑**:`pred --dataset` 会跳过该模型目录 `predictions.jsonl` 中已成功的 id,只补缺失/失败的——大测试集中断后重跑无需从头。

## 扩展评分器

内置 scorer:`exact_match`(归一化精确匹配 + 子串命中,适合多选 / 短答案 / 标签)与 `token_f1`(字符级 P/R/F1,中文友好,适合轮1这类开放式描述)。新增评分器:

1. 在 `src/eval_vlm/scoring/` 下新建文件,继承 `Scorer`;
2. 用 `@register("your_name")` 装饰;
3. 在 `scoring/__init__.py` 里 import 一次以触发注册;
4. 用 `--scorer your_name` 或配置 `scoring.scorer` 选用。

核心代码无需改动(典型可扩展:`bleu`、`rouge`、`llm_judge` 等)。

## 目录结构

```
src/eval_vlm/
├── config.py            # 配置加载(YAML -> dataclass)
├── data/
│   ├── schema.py        # Sample / Prediction
│   ├── loader.py        # 解析 LlamaFactory JSON(字段映射可配)
│   └── splitter.py      # 确定性划分测试集
├── inference/
│   ├── base.py          # 后端接口(thread_safe 标志:有状态后端串行)
│   ├── openai_backend.py# OpenAI 兼容调用(图片 base64、并发、重试;vllm 为其别名)
│   ├── mnn_backend.py   # 本地 MNN(pymnn)推理(转换后的 mnn 模型,无需服务)
│   └── fake_backend.py  # 离线回显后端(测试/演示)
├── scoring/
│   ├── base.py / registry.py / exact_match.py / token_f1.py   # 可插拔评分
├── results/store.py     # 产物读写(断点续跑)
├── runner.py            # 执行测试(并发编排)
├── predict.py           # 无标注图片文件夹 -> 单轮描述(不评分)
├── evaluate.py          # 评分编排
├── workspace.py         # 工作目录模型:全局配置 + 数据集初始化/定位 + 模板渲染
├── templates/
│   └── config.template.yaml       # 统一配置模板(所有命令共用;split/pred 首次运行渲染)
└── cli.py               # config / split / pred / score / eval
```
