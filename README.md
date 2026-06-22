# eval_vlm

解耦的 **VLM(视觉语言模型)测试集评测工具**。用于评估刚训练完的 VLM 在测试集上的表现。

数据源使用 **LlamaFactory 数据集格式**(sharegpt 风格 JSON),推理通过 **OpenAI 兼容 API** 调用(模型用 vLLM / SGLang / LlamaFactory `api` 部署成 HTTP 服务,**部署与评测彻底分离**)。

## 典型工作流

1. 用大模型按数据源生成**完整数据集**(LlamaFactory 格式 JSON);
2. `split` 把它**纯分割**成 `train.json` / `test.json`(`val.json` 可选)——三份都是**原样 LlamaFactory 格式**;
3. 拿 `train.json` 去 **LlamaFactory 训练**模型;
4. 训练好的模型用 **vLLM 部署成 OpenAI 兼容 API**;
5. 本项目 `run` + `score` 拿 `test.json` 评测该 API。

## 设计:三步解耦

三个步骤是**独立 CLI 命令**,步骤间只通过**文件产物**交接,可在不同机器、不同时间独立运行与重跑:

```
完整数据集 JSON ──split──▶ train.json / val.json / test.json ──run──▶ predictions.jsonl ──score──▶ metrics.json
 (LlamaFactory)            (均为原样 LlamaFactory 格式)        (原始预测+原图地址,可断点续跑)  scored.jsonl / failures.jsonl / summary.md
```

- **split 只做纯分割**:每条记录原样写出(答案、对话结构、**图片路径全不动**),`train.json` 直接能喂给 LlamaFactory 训练。
- `run` 按 LlamaFactory 格式读 `test.json` → 调 API → 拿预测;不关心数据怎么切的。
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

为免去"每个数据集复制改一份 YAML、run/score 还要回去改配置"的麻烦,本工具采用**工作目录模型**:

- **机器级设置**(工作目录 `workspace`、图片根 `media_root`、跨机前缀 `image_strip_prefix`)放**全局配置** `~/.eval_vlm/config.yaml`(`EVAL_VLM_CONFIG` 环境变量可改路径),所有数据集共享、只配一次。
- **每个数据集**是 `workspace/<数据集名>/` 一个**自包含文件夹**:`split` 时从内置模板自动生成该数据集的 `config.yaml`,连同 `split_meta.json`、`train/test.json` 及后续全部产物都落在里面。
- **`--dataset` 两种含义**:`split --dataset <源JSON>` 是**初始化**(在工作目录建同名文件夹);`run/score/eval --dataset <名|路径>` 是**读取**已存在的数据集文件夹(自动找夹内 `config.yaml`)。
- 后续微调直接**手改文件夹内的 `config.yaml`**;`--base-url/--model/--scorer` 为临时覆盖,不回写。

### 首次设置(一次性)

```bash
eval-vlm config init                              # 生成 ~/.eval_vlm/config.yaml(带注释)
eval-vlm config set workspace /root/autodl-tmp/capt/eval_runs   # 所有数据集文件夹的父目录
eval-vlm config set media_root /root/autodl-tmp/capt/code/LlamaFactory   # 图片根
eval-vlm config show                              # 查看当前全局配置
```

### 快速开始(用内置 fixture 跑通,不联网)

```bash
# 1) 初始化数据集文件夹(--dataset = 源 JSON;--train/--test 设置比例)
eval-vlm split --dataset tests/fixtures/llamafactory_demo.json --train 0.6 --test 0.4
# -> 在 workspace/llamafactory_demo/ 生成 config.yaml + train/test.json

# 2) 把该文件夹内 config.yaml 的 inference.backend 改成 fake(离线回显)

# 3) 一键 run + score(--dataset = 数据集名)
eval-vlm eval --dataset llamafactory_demo
# 结果在 workspace/llamafactory_demo/
```

### 评测真实模型(OpenAI 兼容服务)

```bash
# 1) 初始化数据集(可一次性传入划分参数)
eval-vlm split --dataset /root/autodl-tmp/capt/data/emo_v4.json --train 0.95 --test 0.05

# 2) 拿 workspace/emo_v4/train.json 去 LlamaFactory 训练;再用 vLLM 部署成 HTTP 服务:
#    vllm serve /path/to/checkpoint --served-model-name trained-vlm --port 8000
#    并把 emo_v4/config.yaml 的 inference.base_url / model 改好(或下面用命令行临时覆盖)

# 3) 一键 run + score
export OPENAI_API_KEY=EMPTY
eval-vlm eval --dataset emo_v4 --base-url http://localhost:8000/v1 --model trained-vlm
```

> **split 与 run/score 不连续**:split 之后要先训练 + 部署模型,故 `eval` 命令只连续执行 **run + score**,不含 split。

### 命令一览

| 命令 | `--dataset` 含义 | 作用 |
| --- | --- | --- |
| `config init / show / set <k> <v>` | — | 管理全局配置(workspace/media_root/image_strip_prefix) |
| `split --dataset <源JSON>` | 源数据集 JSON 路径 | **初始化**:建文件夹 + 生成 config.yaml + 分割 |
| `run --dataset <名\|路径>` | 已存在数据集 | 读 test.json → predictions.jsonl |
| `score --dataset <名\|路径>` | 已存在数据集 | 评分 → metrics/scored/failures/summary |
| `eval --dataset <名\|路径>` | 已存在数据集 | 一键 **run + score**(不含 split) |

### split / 临时覆盖参数

| 参数 | 适用命令 | 作用 |
| --- | --- | --- |
| `--train / --test / --val` | split | 划分**比例**(如 `--train 0.8 --test 0.2`) |
| `--seed` / `--stratify-by` | split | 随机种子 / 分层抽样字段 |
| `--name NAME` | split | 数据集文件夹名(默认取源文件名,不含扩展名) |
| `--force` | split | 文件夹已存在时重建(覆盖 config.yaml + 重新分割) |
| `--train-out / --val-out / --test-out` | split | 把对应产物直接写到任意目录(如 LlamaFactory `data/`) |
| `--base-url / --model` | run / eval | 临时覆盖部署地址 / 模型名(不回写 config.yaml) |
| `--scorer` | score / eval | 临时覆盖评分器 |
| `--workspace DIR` | split/run/score/eval | 临时覆盖全局 workspace |

## 配置说明

配置分两层:

- **全局配置** `~/.eval_vlm/config.yaml`(机器级,一次性):`workspace` / `media_root` / `image_strip_prefix`。用 `eval-vlm config set <k> <v>` 修改。
- **数据集配置** `<workspace>/<名>/config.yaml`(每个数据集一份,split 时从内置模板生成):下表各段。`configs/example.yaml` 是同结构的**可读参考**。

关键段:

| 段 | 作用 |
| --- | --- |
| `data` | 数据源路径、图片根目录、LlamaFactory 字段映射(对齐 `dataset_info.json`) |
| `split` | `ratio` 或 `count`、`seed`(确定性)、`stratify_by`(分层抽样) |
| `inference` | 后端、`base_url`、`model`、并发、重试、超时等 |
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

## 输出产物(`<workspace>/<数据集名>/`)

| 文件 | 内容 |
| --- | --- |
| `config.yaml` | 该数据集的自包含配置(split 时从内置模板生成;run/score/eval 直接读它) |
| `train.json` / `test.json` / `val.json` | 划分出的数据集,**均为原样 LlamaFactory 格式**(val 仅 `val>0` 时产出) |
| `split_meta.json` | 划分元信息(seed/比例/计数/源哈希/原始下标),用于复现与审计 |
| `predictions.jsonl` | 每行一条预测(id/**turn**/prediction/**images**/latency/error),`images` 为原图地址可追溯回原图人工核查;多轮下每样本多行,**追加写,支持断点续跑** |
| `metrics.json` | 聚合指标(含 `per_turn` 逐轮分组指标 + `overall_mean_score` + `num_failures`) |
| `scored.jsonl` | 逐(样本,轮)得分(id/turn/ordinal/scorer/score/**images**/...) |
| `failures.jsonl` | **exact_match 未命中清单(含缺失/报错),每条带原图地址,供人工审核** |
| `summary.md` | 人类可读摘要(含未命中条数) |
| `run_meta.json` | 运行元信息(模型/配置/时间/计数),用于复现 |

**断点续跑**:`run` 会跳过 `predictions.jsonl` 中已成功的 id,只补缺失/失败的——大测试集中断后重跑无需从头。

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
│   ├── base.py          # 后端接口
│   ├── openai_backend.py# OpenAI 兼容调用(图片 base64、并发、重试)
│   └── fake_backend.py  # 离线回显后端(测试/演示)
├── scoring/
│   ├── base.py / registry.py / exact_match.py / token_f1.py   # 可插拔评分
├── results/store.py     # 产物读写(断点续跑)
├── runner.py            # 执行测试(并发编排)
├── evaluate.py          # 评分编排
├── workspace.py         # 工作目录模型:全局配置 + 数据集初始化/定位 + 模板渲染
├── templates/
│   └── config.template.yaml   # 内置数据集配置模板(split 时渲染)
└── cli.py               # config / split / run / score / eval
```
