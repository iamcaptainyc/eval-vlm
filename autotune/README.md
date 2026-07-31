# autotune — LlamaFactory × eval-vlm 自动超参搜索

用 **Optuna** 搜索 Qwen3.5-VL 的 LoRA 超参。每个 trial 自动完成:

```
采样超参 → llamafactory-cli train → llamafactory-cli export(合并) →
eval-vlm field-eval(自给自足:无预测则用离线 vLLM 自动 pred,再抽字段逐字段比对)→ 读 run_dir 里的 json 指标 → 回传 Optuna
```

单卡单机**串行**;训练与评测用**不同 conda 环境**;所有路径/超参在 `config.yaml` 里配置。

## 前置

- 一台带 GPU 的服务器(整个闭环在服务器上跑)。
- 两个 conda 环境:
  - `train_env`:装了 **LlamaFactory**(`llamafactory-cli` 可用)。
  - `eval_env`:装了 **vllm** 和 **eval-vlm**(`eval-vlm` 可用,且含 `vllm_offline` 后端)。
- 跑本驱动的环境装 `optuna` 和 `pyyaml`(可以就用 `eval_env`,也可单独环境)。
- **value-extract 服务在线**:`field_eval: true` 时,eval-vlm field-eval 会调它抽字段(见 eval-vlm 数据集 config 里的 `label_extract`)。
- eval-vlm 数据集文件夹已就绪(含 `config.yaml` + `test.json`),且其 `eval.targets` 配置会预测到**第一轮描述**那一轮(field-eval 评的是描述)。

## 用法

```bash
cp config.example.yaml config.yaml   # 按注释填好服务器上的真实路径/环境名/搜索空间
conda run -n <driver_env> python tune.py --config config.yaml
```

- **断点续跑**:重跑同一命令即可。`n_trials` 是**总预算**——已终结(完成/剪枝/失败)的 trial 都计入,本次只补齐到总数(不是每次都新跑 `n_trials` 个)。study 持久化在 `<work_root>/study.db`。要多做试验就调大 `n_trials`。
- **临时改 trial 数**:`--n-trials 5`。
- **看结果**:`<work_root>/leaderboard.csv`(每 trial 一行:目标值、逐字段准确率、超参、合并权重路径);跑完终端打印最优 trial 的超参与逐字段准确率、最优合并权重目录。

## 每个 trial 的产物

```
<work_root>/
  <模型名>_trial_0007/           # 合并后全精度权重(唯一命名 = 模型名_trial号,eval-vlm 加载它)
  <模型名>_trial_0007.merge.yaml # 本 trial 的合并配置(留存,可复现)
```
训练用的 LoRA adapter 是**临时目录**(`.<模型名>_trial_0007.adapter/`),合并成功后自动删除——只保留合并权重,省磁盘。

评测结果在 eval-vlm 侧:`<eval_vlm_dataset>/<模型名>_trial_0007/vllm_offline/field_metrics.json`。合并目录名带模型信息且每 trial 唯一 → **run_dir 互不覆盖**;`result_name` 也因此含模型名。ref 字段抽取结果缓存在数据集级(`<eval_vlm_dataset>/fields_ref.jsonl`),**跨 trial 只抽一次**。

## 配置要点(详见 config.example.yaml 注释)

- `base_train_args`:照抄你的 `llama_train.sh` 参数(不搜索的都放这);`model_name_or_path` / `output_dir` 由驱动器自动设,不用写。
- `search_space`:键就是 llamafactory 的参数名(会覆盖 `base_train_args` 同名项)。类型 `float_log`/`float`/`int`/`categorical`。
- `objective.metric`:`field_exact_match`(全对率)/ `field_micro`(逐字段总体)/ `field_macro`;若 `eval.field_eval: false` 则用 `mean_score`(走 eval-vlm score)。
- `eval.image_max_pixels`:离线 vLLM 的视觉像素上限,**建议与训练 `image_max_pixels` 对齐**(否则视觉 token 分布偏离训练)。

## 常见问题

- **显存**:训练与离线 vLLM 都吃满卡,但各步是独立子进程、结束即释放,串行天然错开。若 export/eval OOM,调低 `eval.gpu_memory_utilization`。
- **某 trial 训练/合并/评测报错**:该 trial 被标记为 pruned(错误记进 `trial.user_attrs['error']`),不影响其余 trial;修好后重跑续跑即可。
- **多保真度提速**(可选):把 `search_space` 里 `num_train_epochs` 调小、或先用小 `n_trials` 粗筛,再对好区间细搜。
