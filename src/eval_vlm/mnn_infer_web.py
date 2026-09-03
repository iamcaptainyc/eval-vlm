import gradio as gr
import argparse
import os

from eval_vlm.cli import _cmd_infer

MNN_MODEL_ROOT = "/root/autodl-tmp/models_mnn"
MNN_CONFIG = "/root/autodl-tmp/models_mnn/train_2026-08-03-10-12-58_Qwen3_5_0_8B_Instruct_all/config.json"

def get_mnn_configs():

    models = []

    if not os.path.exists(MNN_MODEL_ROOT):
        return models

    for name in os.listdir(MNN_MODEL_ROOT):

        model_dir = os.path.join(
            MNN_MODEL_ROOT,
            name
        )

        config = os.path.join(
            model_dir,
            "config.json"
        )

        if os.path.isdir(model_dir) and os.path.exists(config):
            models.append(name)

    return sorted(models)

def refresh_models():
    models = get_mnn_configs()

    return gr.update(
        choices=models,
        value=models[0] if models else None
    )

def infer(
    image,
    question,
    mnn_model,
    image_max_pixels,
    max_tokens,
    temperature,
    top_k,
    top_p
):

    if image is None:
        return "请上传图片", ""

    if not question:
        return "请输入问题", ""

    config_path = os.path.join(
        MNN_MODEL_ROOT,
        mnn_model,
        "config.json"
    )

    args = argparse.Namespace(
        mnn_config=config_path,

        img_path=image,
        prompt=question,

        max_pixels=image_max_pixels,
        min_pixels=None,
        system_prompt=None,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


    try:

        pred = _cmd_infer(args)


    except Exception as e:

        return f"ERROR:\n{e}", ""


    # 获取回答
    answer = str(pred.prediction)


    print(pred.raw)
    # 获取性能数据

    raw = getattr(pred, "raw", {})

    info = f"""
<div style="
    border:1px solid #ddd;
    border-radius:10px;
    padding:15px;
    background:#fafafa;
    font-family:Arial, sans-serif;
">

<h3 style="margin-top:0;">
🚀 推理性能 / Inference Performance
</h3>


<table style="
    width:100%;
    border-collapse:collapse;
    font-size:14px;
">

<tr>
<td colspan="2" style="font-weight:bold;padding:8px 0;">
⚙️ 模型信息 / Model Information
</td>
</tr>

<tr>
<td style="padding:6px;">Backend / 后端</td>
<td>{raw.get('backend','N/A')}</td>
</tr>

<tr>
<td style="padding:6px;">Prompt Tokens / 输入Token</td>
<td>{raw.get('prompt_token_count','N/A')}</td>
</tr>

<tr>
<td style="padding:6px;">Generated Tokens / 输出Token</td>
<td>{raw.get('gen_seq_len','N/A')}</td>
</tr>


<tr>
<td colspan="2" style="font-weight:bold;padding:10px 0 6px;">
⏱️ 延迟 / Latency
</td>
</tr>

<tr>
<td style="padding:6px;">Vision Encoder / 视觉编码</td>
<td>{raw.get('vision_us',0)/1000:.2f} ms</td>
</tr>

<tr>
<td style="padding:6px;">Prefill</td>
<td>{raw.get('prefill_us',0)/1000:.2f} ms</td>
</tr>

<tr>
<td style="padding:6px;">Decode</td>
<td>{raw.get('decode_us',0)/1000:.2f} ms</td>
</tr>

<tr>
<td style="padding:6px;">TTFT / 首Token时间</td>
<td>{raw.get('ttft_ms','N/A')} ms</td>
</tr>

<tr>
<td style="padding:6px;">TPOT / 输出token间隔</td>
<td>{raw.get('tpot_ms','N/A')} ms</td>
</tr>

<tr>
<td style="padding:6px;">E2E / 总耗时</td>
<td>{raw.get('e2e_ms','N/A')} ms</td>
</tr>


<tr>
<td colspan="2" style="font-weight:bold;padding:10px 0 6px;">
📈 速度 / Throughput
</td>
</tr>


<tr>
<td style="padding:6px;">Prefill Speed</td>
<td>{raw.get('prefill_toks_per_s','N/A')} tok/s</td>
</tr>

<tr>
<td style="padding:6px;">Decode Speed</td>
<td>{raw.get('decode_toks_per_s','N/A')} tok/s</td>
</tr>

<tr>
<td style="padding:6px;">Total Speed</td>
<td>{raw.get('total_toks_per_s','N/A')} tok/s</td>
</tr>


<tr>
<td colspan="2" style="font-weight:bold;padding:10px 0 6px;">
🖼️ 图像 / Image
</td>
</tr>


<tr>
<td style="padding:6px;">Pixels</td>
<td>{raw.get('pixels_mp','N/A')} MP</td>
</tr>

<tr>
<td style="padding:6px;">Resized Pixels</td>
<td>{raw.get('resized_pixels','N/A')}</td>
</tr>


</table>

</div>
"""


    return answer, info





with gr.Blocks(title="MNN VLM Demo") as demo:


    gr.Markdown(
        """
        # MNN VLM 在线问答

        上传图片，然后输入问题。
        """
    )

    with gr.Row():
    
        mnn_model = gr.Dropdown(
            choices=get_mnn_configs(),
            value=(
                get_mnn_configs()[0]
                if get_mnn_configs()
                else None
            ),
            label="MNN模型",
            scale=4
        )


        refresh_model = gr.Button(
            "🔄 刷新模型",
            scale=1
        )


    with gr.Row():

        image = gr.Image(
            type="filepath",
            label="图片"
        )


        output = gr.Textbox(
            label="回答",
            lines=15
        )


    question = gr.Textbox(
        label="问题",
        value="请描述这张图片。"
    )

    

    with gr.Accordion(
        "高级参数",
        open=False
    ):


        image_max_pixels = gr.Number(
            value=1024*1024,
            label="image_max_pixels"
        )


        max_tokens = gr.Slider(
            minimum=1,
            maximum=4096,
            value=1024,
            step=1,
            label="max_tokens"
        )


        temperature = gr.Slider(
            minimum=0,
            maximum=2,
            value=0.7,
            step=0.05,
            label="temperature"
        )


        top_k = gr.Slider(
            minimum=0,
            maximum=200,
            value=20,
            step=1,
            label="top_k"
        )


        top_p = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.8,
            step=0.05,
            label="top_p"
        )


    submit = gr.Button(
        "开始问答"
    )

    # stats = gr.HTML(
    #         value="""
    #         <div style="
    #             padding:20px;
    #             color:#888;
    #             text-align:center;
    #         ">
    #         等待推理...<br>
    #         Waiting for inference...
    #         </div>
    #         """,
    #         label="推理统计"
    #     )

    with gr.Group():

        gr.Markdown(
            "### 📊 推理统计 / Performance"
        )

        stats = gr.HTML(
            value="""
            <div style="
                padding:20px;
                color:#888;
                text-align:center;
            ">
            等待推理...<br>
            Waiting for inference...
            </div>
            """
        )
    
    # =========================
    # 刷新模型列表
    # =========================

    refresh_model.click(
        fn=lambda: gr.update(
            choices=get_mnn_configs(),
            value=(
                get_mnn_configs()[0]
                if get_mnn_configs()
                else None
            )
        ),
        inputs=[],
        outputs=mnn_model
    )

    submit.click(
        fn=infer,
        inputs=[
            image,
            question,
            mnn_model,
            image_max_pixels,
            max_tokens,
            temperature,
            top_k,
            top_p
        ],
        outputs=[
            output,
            stats
        ]
    )



demo.launch(
    server_name="0.0.0.0",
    server_port=6008
)