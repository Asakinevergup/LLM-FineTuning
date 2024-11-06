import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 基础模型路径
BASE_MODEL_PATH = "../chatglm3-6b"

# 加载分词器和基础模型
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_model = AutoModel.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True).to("cuda").eval()

# LoRA 权重路径列表（放置多个 LoRA 权重文件夹以供选择）
LORA_WEIGHTS_PATHS = {
    "仅基础模型": None,
    "LoRA 权重 1": "./output-glm3/epoch-2-step-720",
    "KidMedicalLoRA 权重 2": "./output-KidMedicalGLM3/epoch-4-step-7020",
    "LawLoRA 权重 3": "./output-LawGLM3/epoch-5-step-615/",

    # 可以添加更多的 LoRA 权重路径
}


# 函数：根据选择加载对应的模型
def load_model(lora_choice):
    if LORA_WEIGHTS_PATHS[lora_choice] is None:
        return base_model
    else:
        lora_model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATHS[lora_choice],
                                               torch_dtype=torch.float16).to("cuda")
        return lora_model


# 函数：执行模型推理
def generate_response(instruction, input_text, lora_choice):
    model = load_model(lora_choice)
    model.eval()
    combined_input = f"{instruction}\n\n{input_text}"  # 将指令和输入文本组合
    with torch.no_grad():
        response, _ = model.chat(tokenizer, combined_input, history=[])
    return response


# 自定义 CSS 样式
css = """
.container {
    background-color: #1f2937;
    color: #d1d5db;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
}

#header {
    text-align: center;
    background-color: #3b82f6;
    padding: 15px;
    color: white;
    border-radius: 8px;
    margin-bottom: 20px;
}

#left-container, #right-container {
    padding: 15px;
    border-radius: 8px;
}

#left-container {
    background-color: #374151;
    border: 1px solid #4b5563;
}

#right-container {
    background-color: #4b5563;
    border: 1px solid #6b7280;
}

#generate-button {
    background-color: #3b82f6;
    color: white;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 16px;
}

#generate-button:hover {
    background-color: #2563eb;
}
"""

# 创建 Gradio 界面
with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        <div id="header">
            <h1>ChatGLM with LoRA Web Interface</h1>
            <p>选择不同的 LoRA 权重或仅使用基础模型来生成回答。</p>
        </div>
        """
    )

    with gr.Row():
        # 左侧容器：Instruction + 选择 LoRA 权重 + 输入框
        with gr.Column(elem_id="left-container"):
            instruction = gr.Textbox(
                label="Instruction（指导）",
                placeholder="输入指令，如：'使用正式语言回答问题' 或 '简洁地解释问题'...",
                lines=2
            )
            lora_choice = gr.Dropdown(
                choices=list(LORA_WEIGHTS_PATHS.keys()),
                value="仅基础模型",
                label="选择 LoRA 权重",
                interactive=True
            )
            input_text = gr.Textbox(
                label="输入您的问题",
                placeholder="请输入您的问题或文本内容...",
                lines=8
            )
            generate_button = gr.Button("生成回复", elem_id="generate-button")

        # 右侧容器：输出框
        with gr.Column(elem_id="right-container"):
            output_text = gr.Textbox(
                label="模型回复",
                interactive=False,
                lines=15
            )

    # 将按钮绑定到生成函数
    generate_button.click(
        fn=generate_response,
        inputs=[instruction, input_text, lora_choice],
        outputs=output_text
    )

# 启动 Gradio 应用
demo.launch(share=True)

