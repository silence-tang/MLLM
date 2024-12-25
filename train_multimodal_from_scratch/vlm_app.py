import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from train import VLMConfig, VLM
import torch
from torch.nn import functional as F


device = "cuda:0"
processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-224', cache_dir="/data/vdc/tangzichen/siglip")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir="/data/vdc/tangzichen/qwen2.5")

AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)
pretrain_model = AutoModelForCausalLM.from_pretrained('/data/vdc/tangzichen/MLLM/train_multimodal_from_scratch/save/pretrained')
pretrain_model.to(device)
pretrain_model.eval()

sft_model = AutoModelForCausalLM.from_pretrained('/data/vdc/tangzichen/MLLM/train_multimodal_from_scratch/save/sft')
sft_model.to(device)
sft_model.eval()


def generate(mode, image_input, text_input, max_new_tokens=100, temperature=0.0, top_k=None):
    # 选择模型
    if mode == 'pretrain':
        model = pretrain_model
    else:
        model = sft_model

    # 根据用户上传的图片和输入的文本构造input_ids
    q_text = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": 'You are a helpful assistant.'
            }, 
            {
                "role": "user",
                "content":f'{text_input}\n<image>'
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    ).replace('<image>', '<|image_pad|>'*49)

    input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
    input_ids = input_ids.to(device)
    # image = Image.open(image_input).convert("RGB")
    pixel_values = processor(text=None, images=image_input)['pixel_values']
    # pixel_values = processor(text=None, images=image_input).pixel_values
    pixel_values = pixel_values.to(device)
    eos = tokenizer.eos_token_id
    s = input_ids.shape[1]

    # 只要当前序列的长度小于原输入序列长度+允许输出的最长序列长度, 就一直不断预测next token
    while input_ids.shape[1] < s + max_new_tokens - 1:
        inference_res = model(input_ids, pixel_values, None)
        logits = inference_res.logits
        # 取出当前序列最后一个token输出的logits(表示预测出来的下一个token的logits)
        logits = logits[:, -1, :]

        for token in set(input_ids.tolist()[0]):
            logits[:, token] /= 1.0

        # 当温度系数为0时, prob由最大值主导, 其他位置的值趋于0, 因此此时只要选取top1所在位置作为结果的token_ids即可
        if temperature == 0.0:
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        # 当温度系数不为0时, 要根据带有温度系数的softmax公式(其实就是logits先除以tau然后取softmax)来计算最终输出的prob然后选取
        else:
            logits = logits / temperature
            # 如果启用topk解码策略, 则先把当前logit值小于第k大的logit值的所有位置mask掉, 然后再取softmax
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            # 从概率分布中随机采样1个样本
            idx_next = torch.multinomial(probs, num_samples=1, generator=None)

        if idx_next == eos:
            break

        # 将当前预测出来的next token拼接到input_ids最后, 随后进入下一个token的预测
        input_ids = torch.cat((input_ids, idx_next), dim=1)

    return tokenizer.decode(input_ids[:, s:][0])


with gr.Blocks() as demo:
    with gr.Row():
        # 上传图片
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="选择图片")
        with gr.Column(scale=1):
            mode = gr.Radio(["pretrained", "sft"], label="选择模型")
            text_input = gr.Textbox(label="输入文本")
            text_output = gr.Textbox(label="输出文本")
            generate_button = gr.Button("生成")
            generate_button.click(generate, inputs=[mode, image_input, text_input], outputs=text_output)
            

if __name__ == "__main__":

    demo.launch(share=False, server_name="0.0.0.0", server_port=8008)
    
