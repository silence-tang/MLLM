import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from typing import List, Dict, Any

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self,
                 llm_model_name='Qwen/Qwen2.5-0.5B-Instruct',
                 vision_model_name='google/siglip-base-patch16-224',
                 freeze_vision_model=True,
                 image_pad_num=49,
                 **kwargs):
        # 调用super()__init__()对父类的属性和方法进行初始化, 这样子类的实例可以调用父类的属性和方法
        super().__init__(**kwargs)
        self.vision_model_name = vision_model_name
        self.llm_model_name = llm_model_name
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        
        
class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):

        super().__init__(config)
        self.config = config

        # load models (vision encoder, tokenizer, vl connector, llm)
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_name, cache_dir="/data/vdc/tangzichen/siglip")
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_name, cache_dir="/data/vdc/tangzichen/qwen2.5")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name, cache_dir="/data/vdc/tangzichen/qwen2.5")

        # define vl connector
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*4, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        
        # ce loss func, 设置ignore_index让模型在计算loss时只计算answer(且不带padding的)部分
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # freeze parameters
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        for param in self.llm_model.parameters():
            param.requires_grad = False
        
    # 处理一个batch的数据
    def forward(self, input_ids, pixel_values, labels, attention_mask=None):
        # get text embeddings
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)  # [b, n_text_tokens, dim_text]
        # get image embeddings
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        # batch_size, token_length, image_token_dim
        b, n, d = image_embeds.shape
        # 由于image emb数量明显多于text emb,会削弱模型对的文本的处理能力, 因此这边选择对iamge emb的长度进行压缩
        image_embeds = image_embeds.view(b, -1, d * 4)  # [b, 196, 768] -> [b, 49, 3072]
        # 经过两层linear, 与llm的text space对齐
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))  # [b, 49, 896]
        text_embeds = text_embeds.to(image_features.dtype)
        # 合并text emb与image emb
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        
        # 执行llm的forward前向推理输出预测结果
        # NOTE: 这边用的是qwen的基础模型, 因此model.forward返回的仅仅是最后输出的hidden_states, 而不会帮你计算loss
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]  # [b, sequence_length, config.vocab_size]不是softmax的结果
        loss = None
        if labels is not None:
            # NOTE 1: loss内部会将logits转为softmax
            # NOTE 2: 虽然labels里的值是整数ids, 但loss内部会根据token的ids自动取出该ids位置上的logits值计算负对数似然-1*log(p_pred)
            # 只计算p_gt=1的位置已经足够了, 若其他p_gt=0的位置再计算loss(如l_i=lambda_i*p_pred_i)则会极大地影响效率, 且收效甚微
            loss = self.loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)
        
    # 将text emb中与image占位符对应的部分替换为真正的image emb (49处)
    def merge_input_ids_with_image_features(self, image_features, text_embeds, input_ids):
        num_images, num_image_embeds, embed_dim = image_features.shape
        # 找到该batch中每个样本的text embedding需要进行替换的位置
        batch_indices, sentence_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        text_embeds[batch_indices, sentence_indices] = image_features.view(-1, embed_dim)
        return text_embeds
    

class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        # 训练数据的meta data(图像名称, 对话内容等)
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)
            
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']  # 图像名称
            conversations = sample['conversations']  # 多轮对话的内容, 但预训练我们只取首轮对话来训练
            # 把输入改成qwen需要的固定模版, 相当于做一个小小的prompt工程, 让模型更好地适应对话场景
            q_text = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": 'You are a helpful assistant.'
                    },
                    {
                        "role": "user",
                        "content": conversations[0]['value']  # 首轮对话user的问题
                    }
                ],
                tokenize=False,
                add_generation_prompt=True) \
            .replace('<image>', '<|image_pad|>' * self.config.image_pad_num)  # 把原始数据中的image标识符换成49个指定的占位符
            # 首轮对话llm的回答+eos token
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            # 获取问题和回答的input_ids
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            # 总的input_ids
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
            # 读取图像
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            # 用self.processor对输入图像进行格式预处理, 获取pixel_values
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        # 若出现某条数据有损, 则用空白图片+描述来补充
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system",
                     "content": 'You are a helpful assistant.'
                    },
                     {"role": "user",
                      "content": "图片内容是什么\n<image>"
                    }
                ],
                tokenize=False,
                add_generation_prompt=True) \
            .replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
            # 描述空白图像
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'labels': labels
        }  # 返回值是字典


# collator负责在训练过程中动态地对每个batch的数据进行处理(如padding)
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    # 在实际训练过程中, 处理到某个batch时会调用collator
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 取到当前batch中所有input的最大长度
        max_len = max(len(example['input_ids']) for example in batch)
        input_ids = []
        pixel_values = []
        labels = []
        # 遍历当前batch中的每个样本
        for example in batch:
            # 向右padding直到长度=max_len
            input_ids.append(example['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(example['input_ids'])))
            pixel_values.append(example['pixel_values'])
            labels.append(example['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(example['labels'])))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
            
        
if __name__ == '__main__':
    
    # get mllm
    config = VLMConfig(llm_model_name='Qwen/Qwen2.5-0.5B-Instruct', vision_model_name='google/siglip-base-patch16-224', image_pad_num=49)
    
    
    # get dataset and collator
    images_path = '/data/vdc/tangzichen/liuhaotian/LLaVA-CC3M-Pretrain-595K/images'
    data_path = '/data/vdc/tangzichen/LinkSoul/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name, cache_dir="/data/vdc/tangzichen/qwen2.5")
    processor = AutoProcessor.from_pretrained(config.vision_model_name, cache_dir="/data/vdc/tangzichen/siglip")
    train_dataset = MyDataset(images_path, data_path, tokenizer, processor, config)
    train_collator = MyDataCollator(tokenizer)

    # out dir
    output_dir = 'save/pretrained_0116'

    # train args
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=4,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        deepspeed="/data/vdc/tangzichen/MLLM/train_multimodal_from_scratch/deepspeed_config_stage1.json"
    )

    # 若使用zero-init, 则模型加载必须在args之后
    model = VLM(config).cuda()
    # print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=train_collator  
    )
    
    # train!
    trainer.train(resume_from_checkpoint=False)

    # save model
    trainer.save_model(output_dir=output_dir)
    trainer.save_state()

    # ds: 0.16.2
