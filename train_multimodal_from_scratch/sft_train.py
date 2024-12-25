from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from transformers import AutoProcessor
import torch
from PIL import Image
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM


def find_assistant_tokens(tokenizer, target):
    result = []
    start_index = 0
    end_index = 0

    while start_index <= len(target)-1:
        if target[start_index] != tokenizer('assistant')['input_ids'][0]:
            start_index += 1
            end_index += 1
        else:
            end_index += 1
            if target[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index + 1, end_index + 1))
                start_index = end_index + 1

    return result


class SFTDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)   
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = 'COCO_train2014_' + str(sample['image'])
            conversations = sample['conversations']
            messages = [{"role": "system", "content": 'You are a helpful assistant.'}]
            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role": "user", "content": conversation['value']})
                else:
                    messages.append({"role": "assistant", "content": conversation['value']})
            text = tokenizer.apply_chat_template(messages, tokenize=False) \
            .replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
            # print(text)
            input_ids = tokenizer(text)['input_ids']
            indexs = find_assistant_tokens(tokenizer, input_ids)
            labels = len(input_ids) * [tokenizer.pad_token_id]
            for index in indexs:
                labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
            input_ids = input_ids[:-1]
            labels = labels[1:]
            image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'}, {"role": "user", "content": "图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }   

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
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    model = AutoModelForCausalLM.from_pretrained('/data/vdc/tangzichen/train_multimodal_from_scratch/save/pretrained')
    
    # 冻结vision encoder和linear层, 只微调llm
    for name, param in model.named_parameters():
        if 'linear' in name or 'vision_model':
            param.requires_grad = False
        if 'llm_model' in name:
            param.requires_grad = True

    print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}')
    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # get tokenizer and image processor
    output_dir = 'save/sft'
    images_path = '/data/vdc/tangzichen/jingyaogong/minimind-v_dataset/sft_images'
    data_path = '/data/vdc/tangzichen/jingyaogong/minimind-v_dataset/llava_instruct_230k.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name, cache_dir="/data/vdc/tangzichen/qwen2.5")
    processor = AutoProcessor.from_pretrained(config.vision_model_name, cache_dir="/data/vdc/tangzichen/siglip")
    train_dataset = SFTDataset(images_path, data_path, tokenizer, processor, config)
    train_collator = MyDataCollator(tokenizer)

    # train args
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        num_train_epochs=2,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=train_collator 
    )
    
    # train!
    trainer.train(resume_from_checkpoint=True)

    # save model
    trainer.save_model('save/sft')
    trainer.save_state()
