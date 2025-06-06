# -*- coding: utf-8 -*-
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    BitsAndBytesConfig,
)
import copy


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-2B-sft-bf16")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/AdvertiseGenChatML/train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="data/AdvertiseGenChatML/dev.json",
        metadata={"help": "Path to the test data."},
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    qlora: bool = field(default=False)
    # DPO相关参数
    use_dpo: bool = field(default=False, metadata={"help": "Whether to use DPO training"})
    dpo_beta: float = field(default=0.1, metadata={"help": "Beta parameter for DPO loss"})
    reference_model_path: Optional[str] = field(default=None, metadata={"help": "Path to reference model for DPO"})
    # SFT损失权重参数
    sft_loss_weight: float = field(default=0.0, metadata={"help": "Weight for SFT loss when combined with DPO"})


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length=4096,
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue
            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = [self.tokenizer.bos_token_id]
        label_ids = [self.ignore_index]

        for message in example["messages"]:
            role = message["role"]
            content = message["content"]

            content_ids = self.tokenizer.apply_chat_template([message])

            if role == "user":
                if self.tokenizer.eos_token_id == 73440:  # minicpm3.0 and minicpm4.0
                    input_ids += self.tokenizer.apply_chat_template(
                        [message], add_generation_prompt=True
                    )
                    label_ids += [self.ignore_index] * len(
                        self.tokenizer.apply_chat_template(
                            [message], add_generation_prompt=True
                        )
                    )
                else: # minicpm2.0
                    input_ids += content_ids
                    label_ids += [self.ignore_index] * len(content_ids)
            elif role == "system":
                input_ids += content_ids
                label_ids += [self.ignore_index] * len(content_ids)
            elif role == "assistant":
                if self.tokenizer.eos_token_id == 73440:  # minicpm3.0 and minicpm4.0
                    input_ids += self.tokenizer.encode(content, add_special_tokens=False)
                    label_ids += self.tokenizer.encode(content, add_special_tokens=False)
                else: # minicpm2.0
                    input_ids += content_ids
                    label_ids += content_ids

        input_ids.append(self.tokenizer.eos_token_id)
        label_ids.append(self.tokenizer.eos_token_id)
        # truncate to max len
        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]
        attention_mask = [1] * len(input_ids)
        # pad to max len
        input_ids += [self.tokenizer.eos_token_id] * (
            self.model_max_length - len(input_ids)
        )
        label_ids += [self.ignore_index] * (self.model_max_length - len(label_ids))
        attention_mask += [0] * (self.model_max_length - len(attention_mask))
        # convert to pt tensor
        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        attention_mask = torch.LongTensor(attention_mask)
        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


class DPODataset(Dataset):
    """Dataset for DPO training with optional SFT data."""
    
    def __init__(self, data_path, tokenizer, model_max_length=4096, include_sft_data=False):
        super(DPODataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        self.include_sft_data = include_sft_data
        
        # 展示第一个样本的处理结果
        if len(self.data) > 0:
            item = self.preprocessing(self.data[0])
            print("DPO Dataset Sample:")
            print("Chosen input:", self.tokenizer.decode(item["chosen_input_ids"], skip_special_tokens=True))
            print("Rejected input:", self.tokenizer.decode(item["rejected_input_ids"], skip_special_tokens=True))
    
    def __len__(self):
        return len(self.data)
    
    def build_conversation(self, instruction, input_text="", history=None):
        """构建对话格式"""
        messages = []
        
        # 添加历史对话
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # 添加当前指令
        current_input = instruction
        if input_text:
            current_input = f"{instruction}\n{input_text}"
        messages.append({"role": "user", "content": current_input})
        
        return messages
    
    def encode_conversation(self, messages, response):
        """编码对话和回复"""
        # 构建完整对话
        full_messages = messages + [{"role": "assistant", "content": response}]
        
        # 使用chat template编码
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_ids = self.tokenizer.apply_chat_template(
                full_messages, 
                tokenize=True, 
                add_generation_prompt=False,
                return_tensors="pt"
            ).squeeze(0)
        else:
            # 如果没有chat template，使用简单拼接
            text = ""
            for msg in full_messages:
                if msg["role"] == "user":
                    text += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    text += f"Assistant: {msg['content']}\n"
            input_ids = self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
        
        # 截断到最大长度
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[:self.model_max_length]
        
        # 计算attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # 填充到固定长度
        if len(input_ids) < self.model_max_length:
            pad_length = self.model_max_length - len(input_ids)
            input_ids = torch.cat([
                input_ids, 
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=attention_mask.dtype)
            ])
        
        return input_ids, attention_mask
    
    def encode_conversation_with_labels(self, messages, response):
        """编码对话和回复，同时生成SFT训练所需的labels"""
        # 构建完整对话
        full_messages = messages + [{"role": "assistant", "content": response}]
        
        # 使用chat template编码
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_ids = self.tokenizer.apply_chat_template(
                full_messages, 
                tokenize=True, 
                add_generation_prompt=False,
                return_tensors="pt"
            ).squeeze(0)
        else:
            # 如果没有chat template，使用简单拼接
            text = ""
            for msg in full_messages:
                if msg["role"] == "user":
                    text += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    text += f"Assistant: {msg['content']}\n"
            input_ids = self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
        
        # 创建labels，只对assistant回复部分计算损失
        labels = input_ids.clone()
        
        # 编码不包含assistant回复的部分，用于确定哪些token需要ignore
        prompt_messages = messages
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors="pt"
            ).squeeze(0)
        else:
            prompt_text = ""
            for msg in prompt_messages:
                if msg["role"] == "user":
                    prompt_text += f"User: {msg['content']}\n"
            prompt_text += "Assistant: "
            prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").squeeze(0)
        
        # 对prompt部分设置ignore_index
        prompt_len = len(prompt_ids)
        if prompt_len < len(labels):
            labels[:prompt_len] = self.ignore_index
        
        # 截断到最大长度
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[:self.model_max_length]
            labels = labels[:self.model_max_length]
        
        # 计算attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # 填充到固定长度
        if len(input_ids) < self.model_max_length:
            pad_length = self.model_max_length - len(input_ids)
            input_ids = torch.cat([
                input_ids, 
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_length,), self.ignore_index, dtype=labels.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=attention_mask.dtype)
            ])
        
        return input_ids, attention_mask, labels
    
    def preprocessing(self, example):
        """预处理DPO数据样本"""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        chosen = example["chosen"]
        rejected = example["rejected"]
        history = example.get("history", [])
        
        # 构建对话消息
        messages = self.build_conversation(instruction, input_text, history)
        
        # 编码chosen和rejected回复
        chosen_input_ids, chosen_attention_mask = self.encode_conversation(messages, chosen)
        rejected_input_ids, rejected_attention_mask = self.encode_conversation(messages, rejected)
        
        result = {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }
        
        # 只有在需要SFT损失时才生成相关数据
        if self.include_sft_data:
            chosen_input_ids_sft, chosen_attention_mask_sft, chosen_labels = self.encode_conversation_with_labels(messages, chosen)
            result.update({
                "chosen_input_ids_sft": chosen_input_ids_sft,
                "chosen_attention_mask_sft": chosen_attention_mask_sft,
                "chosen_labels": chosen_labels,
            })
        
        return result
    
    def __getitem__(self, idx):
        return self.preprocessing(self.data[idx])


class DPODataCollator:
    """自定义的DPO数据collator，处理特殊的DPO数据格式"""
    
    def __init__(self, tokenizer, include_sft_data=False):
        self.tokenizer = tokenizer
        self.include_sft_data = include_sft_data
    
    def __call__(self, features):
        batch = {}
        
        # 处理基本的DPO字段
        dpo_keys = ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]
        
        for key in dpo_keys:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])
        
        # 如果包含SFT数据，也处理SFT相关字段
        if self.include_sft_data:
            sft_keys = ["chosen_input_ids_sft", "chosen_attention_mask_sft", "chosen_labels"]
            for key in sft_keys:
                if key in features[0]:
                    batch[key] = torch.stack([f[key] for f in features])
        
        return batch


class DPOTrainer(Trainer):
    """Custom Trainer for DPO with optional SFT loss."""
    
    def __init__(self, reference_model=None, dpo_beta=0.1, sft_loss_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        self.reference_model = reference_model
        self.dpo_beta = dpo_beta
        self.sft_loss_weight = sft_loss_weight
        self.use_sft = sft_loss_weight > 0
        
        # 将参考模型移动到正确的设备
        if self.reference_model is not None:
            self.reference_model.to(self.args.device)
            self.reference_model.eval()
            # 确保参考模型不需要梯度
            for param in self.reference_model.parameters():
                param.requires_grad = False
    
    def get_log_probabilities(self, model, input_ids, attention_mask):
        """计算序列的log概率"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 计算每个token的log概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取实际token的log概率
        # shift操作：预测下一个token
        shift_log_probs = log_probs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention = attention_mask[..., 1:].contiguous()
        
        # 收集每个位置的log概率
        gathered_log_probs = torch.gather(
            shift_log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 只计算非padding部分的平均log概率
        masked_log_probs = gathered_log_probs * shift_attention.float()
        sequence_log_prob = masked_log_probs.sum(dim=-1) / (shift_attention.sum(dim=-1).float() + 1e-8)
        
        return sequence_log_prob
    
    def compute_dpo_loss(self, policy_chosen_logps, policy_rejected_logps, 
                         reference_chosen_logps, reference_rejected_logps):
        """计算DPO损失函数"""
        # 计算相对于参考模型的log概率比值
        policy_ratio_chosen = policy_chosen_logps - reference_chosen_logps
        policy_ratio_rejected = policy_rejected_logps - reference_rejected_logps
        
        # DPO损失
        logits = self.dpo_beta * (policy_ratio_chosen - policy_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()
        
        # 计算准确率（chosen概率高于rejected的比例）
        accuracy = (policy_ratio_chosen > policy_ratio_rejected).float().mean()
        
        return loss, accuracy
    
    def compute_sft_loss(self, logits, labels):
        """计算SFT损失"""
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = loss_fct(shift_logits, shift_labels)
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算DPO损失和可选的SFT损失，并分别输出"""
        # 计算策略模型的log概率（用于DPO）
        policy_chosen_logps = self.get_log_probabilities(
            model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"]
        )
        policy_rejected_logps = self.get_log_probabilities(
            model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"]
        )
        
        # 计算参考模型的log概率（用于DPO）
        with torch.no_grad():
            reference_chosen_logps = self.get_log_probabilities(
                self.reference_model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"]
            )
            reference_rejected_logps = self.get_log_probabilities(
                self.reference_model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"]
            )
        
        # 计算DPO损失
        dpo_loss, accuracy = self.compute_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        # 初始化总损失为DPO损失
        total_loss = dpo_loss
        
        # 准备日志字典
        log_dict = {
            "dpo_loss": dpo_loss.item(), 
            "dpo_accuracy": accuracy.item()
        }
        
        # 计算SFT损失（如果启用且数据可用）
        sft_loss = None
        if self.use_sft and "chosen_labels" in inputs:
            # 使用chosen回复计算SFT损失
            outputs = model(
                input_ids=inputs["chosen_input_ids_sft"], 
                attention_mask=inputs["chosen_attention_mask_sft"]
            )
            sft_loss = self.compute_sft_loss(outputs.logits, inputs["chosen_labels"])
            
            # 将SFT损失加入总损失
            total_loss = total_loss + self.sft_loss_weight * sft_loss
            
            # 添加SFT损失到日志
            log_dict.update({
                "sft_loss": sft_loss.item(),
                "sft_loss_weight": self.sft_loss_weight,
                "total_loss": total_loss.item()
            })
        else:
            # 如果没有SFT损失，总损失就是DPO损失
            log_dict["total_loss"] = total_loss.item()
        
        # 记录所有指标
        self.log(log_dict)
        
        return (total_loss, None) if return_outputs else total_loss


def load_model_and_tokenizer(
    model_path: str,
    max_length: int = 4096,
    use_lora: bool = True,
    qlora: bool = False,
    bf16: bool = False,
    fp16: bool = False,
):
    """load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    if qlora:
        assert use_lora, "use_lora must be True when use_qlora is True"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 是否进行4bit量化
            load_in_8bit=False,  # 是否进行8bit量化
            bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
            bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
            bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正太分布的int4
            bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
            llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
            llm_int8_has_fp16_weight=False,  # 是否启用混合精度
            # llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
            llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
            target_modules=(
                ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj"]
                if hasattr(model.config, 'architectures') and model.config.architectures == ["MiniCPM3ForCausalLM"]
                else ["q_proj", "v_proj"]
            ),
            r=64,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer


def load_reference_model(model_path, bf16=False, fp16=False):
    """加载参考模型（用于DPO训练）"""
    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    # 参考模型不需要梯度
    for param in reference_model.parameters():
        param.requires_grad = False
    
    return reference_model


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        use_lora=training_args.use_lora,
        qlora=training_args.qlora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
    )

    if training_args.use_dpo:
        # 如果没有指定参考模型路径，则使用当前模型作为参考模型
        reference_model_path = training_args.reference_model_path or model_args.model_name_or_path
        reference_model = load_reference_model(
            model_path=reference_model_path,
            bf16=training_args.bf16,
            fp16=training_args.fp16,
        )
        
        train_dataset = DPODataset(
            data_path=data_args.train_data_path,
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length,
            include_sft_data=training_args.sft_loss_weight > 0,
        )
        eval_dataset = DPODataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length,
            include_sft_data=training_args.sft_loss_weight > 0,
        ) if os.path.exists(data_args.eval_data_path) else None
        
        # 创建自定义数据collator
        data_collator = DPODataCollator(
            tokenizer=tokenizer,
            include_sft_data=training_args.sft_loss_weight > 0
        )
        
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,  # 使用自定义data collator
            reference_model=reference_model,
            dpo_beta=training_args.dpo_beta,
            sft_loss_weight=training_args.sft_loss_weight,
        )
    else:
        train_dataset = SupervisedDataset(
            data_path=data_args.train_data_path,
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length,
        )
        eval_dataset = SupervisedDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length,
        ) if os.path.exists(data_args.eval_data_path) else None

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

    trainer.train()
    
    # save the incremental PEFT weights, more details can be found in https://huggingface.co/blog/peft
    trainer.save_model()