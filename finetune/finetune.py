# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)


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
class TrainingArguments(transformers.TrainingArguments):
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
        for id_ in item["label_ids"]:
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
                if self.tokenizer.eos_token_id == 73440:  # minicpm3.0 is true
                    input_ids += self.tokenizer.apply_chat_template(
                        [message], add_generation_prompt=True
                    )
                    label_ids += [self.ignore_index] * len(
                        self.tokenizer.apply_chat_template(
                            [message], add_generation_prompt=True
                        )
                    )
                else:
                    input_ids += content_ids
                    label_ids += [self.ignore_index] * len(content_ids)
            elif role == "system":
                input_ids += content_ids
                label_ids += [self.ignore_index] * len(content_ids)
            elif role == "assistant":
                if self.tokenizer.eos_token_id == 73440:  # minicpm3.0 is true
                    input_ids += tokenizer.encode(content, add_special_tokens=False)
                    label_ids += tokenizer.encode(content, add_special_tokens=False)
                else:
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
            "label_ids": label_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


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
                if model.config.architectures == ["MiniCPM3ForCausalLM"]
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
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer


if __name__ == "__main__":
    model_path = "/mnt/data/user/tc_agi/yh/models/MiniCPM"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        use_lora=training_args.use_lora,
        qlora=training_args.qlora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
    )

    train_dataset = SupervisedDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    eval_dataset = SupervisedDataset(
        data_path=data_args.eval_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    # save the incremental PEFT weights, more details can be found in https://huggingface.co/blog/peft
    # model.save_pretrained("output_dir")
