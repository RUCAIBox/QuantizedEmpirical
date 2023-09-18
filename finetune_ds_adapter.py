import os, sys, utils, gc
import sys
from typing import List
from dataclasses import dataclass, field

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union, Dict, Sequence

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from peft import (  # noqa: E402
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    GPTQLoraConfig,
)
import copy
import numpy as np
import random


import logging
from accelerate.utils import infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaConfig, AutoModel  # noqa: F402
from transformers import LlamaForCausalLM, Trainer
from transformers.modeling_utils import _load_state_dict_into_meta_model
from models.quant.quant_linear_lora import make_quant_linear
from models.bigmodeling import init_empty_weights, load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model
from torch.utils.data import Dataset

seed_value = 2023   # 设定随机数种子
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
torch.backends.cudnn.deterministic = True # 卷积

module_dic = {
    'q_proj': "layers.{}.self_attn.q_proj",
    'k_proj': "layers.{}.self_attn.k_proj",
    'v_proj': "layers.{}.self_attn.v_proj",
    'o_proj': "layers.{}.self_attn.o_proj",
    'gate_proj': "layers.{}.mlp.gate_proj",
    'down_proj': "layers.{}.mlp.down_proj",
    'up_proj': "layers.{}.mlp.up_proj",
}

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_dialogue":(
        "The following is a conversation between a human and an AI assistant. "
        "The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "{input}\n\n[|AI|]:"
    )
}
def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None, # also use for FFT and MPO
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        # mpo params
        lora_mpo: bool = False,
        # quant_params
        bits : int = 16,
        groupsize : int = 128,
        quant_checkpoint : str = "",
        # survey param
        model_max_length : int = 2048,
        fsdp : str = "",
        fsdp_transformer_layer_cls_to_wrap : str = "",
        load_in_8bit: bool = False
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"lora_mpo: {lora_mpo}\n"
        f"bits: {bits}\n"
        f"groupsize: {groupsize}\n"
        f"transformers version: {transformers.__version__}\n"
        f"quant_checkpoint: {quant_checkpoint}\n"
        f"model_max_length: {model_max_length}\n"
        f"fsdp: {fsdp}\n"
        f"fsdp_transformer_layer_cls_to_wrap: {fsdp_transformer_layer_cls_to_wrap}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = {"":0}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    n_gpus = torch.cuda.device_count()
    max_memory = f'{80000}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(base_model, add_eos_token=True,
                                                   model_max_length=model_max_length, padding_side="right", use_fast=False) # from zk
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, add_eos_token=True)
    # tokenizer.add_special_tokens(
    #     {
    #         "eos_token": DEFAULT_EOS_TOKEN,
    #         "bos_token": DEFAULT_BOS_TOKEN,
    #         "unk_token": DEFAULT_UNK_TOKEN,
    #     }
    # )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    # tokenizer.padding_side = "left"  # Allow batched inference
    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        """Resize tokenizer and embedding.
        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    
    if "chatglm" in base_model:
        model = AutoModel.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    elif adapter_name == 'lora':
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            lora_mpo=lora_mpo
        )
        with init_empty_weights():
            model = LlamaForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16,device_map=device_map)
        model = get_peft_model(model, config)
        torch.cuda.empty_cache()
        gc.collect()
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        torch.cuda.empty_cache()
        gc.collect()
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        torch.cuda.empty_cache()
        gc.collect()
    elif adapter_name == 'gptqlora':  
        config = GPTQLoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            bits=bits,
            groupsize=128
        )
        with init_empty_weights():
            model = LlamaForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16)
        model = get_peft_model(model, config)
        load_checkpoint_in_model(model,quant_checkpoint,device_map=device_map) #加载权重
        torch.cuda.empty_cache()
        gc.collect()
    
    if adapter_name == "prefix-tuning":
        model.to('cuda')
    if adapter_name == "lora":
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    # for name, param in model.named_parameters():
    #     if 'lora' not in name:
    #         param.requires_grad = False
    
    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    print(f"After Loaded  with {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB")

    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None

    def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        # examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_id_list, label_list = [], []
        for example in examples:
            raw_text = example.rstrip(tokenizer.eos_token)
            text = []
            # input_ids_list, labels_list = [], []
            for i, txt in enumerate(raw_text.split('\n\n[|AI|]:')):
                if i == 0:
                    text.append(txt + '\n\n[|AI|]:')
                else:
                    split_txt = txt.split('\n\n[|Human|]:')
                    ai_txt = split_txt[0]
                    text.append(ai_txt + tokenizer.eos_token)
                    if len(split_txt) == 2:
                        human_txt = split_txt[1]
                        text.append('\n\n[|Human|]:' + human_txt + '\n\n[|AI|]:')
            inputs = tokenizer(text=text, max_length=tokenizer.model_max_length, truncation=True)
            input_ids, labels = [], []
            for i, iids in enumerate(inputs['input_ids']):
                if i != 0:
                    iids = iids[1:]
                # LPY! add max length constraint
                if len(input_ids) + len(iids) > tokenizer.model_max_length:
                    break
                # LPY! add max length constraint
                input_ids.extend(iids)
                if i % 2 == 0:
                    labels.extend([IGNORE_INDEX] * len(iids))
                else:
                    labels.extend(iids)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            input_id_list.append(input_ids)
            label_list.append(labels)
        return dict(input_ids=input_id_list, labels=label_list)
                
    class SupervisedDataset(Dataset):
        """Dataset for supervised fine-tuning."""

        def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
            super(SupervisedDataset, self).__init__()
            print("Loading data...")
            list_data_dict = utils.jload(data_path)

            print("Formatting inputs...")
            # prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            prompt = PROMPT_DICT['prompt_dialogue']
            sources = [
                prompt.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            print("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""

        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    total_batch_size = micro_batch_size* gradient_accumulation_steps * (world_size if ddp else 1) 
    total_optim_steps = len(train_dataset) * num_epochs // total_batch_size
    saving_step = int(total_optim_steps/10)
    print(f"word_size: {world_size}, total_optim_steps: {total_optim_steps}")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=saving_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            label_names=['labels'],
            # deepspeed="/home/pyliu/projects/git_pro/QuantizedEmpirical/zero_stage2_offload_config.json"
        ),
        data_collator=data_collator
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if adapter_name == "gptqlora":
        def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
            """Collects the state dict and dump to disk."""
            state_dict = trainer.model.state_dict()
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            torch.save(cpu_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        safe_save_model_for_hf_trainer(trainer, output_dir)
    else:
        model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
