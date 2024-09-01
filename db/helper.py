import os
import gc
import copy
import json
import shutil

import torch
import torch.nn as nn
import bitsandbytes as bnb

from typing import Any
from peft.utils import _get_submodules
from peft import PeftModel, LoraConfig
from peft import get_peft_config, get_peft_model
from bitsandbytes.functional import dequantize_4bit

from transformers import BitsAndBytesConfig
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """ function for getting the pretrained tokenizer from AutoTokenizer library, this module will be from local path

    Args:
        model_name (str): local hub path of pretrained model or tokenizer
    """
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )


def get_config(model_name: str) -> AutoConfig:
    """ function for getting the pretrained configuration file from AutoConfig library, this module will be from local path

    Args:
        model_name (str): local hub path of pretrained model
    """
    return AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )


def get_bit_config() -> BitsAndBytesConfig:
    """ function for getting QLoRA bit config """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def get_qlora_model(
    model_name: str,
    config: AutoConfig,
    bit_config: BitsAndBytesConfig,
    device: str,
    model_dtype: Any = "auto"
) -> nn.Module:
    """ function for loading the pretrained model weight by applying the QLoRA Normal Float 4bit precision

    Args:
        model_name (str):
        config (AutoConfig):
        bit_config (BitsAndBytesConfig):
        device (str):
        model_dtype (str, torch.Tensor, optional): set up your model's bit precision, default value is "auto"
                                                   "auto" means that set up the bit precision from AutoConfig setting
                                                   you can select torch.bfloat16, torch.float16, torch.float32 ...
    """
    return AutoModel.from_pretrained(
        model_name,
        config=config,
        quantization_config=bit_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map=device,
        torch_dtype=model_dtype,
    )


def dequantize_model(
    model: nn.Module,
    dtype=torch.bfloat16,
    device: str = "cuda:0"
):
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using, (our case, we must use fp16 for exporting to onnx)
    'device': device to load the model to
    """
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)

                quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

            else:
                print(f"Converting dtype `{name}`...")
                for param in module.parameters(recurse=False):
                    param.data = param.data.to(dtype)

                for buffer in module.buffers():
                    buffer.data = buffer.data.to(dtype)

        # save your de-quantized model in local disk
        model.is_loaded_in_4bit = False
        return model


def apply_lora(model: nn.Module, path: str, dtype: torch.dtype) -> PeftModel:
    """ function for loading the fine-tuned LoRA weight from local disk

    Args:
        model (nn.Module):
        path (str):
        dtype (torch.dtype):
    """
    return PeftModel.from_pretrained(
        model=model,
        model_id=path,
        dtype=dtype
    )


def merge_llm_with_lora(peft_model: PeftModel) -> PeftModel:
    output_model = peft_model.merge_and_unload()
    return output_model


def save_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    config: AutoConfig,
    model_dtype: torch.dtype = torch.bfloat16,
    to: str = ""
) -> None:
    """ function for save your merged model into local disk """
    print(f"Saving dequantized model to {to}...")

    # Delete the model object if it exists
    if os.path.exists(to):
        shutil.rmtree(to)

    os.makedirs(to, exist_ok=True)

    # save your model checkpoint
    model.save_pretrained(to)

    # save your model's configuration file
    config.pop("quantization_config", None)
    config.pop("pretraining_tp", None)
    config.torch_dtype = model_dtype
    config.save_pretrained(to)

    # save your model's tokenizer module

    return None


if __name__ == '__main__':
    device = "cpu"
    model_dtype = torch.bfloat16
    lora_path = "../saved/fine_tuned-qlora-e5-large-v2"
    model_name = "../saved/e5-large-v2"
    output_path = "../saved/merged-qlora-e5-large-v2"

    config = get_config(model_name)
    bit_config = get_bit_config()
    tokenizer = get_tokenizer(model_name)
    model = get_qlora_model(
        model_name=model_name,
        config=config,
        bit_config=bit_config,
        device=device,
        model_dtype=model_dtype
    )
    dequantized_model = dequantize_model(
        model=model,
        dtype=model_dtype,
        device=device
    )

    peft_model = apply_lora(model=dequantized_model, path=lora_path, dtype=model_dtype)
    merged_model = merge_llm_with_lora(peft_model=peft_model)

    save_model(
        model=merged_model,
        tokenizer=tokenizer,
        config=config,
        model_dtype=model_dtype,
        to=output_path
    )
