import torch
import torch.nn as nn
import importlib.util

from configuration import CFG
from typing import Tuple, Dict
from peft import get_peft_model, LoraConfig
from peft import PromptEncoderConfig, PromptEncoder

from transformers import BitsAndBytesConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification


class AbstractTask:
    """ Abstract model class for all tasks in this project
    Each task should inherit this class for using common functionalities

    Functions:
        1) Init Gradient Checkpointing Flag

        2) Weight Initialization
            - Pytorch Default Weight Initialization: He Initialization (Kaiming Initialization)

        3) Interface method for making model instance in runtime

        4) Apply fine-tune options, which are selected in configuration.json
            - load pretrained weights for fine-tune (own your hub, huggingface model hub ... etc)
            - apply PEFT (Quantization, LoRA, QLoRA, P-Tuning, ...)
    """
    def __init__(self, cfg: CFG) -> None:
        super(AbstractTask, self).__init__()
        self.cfg = cfg

    def _init_weights(self, module: nn.Module) -> None:
        """ over-ride initializes weights of the given module function for torch models
        you must implement this function in your task class

        Args:
            module (:obj:`torch.nn.Module`):
                The module to initialize weights for
        """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            elif self.cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif self.cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def get_config(self) -> AutoConfig:
        """ helper method for returning the pretrained model's configuration file for initializing the model module
        """
        return AutoConfig.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=True
        )

    def get_bit_config(self) -> BitsAndBytesConfig:
        """ method for initializing the quantization module in QLoRA, LLM.int8()
        AWQ doesn't use bitsandbytes library, they use the autoawq library and they applied the quantization previous time
        """
        bit_config = None
        if self.cfg.quantization == "QLoRA":
            bit_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        elif self.cfg.quantization == "LLM.int8":
            bit_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=True,
            )
        elif self.cfg.quantization == "AWQ":
            pass

        return bit_config

    def get_model(self, mode: str, config: AutoConfig, bit_config: BitsAndBytesConfig) -> nn.Module:
        """ method for loading the pretrained model from huggingface hub or local disk with initializing setting

        Args:
            mode (str): value of task module, options for "generation", "classification", "text-similarity"
            config (AutoConfig): pretrained configuration module from AutoConfig in transformers
            bit_config (BitsAndBytesConfig): configuration module for quantization
        """
        model = None
        if mode == "generation":
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                config=config,
                quantization_config=bit_config,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=self.cfg.model_dtype
            )
        elif mode == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.model_name,
                config=config,
                quantization_config=bit_config,
                trust_remote_code=True,
                attn_implementation="sdpa",  # kernel fusion attention module in pytorch, encoder model do not support flash attention
                torch_dtype=self.cfg.model_dtype
            )
        elif mode == "text-similarity":
            model = AutoModel.from_pretrained(
                self.cfg.model_name,
                config=config,
                quantization_config=bit_config,
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=self.cfg.model_dtype
            )
        return model

    def get_lora_config(self) -> LoraConfig:
        """ method for initializing the LoRA configuration module to apply model to LoRA """
        return LoraConfig(
            target_modules="all-linear",
            task_type="None",
            inference_mode=False,
            r=self.cfg.lora_rank,  # rank value
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias='none',
        )

    def apply_peft_lora(self, model: nn.Module) -> nn.Module:
        """ class method for applying peft lora and qlora to pretrained model in fine-tune stage
        if you want to apply LoRA, QLoRA to your model, please check the model architecture what you want to apply

        because, when you use model having the quite different object name from original,
        you must specify the target_modules in LoraConfig. For example, if you want to apply LoRA on Longformer, Bigbird
        which have the qutie different object name, they cannot be applied by default setting (not specified target_modules)

        In this case, you can use the option named "all-linear" in target_modules argument in LoraConfig

        Args:
            model: pretrained model from huggingface model hub

        Notes:
            Default PEFT LoRA setting is applying to query, key, value, and output layers of each attention layer
            You can select the applied layers by changing the argument 'target_modules' in LoraConfig
            => config = LoraConfig(target_modules="all-linear", ...)

        Reference:
            https://github.com/huggingface/peft?tab=readme-ov-file
            https://huggingface.co/docs/peft/en/developer_guides/lora
            https://arxiv.org/abs/2106.09685
            https://arxiv.org/abs/2305.14314
        """
        lora_config = self.get_lora_config()
        peft_model = get_peft_model(model=model, peft_config=lora_config)
        peft_model.enable_input_require_grads()
        return peft_model

    def apply_peft_prompt_tuning(self) -> nn.Module:
        """ class method for applying peft p-tuning to pretrained model in fine-tune stage
        """
        task_type = None if not self.cfg.task_type else self.cfg.task_type
        config = PromptEncoderConfig(
            peft_type=self.cfg.prompt_tuning_type,
            task_type=task_type,
            num_virtual_tokens=self.cfg.num_virtual_tokens,
            token_dim=self.cfg.virtual_token_dim,
            encoder_reparameterization_type=self.cfg.encoder_reparameterization_type,
            encoder_hidden_size=self.cfg.prompt_encoder_hidden_size,
        )
        prompt_encoder = PromptEncoder(config)
        return prompt_encoder

    def select_pt_model(self, mode: str = "generation") -> Dict:
        """ Selects architecture for each task (fine-tune),
        you can easily select your model for experiment from json config files
        or choose the pretrained weight hub (own your pretrain, huggingface ... etc)

        Args:
            mode (str): flag value of task, this value will determine the loading automodel task module

        Reference:
            https://huggingface.co/docs/peft/en/developer_guides/quantization
            https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning
            https://huggingface.co/docs/peft/en/developer_guides/custom_models

        """
        prompt_encoder = None

        # initialize the configuration module
        # (optional) initialize quantization module if necessary, such as AWQ, QLoRA, LLM.int8()
        # load weights from local hub or huggingface model hub
        config = self.get_config()
        if mode == "generation":
            config.use_cache = self.cfg.use_cache
            config.tie_word_embedding = self.cfg.tie_weight  # add to configuration pymodule

        bit_config = self.get_bit_config() if self.cfg.quantization is not None else None
        model = self.get_model(mode=mode, config=config, bit_config=bit_config)

        # (optional) initialize the setting for LoRA
        if self.cfg.lora:
            model = self.apply_peft_lora(model)

        # (optional) apply prompt tuning
        if self.cfg.prompt_tuning:
            prompt_encoder = self.apply_peft_prompt_tuning()

        return {
            'plm_config': config,
            'plm': model,
            'prompt_encoder': prompt_encoder
        }

