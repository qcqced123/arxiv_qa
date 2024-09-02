from typing import List
from configuration import CFG
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from inference.helper import cut_context, get_prompt_for_answering_question, apply_template


def build_prompt(tokenizer: AutoTokenizer, queries: List[str], candidates: List[str]) -> List[str]:
    """ build input prompt for answering the questions from users by using vllm generative backend

    Args:
        tokenizer (AutoTokenizer):
        queries (List[str]):
        candidates (List[str]):
    """
    return [
        apply_template(tokenizer, get_prompt_for_answering_question(q, cut_context(tokenizer, str(d)))) for q, d in zip(queries, candidates)
    ]


def initialize_llm(model_name: str, max_length: int, max_seq_len_to_capture: int, q_method: str) -> LLM:
    """ wrapper function for initializing the vLLM LLM Engine class

    Args:
        model_name (str): model name from huggingface model hub or local model path
        max_length (int): model's max context length, you must pass the default value of model
        max_seq_len_to_capture (int): maximum value of input sequence (<= max_length)
        q_method (str): quantization method of model, passing this function

    LLM Module's Param:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
                        if available, and "slow" will always use the slow tokenizer.

        skip_tokenizer_init: If true, skip initialization of tokenizer and detokenizer.
                             Expect valid prompt_token_ids and None for prompt from the input.

        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.

        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.

        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.

        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", "squeezellm", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.

        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.

        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.

        seed: The seed to initialize the random number generator for sampling.

        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.

        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.

        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.

        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.

        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.

    Reference:
        https://tech.scatterlab.co.kr/vllm-implementation-details/
        https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

    """
    return LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=max_length,
        max_seq_len_to_capture=max_seq_len_to_capture,
        quantization=q_method,
        trust_remote_code=True,
        swap_space=0
    )


def get_sampling_params(cfg: CFG) -> SamplingParams:
    """ function for returning the vllm generative backend sampling param config module

    Args:
        cfg (CFG): configuration module for initializing the vllm generative backend sampling param config module
    """
    return SamplingParams(
        seed=cfg.seed,
        max_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        presence_penalty=cfg.presence_penalty,
        frequency_penalty=cfg.frequency_penalty,
        repetition_penalty=cfg.repetition_penalty,
        stop=cfg.stop,
        skip_special_tokens=cfg.skip_special_tokens
    )


def do_inference(llm: LLM, inputs: List[str], sampling_params: SamplingParams):
    return llm.generate(
        inputs,
        sampling_params=sampling_params
    )


def slice_full_questions(output: str):
    pass
