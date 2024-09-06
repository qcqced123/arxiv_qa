"""python module for inference-pipeline by using Huggingface generate()
"""
import torch
import torch.nn as nn

from typing import List, Dict
from transformers import AutoConfig, AutoTokenizer


@torch.no_grad()
def inference(
    self,
    model: nn.Module,
    max_new_tokens: int,
    max_length: int,
    query: str = None,
    context: str = None,
    prompt: str = None,
    strategy: str = None,
    penalty_alpha: float = None,
    num_beams: int = None,
    temperature: float = 1,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = None,
    length_penalty: float = None,
    no_repeat_ngram_size: int = None,
    do_sample: bool = False,
    use_cache: bool = True,
) -> List[Dict]:
    """ inference function for using pure huggingface generative API (generate() function)
    method for making the answer from the given prompt (context + query) or pre-defined prompt by caller

    this method is designed with native pytorch & huggingface library,
    if you want to use other faster platform such as tensorrt_llm, vllm, you must change the config value,
    named "inference_pipeline"

    generate method's arguments setting guide:
        1) num_beams: 1 for greedy search, > 1 for beam search

        2) temperature: softmax distribution, default is 1 meaning that original softmax distribution
                        (if you set < 1, it will be more greedy, if you set > 1, it will be more diverse)

        3) do_sample: flag for using sampling method, default is False
                      (if you want to use top-k or top-p sampling, set this flag to True)

        4) top_k: top-k sampling, default is 50, must do_sample=True
        5) top_p: top-p (nucleus) sampling, default is 0.9, must do_sample=True
    """
    prompt = prompt if prompt is not None else f"context:{context}\nquery:{query}"

    inputs = self.tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )
    for k, v in inputs.items():
        inputs[k] = torch.as_tensor(v)

    output = model.model.generate(
        input_ids=inputs['input_ids'].to(self.cfg.device),
        max_new_tokens=max_new_tokens,
        # max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        penalty_alpha=penalty_alpha,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        use_cache=use_cache,
    )
    # output has nested tensor, so we need to flatten it for decoding
    result = self.tokenizer.decode(
        output[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return result
