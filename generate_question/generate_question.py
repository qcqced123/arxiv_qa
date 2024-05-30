import os
import pandas as pd
import torch.nn as nn
import google.generativeai as genai


from tqdm.auto import tqdm
from configuration import CFG
from dotenv import load_dotenv
from typing import List, Dict, Any
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def get_necessary_module(cfg: CFG) -> Dict[str, Any]:
    """ function for getting necessary module for the project

    Args:
        cfg (CFG): configuration object for the project

    Returns:
        Dict[str, Any]: dictionary object for the necessary module
    """
    return {
        'tokenizer': AutoTokenizer.from_pretrained(cfg.model_name),
        'model': AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
    }


def generate_with_llama(cfg: CFG, input_ids: List[int], tokenizer: AutoTokenizer, model: nn.Module) -> str:
    """ function for generating "arxiv question-document dataset" by using meta-ai llama3-8b model

    this function will do the "one-shot learning"(meta-learning) for generating the dataset

    Args:
        cfg (CFG): configuration object for the project
        input_ids (List[int]): list of input ids for the model, already passing throug the tokenizer
        tokenizer (AutoTokenizer): default is llama3-8b tokenizer, for tokenizing the input text
        model (nn.Module): default is llama3-8b model, for generating the question
        temperature (float): default 0.0, the temperature value for the diversity of the output text
                             (if you set T < 1.0, the output text will be more deterministic, sharpening softmax dist)
                             (if you set T > 1.0, the output text will be more diverse, flattening softmax dist)
    """
    question = model.generate(
        input_ids=input_ids,
        max_new_tokens=cfg.max_new_tokens,
        max_length=cfg.max_len,
        penalty_alpha=cfg.penalty_alpha,
        num_beams=cfg.num_beams,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        length_penalty=cfg.length_penalty,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        do_sample=cfg.do_sample,
        use_cache=cfg.use_cache,
    )
    return tokenizer.decode(question[0], skip_special_tokens=True)


def google_gemini_api(title: str, context: str, foundation_model: str = 'gemini-pro', temperature: float = 0) -> str:
    """ make Arxiv Questioning & Answering dataset function with Google AI Gemini API

    As you run this function before, you must set up the your own Google API key for the Gemini API.
    you can use the gemini-pro-api for free with the Google API key.

    we will use the Zero-Shot Learning for generating the QA dataset from the given paper link.
    Args:
        title (str): title of the paper
        context (str): sub element of original paper text
        foundation_model (str): The foundation model for extracting food ingredients from the given text,
                                default is 'gemini-pro'
        temperature (float): default 0.0, the temperature value for the diversity of the output text
                             (if you set T < 1.0, the output text will be more deterministic, sharpening softmax dist)
                             (if you set T > 1.0, the output text will be more diverse, flattening softmax dist)

    References:
        https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/tutorials/quickstart_colab.ipynb?hl=ko#scrollTo=HTiaTu6O1LRC
        https://ai.google.dev/gemini-api/docs/models/gemini?hl=ko
        https://ai.google.dev/gemini-api/docs/get-started/python?hl=ko&_gl=1*7ufqxk*_up*MQ..*_ga*MTk2ODk3NDQyNi4xNzE0OTIwMjcw*_ga_P1DBVKWT6V*MTcxNDkyMDI2OS4xLjAuMTcxNDkyMDI2OS4wLjAuOTQwNDMwMTE.
        https://ai.google.dev/gemini-api/docs/quickstart?hl=ko&_gl=1*12k4ofq*_up*MQ..*_ga*MTk2ODk3NDQyNi4xNzE0OTIwMjcw*_ga_P1DBVKWT6V*MTcxNDkyMDI2OS4xLjAuMTcxNDkyMDI2OS4wLjAuOTQwNDMwMTE.
        https://ai.google.dev/api/python/google/generativeai/GenerativeModel?_gl=1*1ajz3qu*_up*MQ..*_ga*MTk2ODk3NDQyNi4xNzE0OTIwMjcw*_ga_P1DBVKWT6V*MTcxNDkyNDAyOC4yLjAuMTcxNDkyNDAyOC4wLjAuMTkwOTQyMjU0#generate_content
    """
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel(foundation_model)
    generation_config = genai.types.GenerationConfig(
        candidate_count=1,
        temperature=temperature
    )

    datasets = ''
    try:
        prompt = f"""[title]\n{title}\n\n[context]\n{context}\n\n
        You're a question machine. Read the context given above and generate the right question.
        Questions should also be able to capture the features or characteristics of a given context.
        The purpose of asking you to create questions is to create a dataset of question-document pairs.
        Please create with purpose and generate creative, informative, and diverse questions.
        """

        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )
        datasets = response.text

    except Exception as e:
        print(e)

    return datasets


if __name__ == '__main__':
    df = pd.read_csv('../dataset_class/datafolder/arxiv_qa/partition/1708.02901.csv')
    questions = [google_gemini_api(row['title'], row['doc'])for i, row in tqdm(df.iterrows(), total=len(df))]
    df['question'] = questions
    df.to_csv('../dataset_class/datafolder/arxiv_qa/partition/test_1708.02901.csv', index=False)

    # prompt = f"""[title]\n{title}\n\n[context]\n{context}\n\n
    #             You're a question machine. Read the context given above and generate the right question.
    #             Questions should also be able to capture the features or characteristics of a given context.
    #             The purpose of asking you to create questions is to create a dataset of question-document pairs.
    #             Please create with purpose and generate creative, informative, and diverse questions."""
    #