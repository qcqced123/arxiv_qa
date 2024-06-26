import os
import time
import torch
import random
import pandas as pd
import google.generativeai as genai

from tqdm.auto import tqdm
from configuration import CFG
from dotenv import load_dotenv
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trainer.trainer import TextGenerationTuner

load_dotenv()


def get_random_sleep(ls: float, ms: float) -> None:
    """ get random sleep time between 5 and 10 seconds

    Args:
        ls: least time to sleep this process
        ms: max time to sleep this process
    """
    sleep_time = random.uniform(ls, ms)
    time.sleep(sleep_time)
    return


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


# def postprocess(output: str) -> str:
#     """ function for postprocessing the output text from question generation model,
#
#     extracting the questions from the output text
#
#     Args:
#         output (str): output text from the question generation model, containing the generated questions
#
#     example:
#         before = [Question 1: Value1, Question 2: Value2, Question 3: Value3]
#         after = Value1?\n Value2?\n Value3?\n
#     """
#     output = output.split('=====')[1]
#     start = output.find('Question 1:')
#     question_list = [text for text in output[start:].split('\n\n') if text.startswith('Question')]
#     print(f"current question list is: {question_list}")
#     question = [text.split(': ')[1].strip() + '\n' for text in question_list]
#     print(f"after postprocess: {question}")
#     return ''.join(question)


def postprocess(output: str) -> str:
    """ function for postprocessing the output text from question generation model,

    extracting the questions from the output text

    Args:
        output (str): output text from the question generation model, containing the generated questions

    example:
        before = [Question 1: Value1, Question 2: Value2, Question 3: Value3]
        after = Value1?\n Value2?\n Value3?\n
    """
    output = output.split('=====')[1]
    start = output.find('Question 1:')

    question_list = [text for text in output[start:].split('\n\n') if text.startswith('Question')]
    question = []
    for text in question_list:
        try:
            question.append(text.split(': ')[1].strip() + '\n')

        except IndexError as e:
            continue

    return ''.join(question)


def get_necessary_module_for_generation_in_local(cfg: CFG, es: Elasticsearch, g: torch.Generator) -> Dict[str, Any]:
    """ function for getting necessary module for the project
    Args:
        cfg (CFG): configuration object for the project
        es (Elasticsearch): elasticsearch object for the project
        g (torch.Generator): torch.Generator object for the project

    Returns:
        Dict[str, Any]: dictionary object for the necessary module
    """
    tuner = TextGenerationTuner(
        cfg=cfg,
        generator=g,
        is_train=False,
        es=es
    )
    _, generator, *_ = tuner.model_setting()
    return {
        'tokenizer': cfg.tokenizer,
        'tuner': tuner,
        'generator': generator
    }


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
        prompt = f"""title:{title}\ncontext:{context}\n\n
        You're a question machine. Read the title and context given above and generate the right question based on given context. Here are some rules for generating the questions:
        1. Questions should also be able to capture the features or characteristics of a given context.
        2. The purpose of asking you to create questions is to create a dataset of question-document pairs.
        3. Please create with purpose and generate creative, informative, and diverse questions.
        4. Do not return questions that are too similar to each other, or too general.
        5. Please only return the question text, keep the number of questions between 1 and 5 with total length less than 100 tokens.
        6. If you want to ask multiple questions, please separate them with spaces without newlines.
        """
        get_random_sleep(8, 10)  # for avoiding the "call api limit"
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
