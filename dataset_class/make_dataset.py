import os
import pandas as pd
import openai
import google.generativeai as genai

from collections import defaultdict
from typing import List, Dict, Any
from tqdm.auto import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer
from multiprocessing import Pool
from dataset_class.preprocessing import cleaning_words, split_longer_text_with_sliding_window, save_pkl, load_pkl
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader


load_dotenv()
BASE_URL = './api/download/train/'
paper_list = os.listdir(BASE_URL)


def image2doc(image_path: str) -> str:
    """ Convert image to text document by using Langchain Community
    """
    image_loader = UnstructuredImageLoader(image_path)
    output = image_loader.load()[0]
    return output


def pdf2doc(pdf_path: str) -> str:
    """ Convert pdf to text document by using Langchain Community
    """
    pdf_loader = UnstructuredPDFLoader(file_path=pdf_path)
    output = pdf_loader.load()[0].page_content
    return output


def openai_gpt3_api(script: str, foundation_model: str = 'gpt-3.5-turbo-16k', temperature: float = 0) -> List:
    """ extract food ingredients from the given text (YouTube video script) with GPT-3.5 Inference API (OpenAi)

    Args:
        script (str): The string text from YouTube video for extracting food ingredients
        foundation_model (str): The foundation model for extracting food ingredients from the given text,
                                default is 'gpt-3.5-turbo-16k'
        temperature (float): default 0.0, the temperature value for the diversity of the output text
                             (if you set T < 1.0, the output text will be more deterministic, sharpening softmax dist)
                             (if you set T > 1.0, the output text will be more diverse, flattening softmax dist)
    Returns:
        ingredient_data: list of food ingredients from the given text

    References:
        https://arxiv.org/abs/2005.14165
        https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        https://platform.openai.com/docs/guides/text-generation
    """
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    openai.api_key = OPENAI_API_KEY

    ingredient_data = []
    try:
        prompt = "Please provide a 'Food Ingredient List' in a 'structured format'." \
                 "The format should include only the 'ingredient name' without quantities or units." \
                 "If you don't have an ingredient list, simply write 'No ingredient'." \
                 "When compiling the 'Food Ingredient List' from the given [script], ensure that it is written in 'English'." \
                 "Additionally, exclude non-food items and focus soely on 'edible ingredients'." \
                 "For example: - Egg, - Cheese, - Tomato, - Salt, - Sugar"

        response = openai.ChatCompletion.create(
            model=foundation_model,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': script},
            ],
            temperature=temperature
        )

    except Exception as e:
        print(e)

    return ingredient_data


def google_gemini_api(paper_link: str, foundation_model: str = 'gemini-pro', temperature: float = 0) -> str:
    """ make Arxiv Questioning & Answering dataset function with Google AI Gemini API

    As you run this function before, you must set up the your own Google API key for the Gemini API.
    you can use the gemini-pro-api for free with the Google API key.

    we will use the Zero-Shot Learning for generating the QA dataset from the given paper link.
    Args:
        paper_link (str): the paper link for QA dataset
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
    # context = pdf2doc(paper_link)
    datasets = ''
    try:
        prompt = f"""[context] {paper_link}
        Please make some questions and answers from [context].
        When creating your questions and answers from [context],
        please generate much longer and detailed information into questions and answers
        When creating content for a single question and answer, you must do not line breaks
        Also, don't use bold, italic, or underline text in the questions and answers and make output shape as below:

        Questions:
        1.
        2.
        3.
        ...
        10.

        Answers:
        1.
        2.
        3.
        ...
        10.

        """

        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )
        datasets = response.text

    except Exception as e:
        print(e)

    return datasets


def get_paper_from_list(base_path: str) -> None:
    """ get paper from the list of papers

    Args:
        base_path (str): the base path for the list of papers
    """
    df = pd.DataFrame(columns=['paper_id'])
    paper_list = os.listdir(base_path)

    df['paper_id'] = [".".join(pid.split('.')[0:2]) for pid in paper_list]
    df.to_csv('paper_id_list.csv', index=False)
    return


def remove_garbage(text: str) -> str:
    """ remove garbage text from arxiv paper
    """
    text = text[text.find("v i X r a\n\n") + len("v i X r a\n\n"):]
    return text


def cal_token_length(text: str) -> int:
    return len(AutoTokenizer.from_pretrained('google-bert/bert-base-uncased').encode(text))


def build_qa_dataframe(text: str) -> pd.DataFrame:
    """ build QA dataset from the given paper link

    Args:
        text (str): the text (Question and Answering from Open Ai GPT-3.5 API or Google AI Gemini API)
    """
    questions_text = text.split("Questions:")[1].split("Answers:")[0].strip()
    answers_text = text.split("Questions:")[1].split("Answers:")[1].strip()

    questions = [q[q.find('. ') + len('. '):].strip() for q in questions_text.split("\n") if q.strip()]
    answers = [a[a.find('. ') + len('. '):].strip() for a in answers_text.split("\n") if a.strip()]
    df = pd.DataFrame({"Questions": questions, "Answers": answers})
    return df


def build_qa_dataset(chunk_size: int = 2, eps: int = 500) -> pd.DataFrame:
    """ build the QA dataset from the arxiv paper list

    Args:
        chunk_size (int): the chunk size for the splitting the long text
        eps (int): the epsilon value for the splitting the text, duplicate area of current whole text
                   for using sliding window
    """
    BASE_URL = '../crawler/arxiv/download/test'
    paper_list = os.listdir(BASE_URL)

    for link in paper_list:
        unstructured_text = pdf2doc(f"{BASE_URL}/{link}")
        text = remove_garbage(unstructured_text)
        clean_text = cleaning_words(text)  # remove all of trash text such as this papers pid
        token_list = clean_text.split(' ')

        print(f"Current Paper's Token Length by Python inherited split() method: {len(token_list)}")
        print(f"Current Paper's Token Length by Huggingface Tokenizer: {cal_token_length(clean_text)}")

        """ save the cleaned text to the file """
        df = pd.DataFrame(columns=['Questions', 'Answers'])
        for sub_text in [token_list[0:len(token_list) // chunk_size + eps], token_list[len(token_list) // chunk_size - eps:]]:
            text = google_gemini_api(''.join(sub_text))
            pd.concat([df, build_qa_dataframe(text)], axis=0, ignore_index=True)

        print(f"Generated Questions and Answers: {df}")
        df.to_csv('./data_folder/arxiv_qa/.csv', index=False)

    return df


def build_train_dataframe() -> pd.DataFrame:
    """ build the paper meta data from the arxiv paper list for train dataset

    url example: 'https://arxiv.org/pdf/2006.03654'

    """
    BASE_URL = '../crawler/arxiv/download/train/'
    paper_list = os.listdir(BASE_URL)

    data = []
    for paper in tqdm(paper_list):
        clean_text = ''
        pid, title = paper.split('_')[0], paper.split('_')[1][:-4]
        try:
            unstructured_text = pdf2doc(BASE_URL + paper)
            text = remove_garbage(unstructured_text)
            clean_text = cleaning_words(text)  # remove all of trash text such as this papers pid

        except Exception as e:
            print(e)
            print(f"Error occurred in the paper: {pid, title}")

        data.append([pid, title, clean_text])

    df = pd.DataFrame(data, columns=['pid', 'title', 'text'])
    output_path = f'./data_folder/arxiv_qa/0_4000_paper_meta_db.csv'
    df.to_csv(output_path, index=False)
    return df


def build_train_dataframe_for_multiprocessing(paper_list: List[str]) -> List[List[str]]:
    """ build the paper meta data from the arxiv paper list for train dataset

    url example: 'https://arxiv.org/pdf/2006.03654'
    """
    data = []
    for paper in tqdm(paper_list):
        clean_text = ''
        pid, title = paper.split('_')[0], paper.split('_')[1][:-4]
        try:
            unstructured_text = pdf2doc(BASE_URL + paper)
            text = remove_garbage(unstructured_text)
            clean_text = cleaning_words(text)  # remove all of trash text such as this papers pid

        except Exception as e:
            print(e)
            print(f"Error occurred in the paper: {pid, title}")

        data.append([pid, title, clean_text])

    return data


def build_train_text() -> Dict[str, List[List[int]]]:
    """ build the text inputs for the casual language modeling for the arxiv paper (Generative QA Task)
    """
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    df = pd.read_csv('./data_folder/arxiv_qa/0_4000_paper_meta_db.csv')
    df.dropna(subset=['text'], inplace=True)
    texts = df['text'].tolist()

    step = 10
    result = defaultdict(list)
    for i in tqdm(range(0, len(texts), step)):
        token, attention_mask = [], []
        contexts = ''.join(texts[i:i+step])
        inputs = tokenizer(contexts, padding=False, truncation=False)
        for k in inputs.keys():
            for data in tqdm(split_longer_text_with_sliding_window(inputs[k], 4096, 1024)):
                token.append(data) if k == 'input_ids' else attention_mask.append(data)

        result['input_ids'].extend(token)
        result['attention_mask'].extend(attention_mask)

    return result


def build_doc_embedding_db(df: pd.DataFrame) -> pd.DataFrame:
    """ build the document embedding for the arxiv paper

    Args:
        df (pd.DataFrame): the dataframe containing the paper meta data
    """
    instance = []
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')  # RAG generator's tokenizer
    for i, row in tqdm(df.iterrows()):
        cnt = 0
        pid, title, text = row['pid'], row['title'], row['text']
        try:
            inputs = tokenizer(text, padding=False, truncation=False, add_special_tokens=False)

        except Exception as e:
            print(e)
            print(f"Error occurred in the paper: {pid, title}")
            print(f"Error text: {text}")
            continue

        for data in split_longer_text_with_sliding_window(inputs['input_ids'], 1536, 256):
            instance.append([pid, f"{pid}_{cnt}", title, tokenizer.decode(data)])
            cnt += 1

    return pd.DataFrame(
        data=instance,
        columns=['paper_id', 'doc_id', 'title', 'doc']
    )


if __name__ == '__main__':
    """ length problem: if you input too much longer pdf, the google gemini api will return 500 ERROR
    you can input your own pdf until then 25 pages (30720 tokens)
    """
    # build_qa_dataset()
    # build_train_dataframe()
    # n_jobs = 5
    # chunked = [paper_list[i:i + len(paper_list)//n_jobs] for i in range(0, len(paper_list), len(paper_list)//n_jobs)]
    # with Pool(processes=n_jobs) as pool:
    #     data = pool.map(build_train_dataframe_for_multiprocessing, chunked)
    #
    # data = [item for sublist in data for item in sublist]
    # df = pd.DataFrame(data, columns=['pid', 'title', 'text'])
    # output_path = f'./data_folder/arxiv_qa/0_4000_paper_meta_db.csv'
    # df.to_csv(output_path, index=False)

    data1 = build_train_text()
    save_pkl(data1, './data_folder/arxiv_qa/train_text1')

    data2 = load_pkl('./data_folder/arxiv_qa/train_text2.pkl')
    data = {k: v + data2[k] for k, v in data1.items()}
    save_pkl(data, './data_folder/arxiv_qa/train_text')
