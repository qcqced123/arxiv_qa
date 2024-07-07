import os
import torch
import argparse
import warnings
import platform
import pandas as pd

from typing import List
from tqdm.auto import tqdm
from dotenv import load_dotenv
from omegaconf import OmegaConf
from multiprocessing import pool
from huggingface_hub import login
from transformers import logging as transformers_logging

from utils.util import sync_config
from utils.helper import check_library, all_type_seed

from configuration import CFG
from prompt.prompt_maker import cut_context
from dataset_class.preprocessing import save_pkl
from trainer.train_loop import train_loop, inference_loop
from document_encoder.document_encoder import document_encoder
from db.run_db import run_engine, create_index, get_encoder, insert_doc_embedding, search_candidates
from prompt.prompt_maker import get_prompt_for_question_generation, get_prompt_for_retrieval_augmented_generation
from generate_question.generate_question import get_necessary_module_for_generation_in_local, postprocess
from dataset_class.text_chunk import chunk_by_length, chunk_by_recursive_search, cut_pdf_to_sub_module_with_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, expandable_segments:True"


load_dotenv()
check_library(False)
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

all_type_seed(CFG, True)
g = torch.Generator()
g.manual_seed(CFG.seed)

# global variables
base_path = r"api/arxiv/train/"


def login_to_huggingface() -> None:
    login(os.environ.get("HUGGINGFACE_API_KEY"))
    return


def get_db_url():
    return os.environ.get('ELASTIC_ENGINE_URL')


def get_db_auth(os_type: str) -> str:
    return os.environ.get('MAC_ELASTIC_ENGINE_PASSWORD') if os_type == "Darwin" else os.environ.get('LINUX_ELASTIC_ENGINE_PASSWORD')


def get_db_cert(os_type: str) -> str:
    return os.environ.get('MAC_CA_CERTS') if os_type == "Darwin" else os.environ.get('LINUX_CA_CERTS')


def make_loop(path_list: List[str]) -> List:
    data_list = []
    for path in tqdm(path_list):
        print(path.split('_'))
        pid, title = path.split('_')  # current pdf file's paper id and title for making dataframe
        try:
            result = cut_pdf_to_sub_module_with_text(
                path=base_path + path,
                strategy="hi_res",
                model_name="yolox"
            )
            document = chunk_by_recursive_search(
                text=result['text'],
                chunk_size=1024,
                over_lapping_size=128
            )
            documents = [e.page_content for e in document]
            for e in result["table"]:
                documents.append(e)

            for e in result["formula"]:
                documents.append(e)
            documents.append(result['reference'])

            data = [[pid, f"{pid}_{i}", title, d] for i, d in enumerate(documents)]
            data_list.extend(data)

            curr = pd.DataFrame(
                data=data,
                columns=['paper_id', 'doc_id', 'title', 'doc']
            )
            output_path = f"dataset_class/datafolder/arxiv_qa/partition/{pid}.csv"
            curr.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Error Message: {e}")

    return data_list


def main(cfg: CFG, pipeline_type: str, model_config: str) -> None:
    """ interface function of main application (main loop function)
    you can choose the pipeline type by setting the pipeline type var

    workflow:
        1) login to huggingface hub
        2) update the config file

        3) run the elasticsearch engine

        # CLI argument signature: "make"
        # this branch will be called when the pipeline type var is set to "make"
        4) make the document embedding db, question-document dataset for fine-tuning query encoder with metric learning
            - run the MySQL DB Server for saving the document DataFrame
            - load the pdf file from pdf db

            - apply text chunking strategy from text_chunk module
                => this process will be done by multi-processing

            - make the document dataframe for document embedding db
            - generate the question data by using any generator llm such as
              (microsoft/Phi-3-mini-128k-instruct, meta-llama2-7b-hf, google-gemini-1.5-flash)

            - insert final document dataframe into the MySQL DB Server

        # CLI argument signature: "insert"
        # this branch will be called when the pipeline type var is set to "insert"
        5) run document encoder
            - create the db index (if not exists)
            - get the document dataframe from the MySQL DB Server (document db)
            - make the document embedding db
            - insert the document embedding into the elastic search engine

        # CLI argument signature: "fine_tune", "inference"
        # this branch will be called when the pipeline type var is set to "fine_tune" or "inference"
        6) run query encoder
            - get query from the user or query db
            - encode or project the query into embedding space
            - search the candidates from the document embedding db

        7) run the text generation tuner for generating the answer for the input query
    """
    # login_to_huggingface()
    config_path = f'config/{pipeline_type}/{model_config}.json'
    sync_config(cfg, OmegaConf.load(config_path))

    os_type = platform.system()
    es = run_engine(url=get_db_url(), auth=get_db_auth(os_type), cert=get_db_cert(os_type))

    # need to abstract this branch
    # branch for calling pipeline that builds the document embedding db, q-doc dataset
    if pipeline_type == "make":

        # branch of getting elements from pdf files and making the document dataframe
        # if configuration var state(cfg.work_flow_state) is "init", then make the document dataframe
        # if configuration var state(cfg.work_flow_state) is "resume", then load the document dataframe
        if cfg.work_flow_state == "init":
            path_list = os.listdir(base_path)

            # apply text chunking strategy from text_chunk module
            # linear for-loop by pdf list
            # multi-processing for-loop by pdf list
            size = len(path_list)//cfg.n_jobs
            chunked = [path_list[i:i+size] for i in range(0, len(path_list), size)]
            df = pd.DataFrame(columns=['paper_id', 'doc_id', 'title', 'doc'])

            with pool.Pool(processes=cfg.n_jobs) as p:
                elements = p.map(make_loop, chunked)

            for element in elements:
                df = pd.concat([df, pd.DataFrame(element, columns=['paper_id', 'doc_id', 'title', 'doc'])], axis=0)

            output_path = f"dataset_class/datafolder/arxiv_qa/total/arxiv_paper_document_db.csv"
            df.to_csv(output_path, index=False)

        elif cfg.work_flow_state == "resume":
            df = pd.read_csv('dataset_class/datafolder/arxiv_qa/total/sampling_metric_learning_total_paper_chunk.csv')

        # branch of question generation
        # you can choose any other generator llm in huggingface model hub (currently google gemini api does not support)
        # you can select the question generator by setting the cfg.question_generator
        # default setting is microsoft/Phi-3-mini-128k-instruct
        questions = []
        modules = get_necessary_module_for_generation_in_local(cfg, es, g)
        tokenizer, tuner, generator = modules['tokenizer'], modules['tuner'], modules['generator']
        for i, row in tqdm(df.iterrows(), total=len(df)):
            context = cut_context(
                cfg=cfg,
                context=row['doc']
            )
            prompt = get_prompt_for_question_generation(context=context)
            questions.append(
                tuner.inference(
                    model=generator,
                    max_new_tokens=cfg.max_new_tokens,
                    max_length=cfg.max_len,
                    prompt=prompt,
                    penalty_alpha=cfg.penalty_alpha,
                    num_beams=cfg.num_beams,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    repetition_penalty=cfg.repetition_penalty,
                    length_penalty=cfg.length_penalty,
                    do_sample=cfg.do_sample,
                    use_cache=cfg.use_cache
                )
            )
        # branch merge point of question generation
        # save the question data object with pickle for backup if current logic will be failed
        save_pkl(questions, 'dataset_class/datafolder/arxiv_qa/total/test_generate_question_document_db')
        df['question'] = [postprocess(question) for question in questions]

        output_path = f"dataset_class/datafolder/arxiv_qa/total/test_generate_question_document_db.csv"
        df.to_csv(output_path, index=False)

    # branch for calling pipeline that inserts the document embedding into the elastic search engine
    elif pipeline_type == "insert":
        retriever = get_encoder()
        df = pd.read_csv('dataset_class/datafolder/arxiv_qa/total/arxiv_paper_document_db.csv')
        document_encoder(
            es=es,
            retriever=retriever,
            df=df
        )
    # branch for calling pipeline that fine-tune the query encoder with metric learning, generator with clm
    elif pipeline_type == "fine_tune":
        train_loop(
            cfg=cfg,
            pipeline_type=pipeline_type,
            model_config=model_config
        )
    # branch for calling pipeline that generates the best answer for the input query
    elif pipeline_type == "inference":
        answers = inference_loop(
            cfg=cfg,
            pipeline_type=pipeline_type,
            model_config=model_config,
            es=es
        )
        for i, answer in enumerate(answers):
            print(f"{i+1}-th question's answer is: \n\n {answer}")
    return


if __name__ == '__main__':
    config = CFG
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("pipeline_type", type=str, help="Train Type Selection")  # insert, inference
    parser.add_argument("model_config", type=str, help="Model config Selection")  # name of retriever-generator
    args = parser.parse_args()

    main(config, args.pipeline_type, args.model_config)
