import os
import torch
import argparse
import warnings
import platform

from omegaconf import OmegaConf
from dotenv import load_dotenv
from utils.helper import check_library, all_type_seed
# from utils.util import sync_config
from huggingface_hub import login

from configuration import CFG
from dataset_class.preprocessing import load_all_types_dataset
from dataset_class.make_dataset import build_doc_embedding_db
from db.run_db import run_engine, create_index, get_encoder, insert_doc_embedding, search_candidates


load_dotenv()
check_library(True)
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
all_type_seed(CFG, True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, expandable_segments:True"


def login_to_huggingface() -> None:
    login(os.environ.get("HUGGINGFACE_API_KEY"))
    return


def get_db_url():
    return os.environ.get('ELASTIC_ENGINE_URL')


def get_db_auth(os_type: str) -> str:
    return os.environ.get('MAC_ELASTIC_ENGINE_PASSWORD') if os_type == "Darwin" else os.environ.get('LINUX_ELASTIC_ENGINE_PASSWORD')


def get_db_cert(os_type: str) -> str:
    return os.environ.get('MAC_CA_CERTS') if os_type == "Darwin" else os.environ.get('LINUX_CA_CERTS')


def main() -> None:
    """ main loop function for running the engine

    workflow:
        1) login to huggingface hub

        2) run the elasticsearch engine
            - create index
            - make doc embedding
            - insert doc embedding

        3) search the doc embedding
    """
    login_to_huggingface()
    os_type = platform.system()

    es = run_engine(url=get_db_url(), auth=get_db_auth(os_type), cert=get_db_cert(os_type))
    create_index(es)

    encoder = get_encoder()
    df = build_doc_embedding_db(
        load_all_types_dataset('./dataset_class/arxiv_qa/train_paper_meta_db.csv')
    )

    insert_doc_embedding(
        df=df,
        encoder=encoder,
        es=es
    )

    query = "What is the self-attention mechanism in transformer?"
    search_candidates(
        query=query,
        encoder=encoder,
        es=es
    )
    return


if __name__ == '__main__':
    # config = CFG
    # parser = argparse.ArgumentParser(description="Train Script")
    # parser.add_argument("train_type", type=str, help="Train Type Selection")
    # parser.add_argument("model_config", type=str, help="Model config Selection")
    # args = parser.parse_args()

    main()
