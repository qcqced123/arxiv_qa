import emoji
import torch
import numpy as np
import pandas as pd
import nemo_text_processing
import re, gc, pickle, json, os
import configuration as configuration

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Callable, Any
from datasets import load_dataset, Dataset, DatasetDict
from nemo_text_processing.text_normalization.normalize import Normalizer
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split


def dataset_split(cfg: configuration.CFG, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split dataset from pandas.DataFrame with sklearn.train_test_split

    Args:
        cfg: configuration.CFG, needed to load split ratio, seed value
        df: pandas.DataFrame, dataset from csv file
    """
    train, valid = train_test_split(
        df,
        test_size=cfg.split_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )
    return train, valid


def chunking(cfg: configuration.CFG, sequences: Dict) -> List[str]:
    """ Chunking sentence to token using pretrained tokenizer

    Args:
        cfg: configuration.CFG, needed to load pretrained tokenizer
        sequences: list, sentence to chunking

    References:
        https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
    """
    return cfg.tokenizer([" ".join(x) for x in sequences['text']])


def group_texts(cfg: configuration.CFG, sequences: Dict) -> Dict:
    """ Dealing Problem: some of data instances are longer than the maximum input length for the model,
    This function is ONLY used to HF Dataset Object

    1) Concatenate all texts
    2) We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    3) customize this part to your needs
    4) Split by chunks of max_len

    """
    concatenated_sequences = {k: sum(sequences[k], []) for k in sequences.keys()}
    total_length = len(concatenated_sequences[list(sequences.keys())[0]])
    if total_length >= cfg.max_seq:
        total_length = (total_length // cfg.max_seq) * cfg.max_seq
    result = {
        k: [t[i: i + cfg.max_seq] for i in range(0, total_length, cfg.max_seq)]
        for k, t in concatenated_sequences.items()
    }
    return result


def apply_preprocess(dataset: Dataset, function: Callable, batched: bool = True, num_proc: int = 4, remove_columns: Any = None) -> Dataset:
    """ Apply preprocessing to text data, which is using huggingface dataset method "map()"
    for pretrained training (MLM, CLM)

    Args:
        dataset: Huggingface Datasets object, dataset from Huggingface Datasets
        function: Callable, function that you want to apply
        batched: bool, default True, if you want to apply function to batched data, set True
        num_proc: int, default 4, number of process for multiprocessing
        remove_columns: any, default None, if you want to remove some columns, set column name

    References:
        https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
    """
    mapped_dataset = dataset.map(
        function,
        batched=batched,
        num_proc=num_proc,
        remove_columns=remove_columns,
    )
    return mapped_dataset


def tokenizing(
    text: str,
    cfg: configuration.CFG,
    max_length: int = None,
    truncation: bool = False,
    padding: bool or str = 'max_length',
    add_special_tokens: bool = False,
    return_token_type_ids: bool = False
) -> Dict[str, torch.Tensor]:
    """ Preprocess text for LLM Input, for common batch system

    Args:
        text: text from dataframe or any other dataset, please pass str type
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        max_length: int, default None, if you want to set max length, init this param to int
        truncation: bool, default False, if you want to use truncation, set True
        padding: padding options, default 'max_length', if you want use smart batching, init this param to False
        add_special_tokens: bool, default False, if you want to use special tokens, set True
        return_token_type_ids: bool, default False, if you want to use token_type_ids, set True
    """
    inputs = cfg.tokenizer.encode_plus(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=None,
        add_special_tokens=add_special_tokens,  # lat we will add ourselves
        return_token_type_ids=return_token_type_ids,
    )
    for k, v in inputs.items():
        inputs[k] = torch.as_tensor(v)  # as_tensor for reducing memory usage, this ops doesn't copy tensor
    return inputs


def adjust_sequences(sequences: List[List], max_len: int):
    """ Similar to dynamic padding concept
    Append slicing index from original, because original source code is implemented weired
    So it generates some problem for applying very longer sequence
    Add -1 value to slicing index, so we can get result what we want

    Args:
        sequences: list of each cell's token sequence in one unique notebook id, must pass tokenized sequence input_ids
        => sequences = [[1,2,3,4,5,6], [1,2,3,4,5,6], ... , [1,2,3,4,5]]
        max_len: max length of sequence into LLM Embedding Layer, default is 2048 for DeBERTa-V3-Large

    Reference:
         https://github.com/louis-she/ai4code/blob/master/ai4code/utils.py#L70
    """
    length_of_seqs = [len(seq) for seq in sequences]
    total_len = sum(length_of_seqs)
    cut_off = total_len - max_len
    if cut_off <= 0:
        return sequences, length_of_seqs

    for _ in range(cut_off):
        max_index = length_of_seqs.index(max(length_of_seqs))
        length_of_seqs[max_index] -= 1
    sequences = [sequences[i][:l-1] for i, l in enumerate(length_of_seqs)]
    return sequences, length_of_seqs


def subsequent_tokenizing(cfg: configuration.CFG, text: str) -> Any:
    """ Tokenize input sentence to longer sequence than common tokenizing
    Append padding strategy NOT Apply same max length, similar concept to dynamic padding
    Truncate longer sequence to match LLM max sequence

    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type

    Reference:
        https://www.kaggle.com/competitions/AI4Code/discussion/343714
        https://github.com/louis-she/ai4code/blob/master/tests/test_utils.py#L6
    """
    inputs = cfg.tokenizer.encode_plus(
        text,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=False,  # No need to special token to subsequent text sequence
    )
    return inputs['input_ids']


def split_list(inputs: List, max_length: int, instance_max: int) -> List[List]:
    """ Split List into sub shorter list, which is longer than max_length
    """
    result = [inputs[i:i + max_length] for i in range(0, len(inputs), instance_max)]
    return result


def split_longer_text_with_sliding_window(inputs: List[List], max_length: int = 512, window_size: int = 500) -> List[List[List[int]]]:
    """ Flatten Nested List to 1D-List """
    return [inputs[i:i + max_length] for i in range(0, len(inputs), max_length-window_size)]


def save_pkl(input_dict: Any, filename: str) -> None:
    """ Save pickle file
    """
    with open(f'{filename}.pkl', 'wb') as file:
        pickle.dump(input_dict, file)


def load_pkl(filepath: str) -> Any:
    """ Load pickle file

    Examples:
        filepath = './dataset_class/data_folder/insert.pkl'
    """
    with open(f'{filepath}', 'rb') as file:
        output = pickle.load(file)
    return output


def load_json(filepath: str) -> pd.DataFrame:
    """ Load json file

    Examples:
        filepath = './dataset_class/data_folder/insert.json'
    """
    output = pd.read_json(filepath)
    return output


def load_parquet(filepath: str) -> pd.DataFrame:
    """ Load parquet file

    Examples:
        filepath = './dataset_class/data_folder/insert.parquet'
    """
    output = pd.read_parquet(filepath)
    return output


def load_csv(filepath: str) -> pd.DataFrame:
    """ Load csv file

    Examples:
        filepath = './dataset_class/data_folder/insert.csv'
    """
    output = pd.read_csv(filepath)
    return output


def load_all_types_dataset(path: str) -> pd.DataFrame:
    """ Load all pickle files from folder
    Args:
        path: path in your local directory

    Examples:
        load_all_types_dataset('./data_folder/squad2/insert.json')
        load_all_types_dataset('./data_folder/yahoo_qa/test.csv')
        load_all_types_dataset('./data_folder/yelp/train_0.parquet')

    All of file types are supported: json, csv, parquet, pkl
    And Then, they are converted to dict type in python
    """
    output = None
    file_types = path.split('.')[-1]
    if file_types == 'pkl':
        output = load_pkl(path)
    elif file_types == 'json':
        output = load_json(path)
    elif file_types == 'parquet':
        output = load_parquet(path)
    elif file_types == 'csv':
        output = load_csv(path)

    return output


def jump_exist_paper(pid: str):
    """jump function if the current pid pdf file is already in partition folder,
    which is meaning that file is already processed by chunking algorithm

    Args:
        pid (str): current pdf file for processing by chunking algorithm
    """
    curr = f"{pid}.csv"
    exist_pids = os.listdir("./datafolder/arxiv_qa/partition/")
    return True if curr in exist_pids else False


def merge_partition_files() -> pd.DataFrame:
    base_path = "./datafolder/arxiv_qa/partition/"
    df = pd.DataFrame(columns=['paper_id', 'doc_id', 'title', 'doc'])

    for file_name in os.listdir(base_path):
        curr = pd.read_csv(base_path + file_name)
        df = pd.concat([df, curr], axis=0)

    return df


def no_multi_spaces(text):
    return re.sub(r"\s+", " ", text, flags=re.I)


def cleaning_words(text):
    text = re.sub(r"[^\w\s.,!?\"'\-]", "", text)  # remove the not sentence symbol
    text = re.sub(r"<[^>]+>", "", text)  # remove markdown code
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # remove markdown link text
    text = re.sub(r"\*\*.*?\*\*", "", text)  # remove makrdown bold
    text = re.sub(r"\#.*?\n", "", text)  # remove header of markdown
    text = re.sub(r"[\U0001F600-\U0001F64F]", "", text)  # remove emoji
    text = re.sub(r"\s+", " ", text, flags=re.I)
    emoji.demojize(text)
    return text.lower()


def init_normalizer(mode: str = "cased", language: str = "en") -> Normalizer:
    """ function for initializing the Text Normalizer from NVIDIA NeMo
    Args:
        mode (str): options for "lower_cased", "cased"
        language (str): default setting is english "en"

    Reference:
        https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html#text-normalization
    """
    return Normalizer(
        input_case=mode,
        lang=language
    )


def apply_normalizer(normalizer: Normalizer, text: str) -> str:
    """ wrapper function for Text Normalizer from NVIDIA NeMo

    normalizer will do normalize tokens from written to spoken form
    e.g. 12 kg -> twelve kilograms

    normalize function's param explain:
        text: string that may include semiotic classes
        punct_pre_process: whether to perform punctuation pre-processing, for example, [25] -> [ 25 ]
        punct_post_process: whether to normalize punctuation
        verbose: whether to print intermediate meta information

    Reference:
        https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/text_normalization/normalize.py
        https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html#text-normalization
    """
    if not isinstance(text, str):
        text = str(text)

    return normalizer.normalize(
        text,
        verbose=False,
        punct_post_process=False
    )


def apply_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    """ wrapper function for AutoTokenizer.apply_chat_template() """
    message = [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]
    return tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
