""" Python module, implemented text chunking strategy wrapping langchain or custom strategy
also combined them together with unstructured partitioning strategy for pdf files
for making better split Arxiv paper document

[Features of Arxiv Paper and Strategy for Splitting]
1) There are a lot of mathematical formulas
    - use latex splitter for splitting
    - convert split latex text to wrap with latex code for making the model to understand the formula
        - this is my hypothesis, Foundation model such as llama, gpt will be good at latex code to understand the formula

2) Sometimes, there are code snippets
    - use python code splitter for splitting
    - ???

3) Arxiv paper is the informative, official, public document. So it has a lot of rules for writing.
    - use RecursiveCharacterTextSplitter for splitting for using the rule of the public writing
    - So, Do not use the cleansing separator word such as \n, \n\n, \t ... etc

4) There are a lot of tables for visualizing the experimental results
    - use pdf partitioning strategy for splitting the table
    - convert pdf table to html code for making the model to understand the table

5)

"""
import os

from typing import List

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

from langchain.text_splitter import LatexTextSplitter
from langchain.text_splitter import KonlpyTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_by_length(
    text: str,
    chunk_size: int = 512,
    over_lapping_size: int = 128,
    separator: str = '',
    strip_whitespace: bool = True
) -> List[str]:
    """ wrapper function for langchain.CharacterTextSplitter, splitting method by character length

    Args:
        text: text to be split
        chunk_size: size of each chunk
        over_lapping_size: size of over lapping
        separator: separator to be used for splitting
        strip_whitespace: whether to strip whitespace or not

    Returns:
        List[Document]: list of documents, split by character length

    Reference:
        https://chunkviz.up.railway.app/
        https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py#L9

    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=over_lapping_size,
        separator=separator,
        strip_whitespace=strip_whitespace
    )
    return splitter.create_documents([text])


def chunk_by_recursive_search(
    text: str,
    chunk_size: int = 512,
    over_lapping_size: int = 128,
):
    """ For this text, 450 splits the paragraphs perfectly,
    You can even switch the chunk size to 469 and get the same splits,

    This is because this splitter builds in a bit of cushion
    and wiggle room to allow your chunks to 'snap' to the nearest separator.

    if you want to use this splitter, you do not cleansing separators word such as \n, \n\n, \t ... etc

    Args:
        text: text to be split
        chunk_size: size of each chunk
        over_lapping_size: size of over lapping

    Returns:
        List[Document]: list of documents, split by recursive search

    Reference:
        https://chunkviz.up.railway.app/
        https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py#L58
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=over_lapping_size,
    )
    return splitter.create_documents([text])


def chunk_by_latex():
    """  """
    splitter = LatexTextSplitter()


def chunk_by_html():
    pass


def chunk_by_markdown():
    pass


def chunk_by_python_code():
    pass


def chunk_by_hf_tokenizer():
    pass


def chunk_by_semantic():
    pass


def chunk_by_agent():
    pass


def chunk_for_korean():
    pass


def convert_pdf_table_to_html(path: str, strategy: str = "hi_res", model_name: str = "yolox"):
    """ extract table from pdf file and convert to html code

    Args:
        path: str, path to the pdf file
        strategy: str, strategy for partitioning
        model_name: str, model name for partitioning

    Reference:
        https://wikidocs.net/156108
        https://docs.unstructured.io/welcome
        https://docs.unstructured.io/open-source/concepts/partitioning-strategies
        https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    """
    elements = partition_pdf(
        filename=path,
        strategy=strategy,
        model_name=model_name,
        infer_table_structure=True,
    )
    for element in elements:
        print(element)

    return elements


if __name__ == '__main__':
    result = convert_pdf_table_to_html(
        path=r"./attention_is_all_you_need.pdf",
        strategy="hi_res",
        model_name="yolox"
    )


