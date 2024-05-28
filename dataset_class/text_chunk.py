import os

from typing import List
from langchain import Document

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
) -> List[Document]:
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


def convert_pdf_table_to_html():
    pass

