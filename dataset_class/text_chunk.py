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
      (convert_pdf_table_to_html())
"""
import os
import re
import unstructured

from typing import List, Tuple, Any, Dict

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
    """ Wrapper function for langchain.RecursiveCharacterTextSplitter, splitting method by recursive search
    Recursive search means that the splitter will recursively search for the separator in the text.

    This splitter is useful for splitting text that has a lot of rules for writing such as paper, article.

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


def chunk_by_latex(
    text: str,
    chunk_size: int = 512,
    over_lapping_size: int = 128,
):
    """ Wrapper function for langchain.LatexTextSplitter, splitting method by latex

    Latex splitter is useful for splitting text that has a lot of mathematical formulas

    """
    splitter = LatexTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=over_lapping_size,
    )
    return splitter.create_documents([text])


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


def build_latex_code(text: str) -> str:
    """ convert python math string into latex style code for expressing mathematical formula in the document text chunk

    output of this function will be used for document of document embedding db

    example:
        inputs_string = "FFN(x) = max(0, xW1 + b1)W2 + b2"
        output_string = "FFN(x) = \text{max}(0, xW1 + b1)W2 + b2"

        => it comes from the extracted text from the pdf file,
        => so when it comes to convert process, some mathematical grammar of formula will be broken
        => this function can not convert perfectly, because input string will not be perfect math string

    """
    pattern = r'(\b\w+\b)(\d*)(?=\()'
    return re.sub(pattern, r'\\text{\1}_{\2}', text)


def build_html_table(tables: List[str], figure_captions: List[str]) -> List[str]:
    """ make html table with figure captions

    add figure captions to the each table html code string

    Args:
        tables: List[str], list of html table code
        figure_captions: List[str], list of figure captions

    Returns:
        List[str], list of html table code with figure captions

    """
    for i, data in enumerate(zip(figure_captions, tables)):
        caption, table = data
        idx = table.find("<table>") + len("<table>")
        table = table[:idx] + f"<caption>{caption}</caption>" + table[idx:]
        tables[i] = table

    return tables


def classify_document_element(
    elements: List
) -> Tuple[str, List[Any], List[str], str]:
    """ function for classifying document elements from the unstructured library

    Args:
        elements: List[unstructured.documents.elements.DocumentElement], list of document elements

    Returns:
        document: str, document text (body of the document)
        tables: List[Any], list of html tables code (experimental results)
        formulas: List[str], list of latex code for expressing formulas (mathematical formulas)
    """
    title_count, ref_count = 0, 0
    document, tables, formulas, references, figure_captions = "", [], [], "", []
    for i, e in enumerate(elements):
        if isinstance(e, unstructured.documents.elements.Table):
            tables.append(str(e.metadata.text_as_html))

        elif isinstance(e, unstructured.documents.elements.Title):
            if title_count:
                document += f"\n\n"

            if not ref_count and str(e) == "References":
                references += f"{str(e)}"
                continue

            document += str(e)
            title_count += 1

        elif isinstance(e, unstructured.documents.elements.NarrativeText):
            document += f"\n{str(e)}"

        elif isinstance(e, unstructured.documents.elements.Formula):
            formulas.append(build_latex_code(str(e)))

        elif isinstance(e, unstructured.documents.elements.ListItem):
            candidate = str(e)
            if candidate.startswith("["):
                references += f"\n{str(e)}"
                ref_count += 1

            else:
                document += f"\n{str(e)}"
        elif isinstance(e, unstructured.documents.elements.FigureCaption):
            candidate = str(e)
            if candidate.startswith("Table"):
                figure_captions.append(candidate)

    tables = build_html_table(tables, figure_captions)
    return document, tables, formulas, references


def cut_pdf_to_sub_module_with_text(path: str, strategy: str = "hi_res", model_name: str = "yolox") -> Dict[str, List[str]]:
    """ extract table from pdf file and convert to html code

    Args:
        path: str, path to the pdf file
        strategy: str, strategy for partitioning
        model_name: str, model name for partitioning

    Returns:
        tables (List[str]): list of html code for the tables in the pdf file

    Reference:
        https://wikidocs.net/156108
        https://docs.unstructured.io/welcome
        https://docs.unstructured.io/open-source/concepts/partitioning-strategies
        https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/documents/elements.py#L642
        https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    """
    elements = partition_pdf(
        filename=path,
        strategy=strategy,
        model_name=model_name,
        infer_table_structure=True,
    )
    document, tables, formulas, references = classify_document_element(elements)
    return {
        'text': document,
        'table': tables,
        'formula': formulas,
        'reference': references
    }


if __name__ == '__main__':
    result = cut_pdf_to_sub_module_with_text(
        path=r"./attention_is_all_you_need.pdf",
        strategy="hi_res",
        model_name="yolox"
    )
    print(f"text: {result['text']}")
    print(f"table: {result['table']}")
    print(f"formula: {result['formula']}")
    print(f"reference: {result['reference']}")

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
    for i, d in enumerate(documents):
        print(f"{i+1}-th document: {d}")
        print()
