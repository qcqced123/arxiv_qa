"""python module for postprocessing to generated output of LLM
"""


def is_valid_question(text: str) -> bool:
    """ bool type function for validating current generative result
    Args:
        text (str): text for checking
    """
    return text.endswith("?") and not text.startswith("Question") and not text.startswith("Context")


def slice_full_questions(output: str) -> str:
    """ post-process function for generating questions from chunked Arxiv paper

    Args:
        output (str):

    """
    return " ".join([sub for sub in output.split("\n") if is_valid_question(sub)])


if __name__ == '__main__':
    case1 = 'How does the "Online PGD Attack with Noise Mult. m of zero point zero08 and a target WER of twenty point five improve the performance of the model?\nHow does the "Ours Attack with Noise Mult. m of zero point zero08 and a target W'
    case2 = 'How does the attack generalize to real-world settings without factoring in the room impulse response function?\nHow are different acoustic environments, with varying reverberation and ambient noise, effectively target by the attack?'

    print(f"This case will be sliced by function: {slice_full_questions(case1)}", end="\n\n")
    print(f"This case will not be sliced by function: {slice_full_questions(case2)}")
