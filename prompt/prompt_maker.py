"""python module to generate prompts for text generation tasks,
this module specially focuses on generating prompts for question generation tasks
and retrieval augmented generation task
"""


def cut_context(cfg, context: str) -> str:
    """ function for cutting the context to make it shorter

    handling the problem that the context is too long to be used in the prompt,
    so question generation part of the context is cut and model doesn't generate questions for the whole context

    we use the tokenizer to cut the context to the maximum length of the mode and then decode it to the text

    Args:
        cfg: configuration object for the project, getting tokenizer object, other config values
        context: str, the context to be cut

    Returns:
        cut_context: str, the cut context
    """
    return cfg.tokenizer.decode(
        cfg.tokenizer(context, max_length=5120, truncation=True, add_special_tokens=False)
    )


def get_prompt_for_question_generation(context: str) -> str:
    """ return the prompt for question generation task
    prompt from this function is made by following the Few-Shot Prompt-Based Learning for Question Generation task

    Args:
        context: str, the context for which questions need to be generated

    Returns:
        prompt: str, the prompt for the question generation task
    """
    prompt = f"""Role Description:
You are a question-generating machine. Your goal is to generate questions based on given contexts, following specific guidelines.

Guidelines:

(Relevance) Ensure the question is directly related to the provided context.
(Clarity) The question should be clear, concise, and easily understood.
(Depth) Craft questions that probe the underlying features, implications, or complexities of the context. Avoid overly simplistic or general inquiries.
(Variety) Generate diverse questions that explore different aspects or perspectives of the context. Avoid redundancy.
(Format) Strictly adhere to the output format: "Question [Number]: [Your Question]". Limit the number of questions to 1-3.
(Contextual Understanding) Demonstrate a nuanced understanding of the context to generate questions that spark deeper thought or discussion.

Context 1:

Context:
"Extreme weather events due to climate change are increasing. This is causing significant damage to agricultural productivity."

Example Question 1:
"What are the main impacts of climate change on agricultural productivity?"

Context 2:

Context:
"As remote work becomes more widespread globally, companies are seeking effective collaboration tools. These tools play a crucial role in enhancing communication and cooperation among teams."

Example Question 2:
"Why are effective collaboration tools important in a remote work environment?"

Your Task:

For given context from Context 3, generate questions based on the specific guidelines provided.
Generate appropriate questions according to the given context from Context 3 only and guidelines.
Keep the number of questions between 1 and 3.
Separate the questions you want to create with newlines

Context 3:

Context:
{context}

Your Question:
=====
[Generate Your Question here]
"""
    return prompt


def get_prompt_for_retrieval_augmented_generation(title: str, context: str) -> str:
    pass