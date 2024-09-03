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
        cfg.tokenizer.encode(context, max_length=5120, truncation=True, add_special_tokens=False),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )


def get_prompt_for_question_generation(context: str) -> str:
    """ return the prompt for question generation task
        prompt from this function is made by following the Few-Shot Prompt-Based Learning for Question Generation task

        Args:
            context: str, the context for which questions need to be generated

        Returns:
            prompt: str, the prompt for the question generation task
        """
    prompt = f"""Role Description: You are a question-generating machine.
Your goal is to generate questions based on given contexts.
Refer to the examples below for guidance.

Context 1: Effective positional interpolation should consider two forms of non-uniformity: varying RoPE dimensions and token positions.
Lower RoPE dimensions and initial starting token positions benefit from less interpolation, but the optimal solutions depend on the target extended length.
By considering these non-uniformity into positional interpolation, we can effectively retain information in the original RoPE, particularly key dimensions and token positions.
This minimizes the loss caused by positional interpolation, and thus provides better initialization for fine-tuning.
Moreover, it allows an 8× extension in non-fine-tuning scenarios.

Question 1: What is positional interpolation in the context of LLMs, and why is it important?
How do varying RoPE dimensions and token positions affect the need for interpolation?

Context 2: We observe that the weights of LLMs are not equally important: there is a small fraction of salient weights that are much more important for LLMs’ performance compared to others.
Skipping the quantization of these salient weights can help bridge the performance degradation due to the quantization loss without any training or regression.
Interestingly, selecting weights based on activation magnitude can significantly improve the performance despite keeping only 0.1%-1% of channels in FP16.
We hypothesize that the input features with larger magnitudes are generally more important.
Keeping the corresponding weights in FP16 can preserve those features, which contributes to better model performance.

Question 2: What is the rationale behind the idea that only a small fraction of salient weights are crucial for LLM performance?
How does the preservation of weights with higher activation magnitudes improve model performance?
How does keeping 0.1%-1% of channels in FP16 significantly improve the performance of quantized models?

Context 3: PagedAttention, an attention algorithm inspired by the classic idea of virtual memory and paging in operating systems.
Unlike the traditional attention algorithms, PagedAttention allows storing continuous keys and values in non-contiguous memory space.
Specifically, PagedAttention partitions the KV cache of each sequence into blocks, each block containing the keys and values for a fixed number of tokens.
During the attention computation, the PagedAttention kernel identifies and fetches these blocks efficiently.

Question 3: What is PagedAttention, and how does it differ from traditional attention algorithms?
How does the idea of virtual memory and paging in operating systems inspire PagedAttention?
How does PagedAttention partition the KV cache of each sequence?

Your Task: For the given text from Context 4, generate questions based on the specific guidelines provided.
Keep the number of questions between 1 and 3.
Do not use bullet points or numerical indicators (e.g., "-", "1.", "2.", "Question", "Context") to start your questions.
Separate the questions you want to create with newlines.
Always end individual questions with a question mark.
Context 4: {context}
Question 4: """

    return prompt


def get_prompt_for_retrieval_augmented_generation(title: str, context: str) -> str:
    pass