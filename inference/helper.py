from transformers import AutoTokenizer


def cut_context(tokenizer: AutoTokenizer, context: str) -> str:
    """ function for cutting the context to make it shorter

    handling the problem that the context is too long to be used in the prompt,
    so question generation part of the context is cut and model doesn't generate questions for the whole context

    we use the tokenizer to cut the context to the maximum length of the mode and then decode it to the text

    Args:
        tokenizer (AutoTokenizer): tokenizer from transformers
        context (str): the context to be cut

    Returns:
        cut_context: str, the cut context
    """
    return tokenizer.decode(
        tokenizer.encode(context, max_length=5120, truncation=True, add_special_tokens=False),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )


def get_prompt_for_answering_question(query: str, candidates: str) -> str:
    """ return the complete prompt template for answering the question from user's queries
    prompt from this function is made by following the Few-Shot Prompt-Based Learning for answering the questions

    Args:
        query (str): question from each user
        candidates (str): concatenated candidates document for answering the question from users more precisely

    Returns:
        prompt: str, the prompt for the question generation task
    """
    prompt = f"""Role Description: You are an answering assistant.
Your task is to generate detailed, accurate, and concise answers based on the provided contexts.
Use the examples below as a guide for answering questions, ensuring that your responses directly address the query while incorporating key information from the context.

Answer Structure: Begin with a clear explanation of the concept or question.
Incorporate specific insights from the context.
Conclude with an application or deeper explanation, if relevant, to clarify or expand the answer.

Example:
Question 1: What is positional interpolation in the context of LLMs, and why is it important?
How do varying RoPE dimensions and token positions affect the need for interpolation?

Context 1: Effective positional interpolation should consider two forms of non-uniformity: varying RoPE dimensions and token positions.
Lower RoPE dimensions and initial starting token positions benefit from less interpolation, but the optimal solutions depend on the target extended length.
By considering these non-uniformity into positional interpolation, we can effectively retain information in the original RoPE, particularly key dimensions and token positions.
This minimizes the loss caused by positional interpolation, and thus provides better initialization for fine-tuning.
Moreover, it allows an 8× extension in non-fine-tuning scenarios.

Answer 1: Positional interpolation in the context of Large Language Models (LLMs) refers to the method of adapting positional encoding, such as Rotary Position Embeddings (RoPE), when extending the sequence length beyond the model's training limits.
This is important because LLMs rely heavily on positional information to interpret the order and structure of input tokens.
When extending sequence lengths, simply stretching the positional encodings can lead to a loss of information, reducing model performance.
Varying RoPE dimensions and token positions affect the need for interpolation because lower RoPE dimensions and tokens closer to the start of a sequence (initial positions) require less interpolation.
This is because those dimensions and positions are already well-optimized to retain crucial positional information.
As token positions move further out in extended sequences, interpolation becomes more necessary.
By carefully adjusting for this non-uniformity, especially when targeting extended lengths, information loss is minimized, leading to better fine-tuning and extended sequence processing without fine-tuning.

Question 2: What is the rationale behind the idea that only a small fraction of salient weights are crucial for LLM performance?
How does the preservation of weights with higher activation magnitudes improve model performance?
How does keeping 0.1%-1% of channels in FP16 significantly improve the performance of quantized models?

Context 2: We observe that the weights of LLMs are not equally important: there is a small fraction of salient weights that are much more important for LLMs’ performance compared to others.
Skipping the quantization of these salient weights can help bridge the performance degradation due to the quantization loss without any training or regression.
Interestingly, selecting weights based on activation magnitude can significantly improve the performance despite keeping only 0.1%-1% of channels in FP16.
We hypothesize that the input features with larger magnitudes are generally more important.
Keeping the corresponding weights in FP16 can preserve those features, which contributes to better model performance.

Answer 2: The rationale behind the idea that only a small fraction of salient weights is crucial for LLM performance is that not all model parameters contribute equally to model accuracy.
A small set of weights, deemed salient, have a disproportionately large impact on performance, while many other weights can be compressed or quantized with minimal effect on the overall model behavior.
Preserving weights with higher activation magnitudes improves model performance because these weights correspond to input features that have more influence on the model’s outputs.
By keeping these critical weights in higher precision (FP16), the model is better able to capture important features, which maintains accuracy even when other parts of the model are quantized.
Keeping 0.1%-1% of channels in FP16 can significantly improve the performance of quantized models because these channels correspond to the most important or impactful ones.
This selective preservation mitigates the quantization loss, as the most critical information pathways are maintained in higher precision, allowing the model to function closer to its original performance.

Question 3: What is PagedAttention, and how does it differ from traditional attention algorithms?
How does the idea of virtual memory and paging in operating systems inspire PagedAttention?
How does PagedAttention partition the KV cache of each sequence?

Context 3: PagedAttention, an attention algorithm inspired by the classic idea of virtual memory and paging in operating systems.
Unlike the traditional attention algorithms, PagedAttention allows storing continuous keys and values in non-contiguous memory space.
Specifically, PagedAttention partitions the KV cache of each sequence into blocks, each block containing the keys and values for a fixed number of tokens.
During the attention computation, the PagedAttention kernel identifies and fetches these blocks efficiently.

Answer 3: PagedAttention is an attention algorithm for LLMs that draws inspiration from virtual memory and paging concepts in operating systems.
Unlike traditional attention algorithms that store keys and values in a contiguous memory space, PagedAttention allows for the storage of continuous keys and values in non-contiguous memory blocks.
This approach enhances memory efficiency, especially for large-scale models handling long sequences.
The idea of virtual memory and paging in operating systems inspires PagedAttention by introducing the concept of dividing the memory space (in this case, the KV cache) into separate, non-contiguous blocks that can be fetched and utilized dynamically during computation.
In operating systems, paging allows the system to manage large amounts of data by splitting it into manageable units, and PagedAttention applies this idea to attention computation.
PagedAttention partitions the KV cache of each sequence into blocks, with each block containing keys and values for a fixed number of tokens.
During attention computation, these blocks are efficiently identified and fetched as needed, rather than relying on a contiguous memory structure.
This allows for more flexible and scalable memory usage.

Template for Answering: Detailed answer based on the context, ensuring all parts of the query are addressed.
Summarize the key points and provide clarity on the topic, as shown in the examples

Question 4: {query}

Context 4: {candidates}

Answer 4: """
    return prompt


def apply_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    """ wrapper function for AutoTokenizer.apply_chat_template()

    Args:
        tokenizer (AutoTokenizer):
        prompt (str):
    """
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
