import torch


class CFG:
    """ Base Configuration Class for various experiments
    this Module has all of hyper-parameters for whole this project such as training, model, data, optimizer, scheduler, loss, metrics, ... etc

    You can change hyper-parameters easily by changing json file in configuration data_folder,
    each json file has same name with model name, so you can easily find hyper-parameters for each model

    And then, individual JSON files adjust only the hyperparameters needed for individual experiments

    param:
        wandb (bool): flag variable of weight and bias tracking service

        seed (int): value of random seed
        n_gpu (int): number of gpu in current workflow
        num_workers (int): number of worker in torch dataloader process, same as number of multi-process
        device (torch.device):

        n_jobs (int): number of workers in text-chunking process (PDF to Python string)
        pipeline_type (str): flag variable of setting the current workflow (make, insert, fine_tune, inference)
        work_flow_state (str): flag variable for "make" pipeline_type
        loop (str): name of train loop function
        name (str): name of current task
        trainer (str): name of trainer module

        dataset (str): name of dataset module
        tokenizer (AutoTokenizer): pretrained tokenizer module of pretrained model
        model_name (str): name or path of model (local disk path or huggingface model hub)
        task (str): name of task module (in this project, model.model.py)
        pooling (str): name of pooling method

        pow_value (str): value of power ops, especially using in GEMPooling (min <= gem <= max)
        checkpoint_dir (str): save path of model
        sampler (torch.nn.Module):
        collator (torch.nn.Module):
        split_ratio (float): split ratio of train & validation dataset
        batching (str): batching strategy of dataloader (options: random, smart)
                        (smart strategy will batch the data by similar sequence length for reducing padding tokens

        amp_scalar (bool): flag variable of using mixed precision training in pytorch
        clipping_grad (bool): flag variable of using clipping gradient norm
        gradient_checkpoint (bool): flag variable of using gradient checkpointing method for reducing memory consuming
        max_grad_norm (int): maximum size of total gradient norm in each training steps
        n_gradient_accumulation (int): interval value of updating weight by backward gradient
        patience (int): remaining value of early stopping
        stop_mode (str): loss based metric will be set to "min", others will be set to "max"

        #freeze (bool): flag variable of using freeze model param
        #num_freeze (bool):

    """
    ##########################################################################################################

    wandb = True
    seed = 42
    n_gpu = 1
    gpu_id = 0
    num_workers = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ##########################################################################################################

    n_jobs = 5
    pipeline_type = "insert"
    work_flow_state = "resume"

    loop = "train_loop"
    name = "MetricLearningModel"
    trainer = "MetricLearningTuner"
    dataset = "QuestionDocumentMatchingDataset"

    ##########################################################################################################

    tokenizer = None
    model_name = "intfloat/e5-large-v2"
    task = "MetricLearningModel"
    pooling = "MeanPooling"
    pow_value = 1
    checkpoint_dir = "./saved/"

    ##########################################################################################################

    sampler = None
    collator = None
    split_ratio = 0.1
    batching = "random"

    ##########################################################################################################

    amp_scaler = True
    clipping_grad = True
    gradient_checkpoint = True

    max_grad_norm = 1
    n_gradient_accumulation_steps = 1

    patience = 3
    stop_mode = "min"
    freeze = False
    num_freeze = 0
    reinit = False
    num_reinit = 0

    max_len = 512
    epochs = 10
    batch_size = 64
    smart_batch = False
    val_check = 1000  # setting for validation check frequency (unit: step)

    optimizer = "AdamW"  # options: SWA, AdamW
    llrd = False
    layerwise_lr = 5e-5
    layerwise_lr_decay = 0.9

    lr = 5e-5
    weight_decay = 1e-2
    adam_epsilon = 1e-6
    use_bertadam = False
    betas = (0.9, 0.999)

    num_labels = 2
    vocab_size = None
    layer_norm_eps = 1e-7
    initializer_range = 0.02
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    init_weight = "orthogonal"  # options: normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal

    ##########################################################################################################

    reduction = "mean"
    loss_fn = "CrossEntropyLoss"  # single loss function
    val_loss_fn = "CrossEntropyLoss"

    metrics = []
    losses_fn = []  # multiple loss function
    val_losses_fn = []

    ##########################################################################################################

    scheduler = "cosine_annealing"  # options: cosine, linear, cosine_annealing, linear_annealing
    batch_scheduler = True
    num_cycles = 0.5  # num_warmup_steps = 0
    warmup_ratio = 0.1  # if you select per step, you should consider size of epoch

    ##########################################################################################################

    quantization = None  # pass the quantization options (QLoRA, AWQ, LLM.int8())
    lora = False
    lora_rank = 8
    lora_alpha = 32
    lora_dropout = 0.1
    task_type = "None"
    prompt_tuning = False
    prompt_tuning_type = "P-TUNING"
    encoder_reparameterization_type = "LSTM"
    num_virtual_tokens = 20
    virtual_token_dim = 768
    prompt_encoder_hidden_size = 768

    ##########################################################################################################

    inference_pipeline = None  # options: "tensorrt_llm", "vllm", "huggingface"
    max_new_tokens = 512
    strategy: str = "beam"
    penalty_alpha: float = 0.6 if strategy == "contrastive" else None
    num_beams: int = 4 if strategy == "beam" else None
    temperature: float = None,
    top_k: int = None,
    top_p: float = None,
    repetition_penalty: float = None,
    length_penalty: float = None,
    no_repeat_ngram_size: int = None,
    do_sample: bool = False,
    use_cache: bool = True,

    ##########################################################################################################
