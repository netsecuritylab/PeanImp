import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import binascii
from typing import Dict, List, Tuple
from scapy.all import rdpcap, Raw, TCP, UDP

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time
import copy

from torch.optim import AdamW # Do this instead

from transformers import (
    CONFIG_MAPPING,
    #AdamW, #TODO: remove this everywhere as its deprecated
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    AutoModel,
)

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
MODEL_TYPES = tuple(CONFIG_MAPPING.keys())
MODEL_CONFIG_CLASSES = [CONFIG_MAPPING[m] for m in MODEL_TYPES]


global pretrain_model_name

def readPcap(pcap_path, output_txt):
    packets = rdpcap(pcap_path)

    logger.info(f"Converting pcap file at {pcap_path} into the specified .txt")
    with open(output_txt, "a", encoding="utf-8") as f:
        for pkt in packets:
            # Start from transport layer if available, otherwise use Raw
            transport_data = b""
            
            if pkt.haslayer(TCP):
                transport_data = bytes(pkt[TCP])
            elif pkt.haslayer(UDP):
                transport_data = bytes(pkt[UDP])
            
            # Add Raw payload if present
            raw_data = b""
            if pkt.haslayer(Raw):
                raw_data = bytes(pkt[Raw])
            
            # Combine both layers if we have data
            if transport_data or raw_data:
                combined = transport_data + raw_data
                hex_bytes = binascii.hexlify(combined).decode("utf-8")
                tokens = " ".join(hex_bytes[i:i+2] for i in range(0, len(hex_bytes), 2))
                f.write(tokens + "\n")

def readPcap_folder(folder_path, output_txt):
    for folder in os.listdir(folder_path):
        path = os.path.join(folder_path, folder)
        if not(os.path.isfile(path)):
            for file in os.listdir(path):
                readPcap(os.path.join(path, file), output_txt)

# TODO: remove this new argument
def param_setup(new):
    pretrain_model_name = "model_128d_8h_2l"
    TRAIN_FILE = "./TrafficData/pretrain_train.txt"
    EVAL_FILE = "./TrafficData/pretrain_test.txt"
    output_dir = "./Model/pretrain/" + pretrain_model_name
    config_name = "./Config/pretrain_config.json"
    tokenizer_name = "./Config/vocab.txt"

    if not os.path.exists(output_dir):
        print(os.getcwd())
        os.mkdir(output_dir)

    train_data_file = TRAIN_FILE
    eval_data_file = EVAL_FILE

    model_type = "bert"
    model_name_or_path = None if new else './Model/pretrain/model_128d_8h_2l/' #TODO: default is None
    do_train = True
    do_eval = True
    do_test = True
    do_fune_tune = False
    overwrite_output_dir = True
    overwrite_cache = False
    seed_flag = True
    evaluate_during_training = False
    each_epoch_eval = True
    each_checkpoint_eval = False
    each_batch_eval = False
    mlm = True
    line_by_line = False
    max_seq_length = 400 # default is 400
    per_gpu_train_batch_size = 128 # default is 128
    per_gpu_eval_batch_size = 128
    per_gpu_test_batch_size = 128
    learning_rate = 1e-3
    warmup_proportion = 0.01 # default is 0.1, changing it for debug
    num_train_epochs = 100
    logging_steps = -1
    save_steps = -1
    gpu_start = 0
    gpu_num = torch.cuda.device_count()
    read_pcap = False
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pcap_folder", default=None, type=str, help="path to folder containing the pcap files"
    )
    parser.add_argument(
        "--pcap_out", default='./test.txt', type=str, help="path where the pretrain .txt dataset will be written"
    )
    parser.add_argument(
        "--read_pcap", default=read_pcap, type=str, help="Whether to create a dataset from a pcap folder"
    )
    parser.add_argument(
        "--train_data_file", default=train_data_file, type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--each_epoch_eval", default=each_epoch_eval, type=str,
    )
    parser.add_argument(
        "--each_batch_eval", default=each_batch_eval, type=str,
    )
    parser.add_argument(
        "--each_checkpoint_eval", default=each_checkpoint_eval, type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, default=model_type, help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--gpu_start", default=gpu_start, type=int,
    )
    parser.add_argument(
        "--gpu_num", default=gpu_num, type=int,
    )
    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=eval_data_file,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        default=line_by_line,
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=model_name_or_path,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", default=mlm, action="store_true",
        help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=config_name,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=tokenizer_name,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=max_seq_length,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", default=do_train, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default=do_eval, action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_fune_tune", default=do_fune_tune, action="store_true")
    parser.add_argument(
        "--evaluate_during_training", default=evaluate_during_training, action="store_true",
        help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=per_gpu_train_batch_size, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=per_gpu_eval_batch_size, type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=learning_rate, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=num_train_epochs, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=warmup_proportion, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=logging_steps, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=save_steps, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", default=overwrite_output_dir, action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", default=overwrite_cache, action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument('--device', default='cuda:0', type=str, help='the training device') # change to cuda:0 
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    args.n_gpu = 1
    device = args.device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    if seed_flag:
        set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
    if args.config_name:
        config = AutoConfig.for_model(args.model_type).from_json_file(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        Tokenizer = BertTokenizer
        tokenizer = Tokenizer(vocab_file=args.tokenizer_name, max_seq_length=args.block_size - 2,
                              max_len=args.block_size)

    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if args.model_name_or_path:
        logger.info("Extracting pretrained model")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        if args.do_fune_tune:
            # model = AutoModelWithLMHead.from_pretrained(args.output_dir)
            model = AutoModel.from_pretrained(args.output_dir)
            # model = finetune_cls(model)
        else:
            # model = AutoModelWithLMHead.from_pretrained(os.path.join(args.output_dir, "checkpoint-50000"))
            model = AutoModelForCausalLM.from_config(config)
            # model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
            # model = BertForMaskedLM(config=config)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) # this was in main of preTrain
    model.to(args.device)
    return args, tokenizer, model

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

    
class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            logger.info("Saving features at %s", cached_features_file)
            #CLS, SEP = '[CLS]', '[SEP]'
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.readlines()
                tokenized_text = []
                for line in text:
                    line = line.strip().split(' ')
                    if len(line) > block_size:
                        line = line[:block_size]
                    if len(line)==1:  # 去掉空字符
                        continue
                    tokenized_line = tokenizer.convert_tokens_to_ids(line)
                    tokenized_text.append(tokenized_line)
                    # print(len(tokenized_line))
            """
            tokenized_text : list, [[id1, id2, ..., ], [id1, id2, ..., ], ..., [id1, id2, ..., ]], 
            self.examples : list, [[cls_id, id1, id2, ..., seq_id], [cls_id, id1, id2, ..., seq_id], ...], 
            """
            for i in range(len(tokenized_text)):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i]))

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    '''
    Takes path of text data and treats each line as a flow example (only for pretrain so no sequence length or label)
    '''
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)
    

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    """
    inputs ：tensor, [[cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], [cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], ...], 
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    """
    probability_matrix : tensor, [[0.15, ..., 0.15], [0.15, ..., 0.15], ...]
    """
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    """
    special_tokens_mask : list, [[1,0,0,...,1,0,0,...], [1,0,0,...,1,0,0,...], ....]
    """
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    if tokenizer.pad_token is not None: # current transformers _pad_token-> pad_token
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    """
    masked_indices : tensor, [[False, False, ..., True, ..., ], [False, False, ..., True, ..., ], ...]
    labels ：tensor, [[-100, id1, -100, ..., idn, -100,..., -100], [-100, id1, -100, ..., idn, -100,..., -100], ...]
    """
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:

    pretrain_model_name = "model_128d_8h_2l"

    def collate_fn(batch):
        # is given to dataloader in order to pad batches on the go
        # rather than all at once
        return pad_sequence(
            batch,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    """
    train_dataset : list, [[cls_id, id1, id2, ..., sep_id], [cls_id, id1, id2, ..., sep_id], ...]
    train_dataset_pad ： tensor, [[cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], [cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], ...]
    """
    # didn't work as pad_sequence expects lists of iterables and dataset is a separate class
    # included padsequence in dataloader which is also more efficent
    # train_dataset_pad = pad_sequence(train_dataset, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_proportion * t_total,
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[args.gpu_start, args.gpu_start + 1, args.gpu_start + 2,
                                                         args.gpu_start + 3])

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()

    train_iterator = range(
        epochs_trained, int(args.num_train_epochs)
    )

    set_seed(args)  # Added here for reproducibility
    best_eval_loss = 9e8

    for e in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        nb_tr_steps = 0
        tr_loss = 0.0
        time_inter = 0.0
        batch_step = 0
        batch_steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            
            # If masking happens on gpu it should be faster
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            # We need this otherwise pad tokens would feed into attention
            # wasn't in original paper
            attention_mask = (inputs != tokenizer.pad_token_id).long() # Creating it from tensors already in gpu puts it in gpu

            model.train()
            # in current version of transformers masked_mlm_labels is unified with labels argument
            outputs = model(inputs, attention_mask=attention_mask, labels=labels) if args.mlm else model(inputs, labels=labels)


            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                end_time = time.time()
                time_inter = time_inter + end_time - start_time
                if args.each_batch_eval:
                    eval_result = evaluate(args, model, tokenizer)

                    batch_step += 1
                    if step != batch_steps - 1:
                        print(
                            "\r============================ -epoch %d[%d/%d] -train_loss %.4f -eval_loss %.4f -train_batch_spend_time %.4fs" %
                            (e, batch_step, batch_steps, tr_loss / nb_tr_steps, eval_result['eval_loss'], time_inter),
                            end="", flush=True)
                    else:
                        print(
                            "\r============================ -epoch %d[%d/%d] -train_loss %.4f -eval_loss %.4f -train_batch_spend_time %.4fs\n" %
                            (e, batch_step, batch_steps, tr_loss / nb_tr_steps, eval_result['eval_loss'], time_inter),
                            end="", flush=True)
                else:
                    batch_step += 1
                    if step != batch_steps - 1:
                        print(
                            "\r============================ -epoch %d[%d/%d] -train_loss %.4f -train_batch_spend_time %.4fs" %
                            (e, batch_step, batch_steps, tr_loss / nb_tr_steps, time_inter), end="", flush=True)
                    else:
                        print(
                            "\r============================ -epoch %d[%d/%d] -train_loss %.4f -train_batch_spend_time %.4fs\n" %
                            (e, batch_step, batch_steps, tr_loss / nb_tr_steps, time_inter), end="", flush=True)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.each_epoch_eval:
            eval_result = evaluate(args, model, tokenizer)

            train2eval_loss = tr_loss / nb_tr_steps
            if e == 0:
                with open(os.path.join(args.output_dir, "train_eval_loss.txt"), "w", encoding="utf-8") as f:
                    f.write("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" %
                            (e, train2eval_loss, eval_result['eval_loss']))
            else:
                with open(os.path.join(args.output_dir, "train_eval_loss.txt"), "a", encoding="utf-8") as f:
                    f.write("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" %
                            (e, train2eval_loss, eval_result['eval_loss']))
            logger.info("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" %
                        (e, train2eval_loss, eval_result['eval_loss']))

        if best_eval_loss > eval_result['eval_loss']:
            best_eval_loss = eval_result['eval_loss']
            best_train_loss = train2eval_loss
            best_epoch = e
            best_model = copy.deepcopy(model)
            output_dir = args.output_dir
            print(f"saving model at {output_dir} ..............")
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                best_model.module if hasattr(best_model, "module") else best_model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(best_model.state_dict(), os.path.join(output_dir, pretrain_model_name + ".pth"))

        if args.max_steps > 0 and global_step > args.max_steps:
            # train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # output_dir = os.path.join(args.output_dir, "epoch-{}-loss-{}-model".format(best_epoch, round(best_eval_loss, 4)))
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        best_model.module if hasattr(best_model, "module") else best_model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(best_model.state_dict(), os.path.join(output_dir, pretrain_model_name+".pth"))
    # best_tokenizer.save_pretrained(output_dir)
    # torch.save(best_args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving epoch-{}-loss-{:.4f}-model to %s".format(best_epoch, best_eval_loss), output_dir)

    return global_step, best_epoch, best_eval_loss, best_train_loss


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu) # default eval_batch_size = 128
    
    def collate_fn(batch):
        # is given to dataloader in order to pad batches on the go
        # rather than all at once
        return pad_sequence(
            batch,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )

    # eval_dataset_pad = pad_sequence(eval_dataset, batch_first=True, padding_value=tokenizer.pad_token_id)
    # same thing that happened in train done here with collate fn
    eval_sampler = RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn
    )
    set_seed(args)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        eval_loss += loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "eval_loss": eval_loss}

    return result

def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)