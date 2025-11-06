import pprint
from itertools import groupby
import os

from src.data_processing import DataPreprocessing
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

def build_vocab():
    from src.data_processing import HOLD_ID

    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    vocab = {t: i for i, t in enumerate(special_tokens)}
    idx = len(vocab)

    # angles + grades
    angle_list = [f"angle{a}" for a in range(20, 61, 5)]
    grade_list = [f"grade{g}" for g in range(13, 28)]
    for token in angle_list + grade_list:
        vocab[token] = idx
        idx += 1

    # holds -> every combo of hold_id and func, regardless of usage/data
    for f in ["feet", "start", "hand", "finish"]:
        for h in HOLD_ID:
            vocab[f"{f}{h}"] = idx
            idx += 1

    print(f"Built vocab with {len(vocab)} tokens ({len(vocab) - len(special_tokens)} custom)")
    return vocab

def train_tokenizer(datasets, output_dir, max_length=25):
    """Train and return a tokenizer for the given datasets."""
    special_tokens = {
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        }

    # Build pre-defined vocabulary
    vocab = build_vocab()

    # Initialize tokenizer with vocab
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=special_tokens["unk_token"]))
    tokenizer.enable_padding(length=max_length, pad_token=special_tokens["pad_token"])
    tokenizer.enable_truncation(max_length=max_length)

    # Split on underscores and whitespace
    # since ['frames'] is "angle30_grade22_start1136_feet1169_hand1234_hand1253_hand1353_finish1391_feet1453"
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"_"), behavior="removed"),
        pre_tokenizers.Whitespace()
    ])

    bos_token_id = tokenizer.token_to_id(special_tokens["bos_token"])
    eos_token_id = tokenizer.token_to_id(special_tokens["eos_token"])

    tokenizer.post_processor = TemplateProcessing(
        single=special_tokens["bos_token"] + " $A " + special_tokens["eos_token"],
        special_tokens=[
            (special_tokens["bos_token"], bos_token_id),
            (special_tokens["eos_token"], eos_token_id),
        ],
    )

    inspect_tokenizer(tokenizer) # visualizign

    # Convert to PreTrainedTokenizerFast
    tokenizer_pretrained = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=max_length,
        padding_side="right",
        truncation_side="right",
        **special_tokens
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving tokenizer to {output_dir}")
    tokenizer_pretrained.save_pretrained(output_dir)

    return tokenizer_pretrained

def inspect_tokenizer(tokenizer):
    vocab = tokenizer.get_vocab()
    print(f"\nVocab size: {len(vocab)} tokens")
    print("First 10 tokens:", list(vocab.items())[:10])

    samples = [
        "angle35_grade14_feet1595_start1400",
        "angle40_grade15_feet1595_start1596_hand1597_finish1598",
    ]

    print("\nSample encodings:")
    for s in samples:
        encoded = tokenizer.encode(s)
        grouped = [
            f"{tok}x{n}" if (n := len(list(g))) > 1 else tok
            for tok, g in groupby(encoded.tokens)
        ]
        print(f"\nInput:  {s}")
        print("Tokens:", grouped)

def tokenize_datasets(datasets, tokenizer):
    def tokenize_function(examples):
        tok = tokenizer(examples["frames"], truncation=True, padding=True)
        ignore_ids = [tokenizer.pad_token_id, tokenizer.bos_token_id]
        tok["labels"] = [
            [-100 if t in ignore_ids else t for t in ids]
            for ids in tok["input_ids"]
        ]
    
        return tok

    for name in ("train", "val", "test"):
        if name in datasets:
            datasets[name] = datasets[name].map(
                tokenize_function, batched=True, remove_columns=datasets[name].column_names
            )
    return datasets



if __name__ == "__main__":
    OUT_DIR = "models/climb_tokenizer"

    dp = DataPreprocessing()
    datasets = dp.load_climbs().train_test_split(test_size=0.2, seed=42)

    print(f"Train: {len(datasets['train'])}, Test: {len(datasets['test'])}")
    print(f"\nSample frame: {datasets['train'][0]['frames']}")

    tokenizer = train_tokenizer(datasets, OUT_DIR)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    print(f"Before preprocess_datasets:\n  {datasets['train']}")
    datasets = tokenize_datasets(datasets, tokenizer)
    print(f"\nAfter:\n  {datasets['train']}")