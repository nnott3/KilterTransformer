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
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    idx = len(vocab)
     
     
    for angle in range(20, 61, 5):
        vocab[f"angle{angle}"] = idx
        idx += 1
    
    
    for grade in range(13, 28):
        vocab[f"grade{grade}"] = idx
        idx += 1
        
    # every combos. of hold_id and func
    func = ["feet", "start", "hand", "finish"]
    for hold_id in HOLD_ID:
        for f in func:
            vocab[f"{f}{hold_id}"] = idx
            idx += 1
    

    
    print(f"Built vocabulary with {len(vocab)} tokens ({len(vocab) - len(special_tokens)} holds)")
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

    inspect_tokenizer(tokenizer)

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
    print(f"Saving tokenizer to {output_dir}")
    tokenizer_pretrained.save_pretrained(output_dir)

    return tokenizer_pretrained


def inspect_tokenizer(tokenizer):
    vocab_list = list(tokenizer.get_vocab().items())
    print(f"\nVocab size: {len(vocab_list)} tokens")
    print("First 10 tokens:", vocab_list[:10])
    
    samples = [
        "angle35_grade14_feet1595_start1400",
        "angle40_grade15_feet1595_start1596_hand1597_finish1598",
    ]
    
    print("\nSample encodings:")
    for sample in samples:
        inspect_sample(tokenizer, sample)


def inspect_sample(tokenizer, input_str):
    def collapse_repeats(tokens):
        for token, group in groupby(tokens):
            count = len(list(group))
            yield (token, count) if count > 1 else token
    
    print(f"\nInput: {input_str}")
    encoded = tokenizer.encode(input_str)
    print(f"Tokens: {list(collapse_repeats(encoded.tokens))}")


if __name__ == "__main__":
    OUT_DIR = "models/climb_tokenizer"
    
    dp = DataPreprocessing()
    datasets = dp.load_climbs().train_test_split(test_size=0.2, seed=42)
    
    print(f"Train: {len(datasets['train'])}, Test: {len(datasets['test'])}")
    print(f"\nSample frame: {datasets['train'][0]['frames']}")
    
    tokenizer = train_tokenizer(datasets, OUT_DIR)
    
    print(f"\n✓ Tokenizer saved to {OUT_DIR}")