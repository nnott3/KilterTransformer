"""
Climbing route generation model - training and inference.
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from typing import List, Tuple
import re


class KilterGPT(nn.Module):
    """GPT-2 model for generating Kilter Board climbing routes."""

    def __init__(
        self, 
        vocab_size: int,
        n_embd: int = 192,
        n_head: int = 3,
        n_layer: int = 3,
        n_positions: int = 128,
        dropout: float = 0.1
        ):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_positions=n_positions,
            n_ctx=n_positions,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.model = GPT2LMHeadModel(config)
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        
        if labels is not None:
            # Multi-label loss: any remaining token is valid
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = 0
            count = 0
            for i in range(shift_labels.size(0)):
                for j in range(shift_labels.size(1)):
                    # Get all valid next tokens (remaining holds)
                    valid_tokens = shift_labels[i, j:]
                    valid_tokens = valid_tokens[valid_tokens != -100]
                    
                    if len(valid_tokens) > 0:
                        log_probs = F.log_softmax(shift_logits[i, j], dim=-1)
                        valid_log_probs = log_probs[valid_tokens]
                        loss -= torch.logsumexp(valid_log_probs, dim=0)
                        count += 1
            
            outputs.loss = loss / count if count > 0 else loss
        
        return outputs

    def generate_route(
        self,
        tokenizer: PreTrainedTokenizerFast,
        angle: int = 40,
        grade: int = 18,
        max_length: int = 60,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        device: str = "cpu",
        logits_processor = None,
        ) -> List[str]:
        self.model.eval()
        self.model.to(device)

        angle_rounded = max(20, min(60, round(angle / 5) * 5))
        grade = max(13, min(27, grade))

        prompt = f"angle{angle_rounded} grade{grade} "
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor,
                repetition_penalty=1.2,
            )

        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    
    def validate_route(self, route_str: str) -> Tuple[bool, str]:
        pattern = r"angle(\d+)\s+grade(\d+)\s+(.*)"
        match = re.match(pattern, route_str)

        if not match:
            return False, "Invalid route format"

        angle, grade, holds_str = match.groups()
        angle, grade = int(angle), int(grade)

        if not (20 <= angle <= 60):
            return False, f"Invalid angle: {angle}"
        if not (13 <= grade <= 27):
            return False, f"Invalid grade: {grade}"

        holds = holds_str.split()
        num_start = sum(1 for h in holds if h.startswith("start"))
        num_finish = sum(1 for h in holds if h.startswith("finish"))
        num_hand = sum(1 for h in holds if h.startswith("hand"))
        num_feet = sum(1 for h in holds if h.startswith("feet"))

        if not (1 <= num_start <= 2):
            return False, f"Must have 1-2 start holds, got {num_start}"
        if not (1 <= num_finish <= 2):
            return False, f"Must have 1-2 finish holds, got {num_finish}"
        if len(holds) >= 20:
            return False, f"Too many holds: {len(holds)}"
        if num_hand == 0 and num_feet == 0:
            return False, "Must have at least one hand or feet hold"

        return True, ""


def load_model(model_path: str, device: str = "cpu") -> KilterGPT:
    gpt2_model = GPT2LMHeadModel.from_pretrained(model_path)
    model = KilterGPT(vocab_size=gpt2_model.config.vocab_size)
    model.model = gpt2_model
    model.config = gpt2_model.config
    model.to(device)
    model.eval()
    return model

# def tokenize_dataset(dataset, tokenizer):
#     """Tokenize frames column."""
#     return dataset.map(lambda example: tokenizer(example["frames"]), batched=True)

# def preprocess_datasets(datasets, tokenizer):
#     """Tokenize and remove original columns."""
#     for name in ("train", "val", "test"):
#         col_names = datasets[name].column_names
#         datasets[name] = tokenize_dataset(datasets[name], tokenizer).remove_columns(col_names)
#     return datasets

def preprocess_datasets(datasets, tokenizer):
    def tokenize_function(examples):
        tok = tokenizer(examples["frames"], truncation=True, padding=False)
        tok["labels"] = [
            [-100] + ids[1:] if ids and ids[0] == tokenizer.bos_token_id else ids
            for ids in tok["input_ids"]
        ]
        return tok

    for name in ("train", "val", "test"):
        if name in datasets:
            datasets[name] = datasets[name].map(
                tokenize_function, batched=True, remove_columns=datasets[name].column_names
            )
    return datasets


# def train_model():
#     from src.data_processing import DataPreprocessing
#     from src.tokenizer import train_tokenizer

#     run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     OUT_DIR = f"models/climb_gpt/{run_name}"
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     dp = DataPreprocessing()
#     datasets = dp.load_climbs()

#     tokenizer = train_tokenizer(datasets, OUT_DIR)
#     # move preprocess_datasets from dp to gpt
#     datasets = preprocess_datasets(datasets, tokenizer)

#     model = KilterGPT(vocab_size=tokenizer.vocab_size)

#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     training_args = TrainingArguments(
#         output_dir=OUT_DIR,
#         eval_strategy="steps",
#         save_strategy="best",
#         save_total_limit=3,
#         overwrite_output_dir=True,
#         logging_steps=1000,
#         num_train_epochs=1,
#         per_device_train_batch_size=16,
#         gradient_accumulation_steps=1,
#         learning_rate=1e-5,
#         weight_decay=0.01,
#         adam_beta1=0.9,
#         adam_beta2=0.999,
#         report_to="tensorboard",
#         remove_unused_columns=False,
#         greater_is_better=False,
#         logging_dir=f"{OUT_DIR}/logs",
#         load_best_model_at_end=True,
#         dataloader_pin_memory=False,
#     )

#     trainer = Trainer(
#         model=model.model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=datasets["train"],
#         eval_dataset=datasets["test"],
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
#     )

#     trainer.train()

#     model.model.save_pretrained(OUT_DIR)
#     tokenizer.save_pretrained(OUT_DIR)
#     print(f"\n✓ Model saved to {OUT_DIR}")


# if __name__ == "__main__":
#     train_model()