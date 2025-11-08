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
    LogitsProcessor,
    LogitsProcessorList,
)
from typing import List, Tuple, Optional
import re
from src.tokenizer import train_tokenizer, tokenize_datasets
from src.data_processing import DataPreprocessing
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

class KilterGPT(nn.Module):
    """GPT-2 model for generating Kilter Board climbing routes."""

    def __init__(
        self, 
        n_embd: int = 256,
        n_head: int = 4,
        n_layer: int = 6,
        n_positions: int = 128,
        dropout: float = 0.1,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        ):
        
        super().__init__()

        self.tokenizer = tokenizer
        config = GPT2Config(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_positions=n_positions,
            n_ctx=n_positions,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            vocab_size=tokenizer.vocab_size,
            )
        self.config = config
        self.model = GPT2LMHeadModel(config)
        
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass with order-invariant loss.
        
        If labels provided: Computes custom multi-target loss
        If labels None: Returns logits only (for generation)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        
        if labels is not None:
            # Order-invariant loss: any remaining hold is valid
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Use vectorized loss function (imported from your loss module)
            loss = self._any_of_next_token_loss_vectorized(shift_logits, shift_labels)
            outputs.loss = loss
        
        return outputs
    
    @staticmethod
    def _any_of_next_token_loss_vectorized(logits, shift_labels, ignore_index=-100,
                                       bos_token_id=1, eos_token_id=2, print_debug=False):
        """
        Order-invariant loss for climbing routes.

        Loss behavior:
        - BOS (token_id=1): Never predicted (always first in sequence)
        - Angle, Grade, Holds: Set-loss (order-invariant, any remaining token is valid)
        - EOS (token_id=2): Excluded from valid set (standard single-target prediction)
        - Padding: Ignored

        Args:
            logits: (B, L, V) # batch_size, seq_len, vocab_size
            shift_labels: (B, L)
            ignore_index: Padding marker (default -100)
            bos_token_id: BOS token ID (default 1)
            eos_token_id: EOS token ID (default 2)
        """
        B, L, V = logits.shape
        device = logits.device

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)

        # mask for valid label positions
        # [TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, ...] where FALSE are PADs
        valid_positions = (shift_labels != ignore_index)  # (B, L)


        # upper-triangle of 1 size (L, L)
        future_mask = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool))

        # Combine: for each sequence, each position sees all future tokens that are valid
        # Expand: valid_positions.unsqueeze(1): (B, L) → (B, 1, L)
        # valid_future_mask[b, i, j] = True if label[j] is valid future of i and j >= i
        valid_future_mask = valid_positions.unsqueeze(1) & future_mask  # (B, L, L)
        # valid_future_mask[0][0] => [TRUE, TRUE, TRUE, ..., TRUE, FALSE, ..., FALSE] (True for all valid tokens + False for PADs)
        # valid_future_mask[0][1] => [FALSE, TRUE, TRUE, ..., TRUE, FALSE, ..., FALSE]
        # valid_future_mask[0][2] => [FALSE, FALSE, TRUE, ..., TRUE, FALSE, ..., FALSE]
        # valid_future_mask[0][-seq_len] => [FALSE, FALSE, FALSE, ..., FALSE, FALSE, ..., FALSE]


        labels_expanded = shift_labels.unsqueeze(1).expand(-1, L, -1)  # (B, L, L)
        # labels_expanded[0] => (B, L) 2d array, row count = batch_size B=16
        # each row labels_expanded[0][0] is [token0, token1, token2, ..., token_seq_len, -100, -100, ...] (len L=21)

        # Set ignored ones to -1
        labels_expanded = torch.where(valid_future_mask, labels_expanded, -torch.ones_like(labels_expanded))
        # each row labels_expanded[0][0] is now [token0, token1, token2, ..., token_seq_len, -1, -1, ...] (len L=21)
        # labels_expanded[0][0] => [token0, token1, token2, ..., token_seq_len, -1, -1, ...] (len L=21)
        # labels_expanded[0][1] => [-1,     token1, token2, ..., token_seq_len, -1, -1, ...]
        # labels_expanded[0][2] => [-1,       -1,   token2, ..., token_seq_len, -1, -1, ...]


        # ========================================================================
        # NEW: Exclude EOS from the valid set (EOS should be predicted exactly, not as part of set)
        # ========================================================================
        # Create mask for EOS tokens in future positions
        # Shape: (B, L, L) - True where future token is EOS
        is_eos_future = (labels_expanded == eos_token_id)

        # Remove EOS from valid_future_mask
        # This makes EOS tokens NOT contribute to the set-loss
        # They will only be predicted when they are the ONLY remaining token
        valid_future_mask = valid_future_mask & ~is_eos_future
        # Explanation:
        # Before: valid_future_mask[0][i] might include EOS among valid predictions
        # After:  valid_future_mask[0][i] excludes EOS from the set
        # Result: When predicting, model won't see EOS as "any valid token"
        #         EOS only gets predicted when it's the last token (standard behavior)


        # Build boolean mask per vocab id using scatter_
        valid_token_mask = torch.zeros((B, L, V), dtype=torch.bool, device=device)
        scatter_idx = labels_expanded.clone()
        scatter_idx[scatter_idx < 0] = 0  # convert -1 to dummy 0, to ignore the positions

        valid_token_mask.scatter_(dim=2, index=scatter_idx, src=valid_future_mask)  # mark valid vocab positions
        # (B, L, V)(16, 21, 1932)
        # valid_token_mask[0][0]
        # [PAD, BOS, EOS, UNK, ---angle---, ---grade---, ---holds---]
        # valid_token_mask[0][0], sum=12 < seq_len
        # [FALSE, TRUE, FALSE, ---one True angle, ---one True grade, ---several True holds---]
        #              ^^^^^ NOTE: EOS is now FALSE (excluded from set)

        # valid_token_mask[0][1], sum=11
        # [FALSE, FALSE, FALSE, ---one True angle, ---one True grade, ---several True holds---]
        #         ^^^^^ BOS excluded    ^^^^^ EOS excluded

        # valid_token_mask[0][2], sum=10
        # [FALSE, FALSE, FALSE, ---all False angle, ---one True grade, ---several True holds---]

        # valid_token_mask[0][3], sum=9
        # [FALSE, FALSE, FALSE, ---all False angle, ---all False grade, ---several True holds---]

        # and then the hold tokens ...

        # note, for next example in batch:
        # valid_token_mask[1][0], sum=17 < seq_len
        # [FALSE, TRUE, FALSE, ---one True angle, ---one True grade, ---several True holds---]

        valid_token_mask[:, :, 0] &= (labels_expanded[:, :, 0] != 0)  # fix dummy zeros if any


        # ========================================================================
        # NEW: Add back EOS prediction when it's the only valid next token
        # ========================================================================
        # Find positions where the next token should be EOS
        # Shape: (B, L)
        next_is_eos = (shift_labels == eos_token_id)

        # At these positions, enable EOS in the vocabulary mask
        # This allows standard cross-entropy loss for EOS prediction
        valid_token_mask[:, :, eos_token_id] |= next_is_eos
        # Explanation:
        # If position i should predict EOS (next_is_eos[b,i] = True):
        #   - Set valid_token_mask[b, i, eos_token_id] = True
        #   - This makes ONLY EOS valid at this position
        #   - Standard single-target prediction for EOS!
        '''
        valid_token_mask[:, :, bos_token_id] = False
        '''

        # Filter(/mask) log_probs for only valid positions
        masked_log_probs = torch.where(valid_token_mask, log_probs, torch.full_like(log_probs, -1e9))
        # log_probs[0][0] =>        [-7.6137, -6.4946, -7.7223, ...] the usual
        # valid_token_mask[0][0] => [FALSE,    TRUE,    FALSE,   ...] (BOS and EOS excluded)
        # masked_log_probs[0][0] => [-1e9,    -6.4946,  -1e9,   ...]


        # Combine log(P(token1) + P(token2) + ... + P(tokenN))
        log_sum = torch.logsumexp(masked_log_probs, dim=-1)  # (B, L)
        # For positions with multiple valid tokens: log(P1 + P2 + ... + PN) = set-loss
        # For positions with only EOS valid: log(P_eos) = standard single-target loss


        # log(P1) + log(P2) + ... + log(PN)
        # count_valid = valid_token_mask.sum(dim=-1).clamp(min=1)
        # log_sum = masked_log_probs.sum(dim=-1) / count_valid


        # Only keep positions that had at least one valid target
        has_valid = valid_token_mask.any(dim=-1) # Shape: (B, L)

        # average negative log-likelihood
        loss = -log_sum[has_valid].mean()
        # loss = -log_sum[has_valid].sum() / valid_positions.sum()


        if print_debug:
            print("\n" + "="*70)
            print("DEBUG: Loss Computation Details")
            print("="*70)
            print(f"\nSequence 0 breakdown:")
            for i in range(min(8, L)):
                if has_valid[0, i]:
                    valid_tokens = [j for j in range(V) if valid_token_mask[0, i, j]]
                    num_valid = len(valid_tokens)
                    is_eos_pos = next_is_eos[0, i].item()
                    print(f"  Pos {i}: {num_valid} valid tokens | Target={shift_labels[0,i].item():4d} | "
                        f"EOS-only={is_eos_pos} | log_sum={log_sum[0,i].item():7.3f}")


        return loss

    @classmethod
    def load_from_checkpoint(cls, OUT_DIR, datasets):
        """
        Recreate KilterGPT, tokenizer, and Trainer from a saved OUT_DIR.
        Returns: (gpt, trainer, tokenizer)
        """

        # === Paths ===
        CHECKPOINTS = sorted(
            [d for d in os.listdir(OUT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        CHECKPOINT_DIR = os.path.join(OUT_DIR, CHECKPOINTS[-1]) if CHECKPOINTS else OUT_DIR
        MODEL_BIN = f"{OUT_DIR}/pytorch_model.bin"
        ARGS_BIN = f"{CHECKPOINT_DIR}/training_args.bin"

        # === 1. Load tokenizer ===
        tokenizer = PreTrainedTokenizerFast.from_pretrained(OUT_DIR)
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

        # === 2. Load config (must use same vocab_size) ===
        config = GPT2Config.from_pretrained(OUT_DIR)
        config.vocab_size = tokenizer.vocab_size  # force match (avoid resizing warning)

        # === 3. Create model and load weights ===
        gpt = cls(tokenizer=tokenizer)
        gpt.model = GPT2LMHeadModel(config)
        gpt.model.from_pretrained(OUT_DIR)

        # === 4. Load TrainingArguments ===
        training_args = torch.load(ARGS_BIN, weights_only=False)

        # === 5. Create Trainer ===
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=gpt,
            args=training_args,
            data_collator=data_collator,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
        )

        return gpt, trainer, tokenizer


    def generate_route(
        self,
        prompt: str = "angle50 grade20",
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        skip_special_tokens=True,
        device: Optional[str] = None,
        ) -> str:
        """
        Generate a single climbing route from a prompt. 
    
        Returns decoded tokens as a string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Pass tokenizer during __init__ or use set_tokenizer()")
        
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        self.to(device)
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device) #

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids[:, :-1], #self.tokenizer.encode auto-add EOS to prompt, we remove it
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=skip_special_tokens)
    
    def test_generate_from_prompts(
        self,
        test_prompts: Optional[List[str]] = None,
        device: Optional[str] = None,
        **generate_kwargs
        ):
        if test_prompts is None:
            test_prompts = [
                "",
                "angle40",
                "angle40 grade15",
                "angle40 grade15 start1139",
                "angle40 grade15 hand1315 hand1131",
                "angle50 grade20 feet1117 start1131 start1163 hand1234 hand1247 hand1302"
            ]
        
        if device is None:
            device = next(self.parameters()).device
        
        print("=" * 70)
        print("GENERATION TEST")
        print("=" * 70)
        
        for prompt in test_prompts:
            generated = self.generate_route(prompt, device=device, **generate_kwargs)
            # print(f"\n{'Prompt':<15}: {prompt}")
            print(f"{'Prompt':<15}: {prompt}")
            print(f"{'Generated':<15}: {generated}\n")

    def generate_and_compare_with_dataset(
        self,
        dataset,
        num_samples: int = 2,
        prompt_length: int = 7,
        save_fig = False,
        **generate_kwargs
        ):
        """
        Compare model generation with real dataset examples.
        
        Args:
            dataset: Dataset to sample from (should have 'input_ids' field)
            num_samples: Number of examples to compare
            prompt_length: Number of tokens to use as prompt
            viz: Optional visualizer object with plot_boulder() method
            **generate_kwargs: Arguments for generate_route()
        """
        from src.visualization import Visualization
        viz = Visualization()
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set")
        
        device = next(self.parameters()).device
        
        print("=" * 70)
        print("DATASET COMPARISON")
        print("=" * 70)
        
        # Sample random indices
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        for idx in indices:
            # Get dataset example
            example_ids = dataset[idx]['input_ids']
            
            # Create prompt from first k tokens
            prompt_ids = example_ids[:prompt_length]
            prompt_ids = [tid for tid in prompt_ids if tid != self.tokenizer.pad_token_id]
            
            # Decode
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            full_route = self.tokenizer.decode(example_ids, skip_special_tokens=True)
            
            # Generate from prompt
            generated = self.generate_route(prompt, device=device, **generate_kwargs)
            
            print(f"Prompt:    {prompt}")
            print(f"Dataset:   {full_route}")
            print(f"Generated: {generated}")
            
            viz.plot_boulder("_".join(full_route.split()), name=f"Dataset_{idx}", save_fig=save_fig)
            viz.plot_boulder("_".join(generated.split()), name=f"Generated_{idx}", save_fig=save_fig)
            
            if save_fig:
                os.makedirs("figs", exist_ok=True)
                fig.savefig(f"figs/compare_{idx}.png", dpi=150)

    def generate_with_constraint(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        min_holds: int = 5,
        max_holds: int = 12,
        use_constraints: bool = True,
        ) -> str:
        """
        Generate a route with optional constraint enforcement.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        logits_processor = None
        if use_constraints:
            logits_processor = LogitsProcessorList([
                RouteConstraintProcessor(tokenizer, min_holds=min_holds, max_holds=max_holds)
            ])

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor,
                repetition_penalty=1.2,
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"{'Prompt':<10}: {prompt}")
        print(f"{'Generated':<10}: {decoded}")
        
        return decoded


class RouteConstraintProcessor(LogitsProcessor):
    """Enforce logical and structural constraints on climbing route generation."""

    def __init__(self, tokenizer, min_holds=5, max_holds=15):
        self.tokenizer = tokenizer
        self.min_holds = min_holds
        self.max_holds = max_holds

        vocab = tokenizer.get_vocab()
        self.start_tokens = [tid for token, tid in vocab.items() if token.startswith('start')]
        self.finish_tokens = [tid for token, tid in vocab.items() if token.startswith('finish')]
        self.hand_tokens = [tid for token, tid in vocab.items() if token.startswith('hand')]
        self.feet_tokens = [tid for token, tid in vocab.items() if token.startswith('feet')]
        self.all_hold_tokens = (
            self.start_tokens + self.hand_tokens + self.finish_tokens + self.feet_tokens
        )
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            decoded = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)
            parts = decoded.split()

            if len(parts) < 2:
                continue

            holds = [p for p in parts[2:] if any(p.startswith(r) for r in ['start', 'hand', 'finish', 'feet'])]

            start_count = sum(h.startswith('start') for h in holds)
            finish_count = sum(h.startswith('finish') for h in holds)
            hand_count = sum(h.startswith('hand') for h in holds)
            feet_count = sum(h.startswith('feet') for h in holds)
            hold_count = len(holds)

            # --- EOS blocking until minimal structure is formed ---
            if hold_count < self.min_holds or start_count < 1 or finish_count < 1:
                scores[batch_idx, self.eos_token_id] = -float('inf')

            # --- Start/Finish limits ---
            if start_count >= 2:
                scores[batch_idx, self.start_tokens] = -float('inf')
            if finish_count >= 2:
                scores[batch_idx, self.finish_tokens] = -float('inf')

            # --- Force EOS once max reached ---
            if hold_count >= self.max_holds:
                scores[batch_idx, self.all_hold_tokens] = -float('inf')
                scores[batch_idx, self.eos_token_id] = 100.0  # strong bias toward EOS

        return scores

# def train_model():
    

#     run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     OUT_DIR = f"models/climb_gpt/{run_name}"
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     dp = DataPreprocessing()
#     datasets = dp.load_climbs()

#     tokenizer = train_tokenizer(datasets, OUT_DIR)
#     # move tokenize_datasets from dp to gpt
#     datasets = tokenize_datasets(datasets, tokenizer)

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