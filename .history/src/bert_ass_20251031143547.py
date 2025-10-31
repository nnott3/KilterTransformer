# """
# BERT-based difficulty prediction for Kilter Board routes.
# Uses HuggingFace transformers and datasets libraries.
# """
# import torch
# import torch.nn as nn
# import numpy as np
# from transformers import (
#     BertConfig,
#     BertForSequenceClassification,
#     PreTrainedTokenizerFast,
#     Trainer,
#     TrainingArguments,
#     EarlyStoppingCallback,
# )
# from sklearn.model_selection import train_test_split

# from .data_processing import DataPreprocessing
# # from .evaluation import Evaluation


# class RouteRegressorConfig(BertConfig):
#     """BERT config with regression head."""
#     def __init__(self, num_labels=1, **kwargs):
#         super().__init__(num_labels=num_labels, **kwargs)


# class RouteRegressor(BertForSequenceClassification):
#     """BERT model for difficulty regression."""
    
#     def __init__(self, config):
#         super().__init__(config)
#         # Replace classifier with regression head
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.post_init()
    
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = outputs.pooler_output
#         logits = self.classifier(pooled).squeeze(-1)
        
#         loss = None
#         if labels is not None:
#             loss_fn = nn.SmoothL1Loss()
#             loss = loss_fn(logits, labels)
        
#         return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# class KilterBERTTrainer:
#     """Trainer for BERT-based route difficulty prediction."""
    
#     def __init__(self, tokenizer_path: str = None, hidden_dim: int = 128, 
#                  num_layers: int = 4, num_heads: int = 4):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load or create tokenizer
#         if tokenizer_path:
#             self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
#         else:
#             raise ValueError("Tokenizer path required. Train generation model first.")
        
#         # Initialize model
#         config = RouteRegressorConfig(
#             vocab_size=self.tokenizer.vocab_size,
#             hidden_size=hidden_dim,
#             num_hidden_layers=num_layers,
#             num_attention_heads=num_heads,
#             intermediate_size=hidden_dim * 4,
#             max_position_embeddings=128,
#             pad_token_id=self.tokenizer.pad_token_id,
#         )
#         self.model = RouteRegressor(config)
#         self.model.to(self.device)
    
#     def prepare_datasets(self, dataset_dict):
#         """Tokenize datasets and add labels."""
        
#         def tokenize_function(examples):
#             # Tokenize frames
#             tokenized = self.tokenizer(
#                 examples['frames'],
#                 padding='max_length',
#                 truncation=True,
#                 max_length=128,
#             )
#             # Add labels
#             tokenized['labels'] = examples['display_difficulty']
#             return tokenized
        
#         # Tokenize both splits
#         tokenized_datasets = dataset_dict.map(
#             tokenize_function,
#             batched=True,
#             remove_columns=dataset_dict['train'].column_names
#         )
        
#         # Split train into train/val
#         train_val = tokenized_datasets['train'].train_test_split(
#             test_size=0.125, seed=42
#         )
        
#         return train_val['train'], train_val['test'], tokenized_datasets['test']
    
#     def train(self, train_dataset, val_dataset, output_dir: str = 'models/bert_regressor',
#               epochs: int = 30, batch_size: int = 64, lr: float = 1e-5):
#         """Train the model."""
        
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             num_train_epochs=epochs,
#             per_device_train_batch_size=batch_size,
#             per_device_eval_batch_size=batch_size,
#             learning_rate=lr,
#             weight_decay=0.01,
#             eval_strategy='steps',
#             save_strategy='best',
#             save_total_limit=3,
#             load_best_model_at_end=True,
#             overwrite_output_dir=True,
#             metric_for_best_model='eval_loss',
#             greater_is_better=False,
#             logging_steps=500,
#             report_to='tensorboard',
#             remove_unused_columns=False,
#         )
        
#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#             callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
#         )
        
#         trainer.train()
#         self.model.save_pretrained(output_dir)
#         print(f"\n✓ Model saved to {output_dir}")
    
#     def predict(self, test_dataset) -> np.ndarray:
#         """Predict difficulties."""
#         self.model.eval()
#         predictions = []
        
#         batch_size = 64
#         dataloader = torch.utils.data.DataLoader(
#             test_dataset, batch_size=batch_size
#         )
        
#         with torch.no_grad():
#             for batch in dataloader:
#                 batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
#                 outputs = self.model(**batch)
#                 predictions.extend(outputs['logits'].cpu().numpy())
        
#         return np.array(predictions)
    
#     def load_model(self, path: str):
#         """Load trained model."""
#         self.model = RouteRegressor.from_pretrained(path)
#         self.model.to(self.device)
#         self.model.eval()


# def train_and_evaluate():
#     """Complete training and evaluation pipeline."""
    
#     # Load data using DataPreprocessing
#     dp = DataPreprocessing()
#     dataset_dict = dp.load_climbs()
#     print(f"Train: {len(dataset_dict['train'])}, Test: {len(dataset_dict['test'])}")
    
#     # Initialize trainer (requires pre-trained tokenizer from generation model)
#     tokenizer_path = 'models/climb_gpt/run_20251006_231840/checkpoint-4000'
#     trainer = KilterBERTTrainer(
#         tokenizer_path=tokenizer_path,
#         hidden_dim=128,
#         num_layers=4,
#         num_heads=4
#     )
    
#     print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
#     print(f"Device: {trainer.device}")
    
#     # Prepare datasets (tokenize and split)
#     train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(dataset_dict)
    
#     # Train
#     trainer.train(train_dataset, val_dataset, epochs=30, batch_size=64, lr=2e-4)
    
#     # Evaluate
#     print("\nEvaluating on test set...")
#     y_pred = trainer.predict(test_dataset)
#     y_test = np.array(dataset_dict['test']['display_difficulty'])
    
#     # evaluator = Evaluation(dp)
#     # evaluator.get_scores(y_test, y_pred, "BERT Regressor")
#     # evaluator.plot_predictions(y_test, y_pred)


# if __name__ == "__main__":
#     train_and_evaluate()

"""
Transformer-based encoder using HuggingFace for route difficulty prediction.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from tqdm import tqdm
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup
import datetime
from src.data_processing import DataPreprocessing, HOLD_ID, HOLDCOORDINATES
from src.evaluation import Evaluation
from datasets import ClassLabel, disable_progress_bars
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

class KilterBERT(nn.Module):
    """BERT model with custom regression head for difficulty prediction."""

    def __init__(self, vocab_size, hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=100,
            pad_token_id=0
        )

        self.bert = BertModel(config)
        self.angle_proj = nn.Linear(1, hidden_dim)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, angle, attention_mask=None):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )


        # Get [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0]

        angle_emb = self.angle_proj(angle)
        combined = cls_embedding + angle_emb

        return self.regressor(combined).squeeze(-1)


class BoulderDataset(Dataset):
    """PyTorch dataset for boulder routes."""

    def __init__(self, routes_df: pd.DataFrame, vocab: Dict, max_length: int = 25):
        self.routes = routes_df #.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.routes)

    def __getitem__(self, idx):
        row = self.routes[idx]
        tokens = self._tokenize(row['frames'])

        # Handle angle (works for both None and NaN)
        angle = row['angle_y'] if row['angle_y'] is not None and str(row['angle_y']) != 'nan' else 0.0
        difficulty = row['display_difficulty']

        # Pad tokens
        attention_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(0)  # [PAD]
            attention_mask.append(0)

        return {
            'input_ids': torch.LongTensor(tokens[:self.max_length]),
            'angle': torch.FloatTensor([angle]),
            'attention_mask': torch.FloatTensor(attention_mask[:self.max_length]),
            'difficulty': torch.FloatTensor([difficulty])
        }


    def _tokenize(self, frames):
        """Convert holds to frame sequence: [CLS] hold_tokens... [SEP].
        e.g. angle40_grade23_start1109_start1125_hand1163 """

        tokens = [1] # [CLS]
        frame_split = [f for f in frames.split('_') if ('grade' not in f) and ('angle' not in f)]
        for hold in frame_split[:(self.max_length - 2)]:
            token = self.vocab.get(hold, 0)
            tokens.append(token)

        tokens.append(2) # [SEP]
        return tokens


class KilterEncoder:
    """HuggingFace BERT-based encoder"""

    def __init__(self, model_name='transformer', vocab_size=None, hidden_dim=128, num_layers=4, max_length=25):
        self.model_name = model_name
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.vocab_size = vocab_size or len(self.vocab)
        self.model = KilterBERT(self.vocab_size, hidden_dim, num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary: hold_id_func → token_id."""
        vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2}
        token_id = 3
        for angle in range(20, 60, 5):
            vocab[f"angle{angle}"] = token_id
            token_id += 1

        for hold_id in HOLD_ID:
            for func in ['start', 'hand', 'finish', 'feet']:
                vocab[f"{func}{hold_id}"] = token_id
                token_id += 1
        return vocab

    def train_model(self, train_data, val_data=None,
                   epochs: int = 30, batch_size: int = 64, lr: float = 2e-4):
        """Train the transformer model."""
        train_dataset = BoulderDataset(train_data, self.vocab, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data is not None:
            val_dataset = BoulderDataset(val_data, self.vocab, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        criterion = nn.SmoothL1Loss()

        best_val_loss = float('inf')

        print(f"Training {self.model_name} ...")
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                angle = batch['angle'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                difficulty = batch['difficulty'].squeeze().to(self.device)

                optimizer.zero_grad()
                pred = self.model(input_ids, angle, mask)
                loss = criterion(pred, difficulty)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if val_data is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        angle = batch['angle'].to(self.device)
                        mask = batch['attention_mask'].to(self.device)
                        difficulty = batch['difficulty'].squeeze().to(self.device)

                        pred = self.model(input_ids, angle, mask)
                        loss = criterion(pred, difficulty)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model()
                if epoch % 10 == 0 and epoch != 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    self.save_model(f"{self.model_name}_{epoch}_{timestamp}.pt")

                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
            else:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")

    def predict(self, routes_data, batch_size: int = 64) -> np.ndarray:
        """Predict difficulties for routes."""
        dataset = BoulderDataset(routes_data, self.vocab, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                angle = batch['angle'].to(self.device)
                mask = batch['attention_mask'].to(self.device)

                pred = self.model(input_ids, angle, mask)
                predictions.extend(pred.cpu().numpy().tolist())

        return np.array(predictions)

    def save_model(self, name: str = None):
        """Save model weights."""
        if name is None:
            name = f'{self.model_name}.pt'
        torch.save(self.model.state_dict(), f'saved_models/{name}')

    def load_model(self, path: str = None):
        """Load model weights."""
        if path is None:
            path = f'{self.model_name}.pt'
        self.model.load_state_dict(torch.load(f'saved_models/{path}', map_location=self.device))
        self.model.eval()
        
        
        
def train_and_evaluate():
    disable_progress_bars()

    dp = DataPreprocessing()
    climbs = dp.load_climbs()
    print(f"Dataset: {len(climbs)} climbs")

    num_classes = len(set(climbs["v_grade"]))
    climbs = climbs.cast_column("v_grade", ClassLabel(num_classes=num_classes))

    # Split train/test
    climbs = climbs.train_test_split(test_size=0.1, seed=42, stratify_by_column="v_grade")
    test_df = climbs['test']

    # Split train/val
    train_val = climbs['train'].train_test_split(test_size=0.15, seed=42, stratify_by_column="v_grade")
    train_df = train_val['train']
    val_df = train_val['test']

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    model_name = 'good_ass_transformer'
    encoder = KilterEncoder(model_name='good_ass_transformer', hidden_dim=128, num_layers=4)
    print(f"Vocabulary size: {encoder.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in encoder.model.parameters()):,}")
    print(f"Device: {encoder.device}")

    encoder.train_model(train_df, val_df, epochs=40, batch_size=64, lr=2e-4)

    # Load best model
    encoder.load_model('good_ass_transformer.pt')

    # Evaluate
    print("\nEvaluating on test set...")
    y_test = test_df['display_difficulty']
    y_pred = encoder.predict(test_df)

    evaluator = Evaluation()
    scores = evaluator.get_scores(y_test, y_pred, "GoodAss Transformer")

    evaluator.plot_predictions(y_test, y_pred)


if name == '__main__':
    train_and_evaluate()
    
# """
# BERT-based difficulty prediction for Kilter Board routes.
# Uses HuggingFace transformers and datasets libraries.
# """
# import torch
# import torch.nn as nn
# import numpy as np
# from transformers import (
#     BertConfig,
#     BertForSequenceClassification,
#     PreTrainedTokenizerFast,
#     Trainer,
#     TrainingArguments,
#     EarlyStoppingCallback,
# )
# from sklearn.model_selection import train_test_split

# from .data_processing import DataPreprocessing
# # from .evaluation import Evaluation


# class RouteRegressorConfig(BertConfig):
#     """BERT config with regression head."""
#     def __init__(self, num_labels=1, **kwargs):
#         super().__init__(num_labels=num_labels, **kwargs)


# class RouteRegressor(BertForSequenceClassification):
#     """BERT model for difficulty regression."""
    
#     def __init__(self, config):
#         super().__init__(config)
#         # Replace classifier with regression head
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.post_init()
    
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = outputs.pooler_output
#         logits = self.classifier(pooled).squeeze(-1)
        
#         loss = None
#         if labels is not None:
#             loss_fn = nn.SmoothL1Loss()
#             loss = loss_fn(logits, labels)
        
#         return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# class KilterBERTTrainer:
#     """Trainer for BERT-based route difficulty prediction."""
    
#     def __init__(self, tokenizer_path: str = None, hidden_dim: int = 128, 
#                  num_layers: int = 4, num_heads: int = 4):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load or create tokenizer
#         if tokenizer_path:
#             self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
#         else:
#             raise ValueError("Tokenizer path required. Train generation model first.")
        
#         # Initialize model
#         config = RouteRegressorConfig(
#             vocab_size=self.tokenizer.vocab_size,
#             hidden_size=hidden_dim,
#             num_hidden_layers=num_layers,
#             num_attention_heads=num_heads,
#             intermediate_size=hidden_dim * 4,
#             max_position_embeddings=128,
#             pad_token_id=self.tokenizer.pad_token_id,
#         )
#         self.model = RouteRegressor(config)
#         self.model.to(self.device)
    
#     def prepare_datasets(self, dataset_dict):
#         """Tokenize datasets and add labels."""
        
#         def tokenize_function(examples):
#             # Tokenize frames
#             tokenized = self.tokenizer(
#                 examples['frames'],
#                 padding='max_length',
#                 truncation=True,
#                 max_length=128,
#             )
#             # Add labels
#             tokenized['labels'] = examples['display_difficulty']
#             return tokenized
        
#         # Tokenize both splits
#         tokenized_datasets = dataset_dict.map(
#             tokenize_function,
#             batched=True,
#             remove_columns=dataset_dict['train'].column_names
#         )
        
#         # Split train into train/val
#         train_val = tokenized_datasets['train'].train_test_split(
#             test_size=0.125, seed=42
#         )
        
#         return train_val['train'], train_val['test'], tokenized_datasets['test']
    
#     def train(self, train_dataset, val_dataset, output_dir: str = 'models/bert_regressor',
#               epochs: int = 30, batch_size: int = 64, lr: float = 1e-5):
#         """Train the model."""
        
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             num_train_epochs=epochs,
#             per_device_train_batch_size=batch_size,
#             per_device_eval_batch_size=batch_size,
#             learning_rate=lr,
#             weight_decay=0.01,
#             eval_strategy='steps',
#             save_strategy='best',
#             save_total_limit=3,
#             load_best_model_at_end=True,
#             overwrite_output_dir=True,
#             metric_for_best_model='eval_loss',
#             greater_is_better=False,
#             logging_steps=500,
#             report_to='tensorboard',
#             remove_unused_columns=False,
#         )
        
#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#             callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
#         )
        
#         trainer.train()
#         self.model.save_pretrained(output_dir)
#         print(f"\n✓ Model saved to {output_dir}")
    
#     def predict(self, test_dataset) -> np.ndarray:
#         """Predict difficulties."""
#         self.model.eval()
#         predictions = []
        
#         batch_size = 64
#         dataloader = torch.utils.data.DataLoader(
#             test_dataset, batch_size=batch_size
#         )
        
#         with torch.no_grad():
#             for batch in dataloader:
#                 batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
#                 outputs = self.model(**batch)
#                 predictions.extend(outputs['logits'].cpu().numpy())
        
#         return np.array(predictions)
    
#     def load_model(self, path: str):
#         """Load trained model."""
#         self.model = RouteRegressor.from_pretrained(path)
#         self.model.to(self.device)
#         self.model.eval()


# def train_and_evaluate():
#     """Complete training and evaluation pipeline."""
    
#     # Load data using DataPreprocessing
#     dp = DataPreprocessing()
#     dataset_dict = dp.load_climbs()
#     print(f"Train: {len(dataset_dict['train'])}, Test: {len(dataset_dict['test'])}")
    
#     # Initialize trainer (requires pre-trained tokenizer from generation model)
#     tokenizer_path = 'models/climb_gpt/run_20251006_231840/checkpoint-4000'
#     trainer = KilterBERTTrainer(
#         tokenizer_path=tokenizer_path,
#         hidden_dim=128,
#         num_layers=4,
#         num_heads=4
#     )
    
#     print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
#     print(f"Device: {trainer.device}")
    
#     # Prepare datasets (tokenize and split)
#     train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(dataset_dict)
    
#     # Train
#     trainer.train(train_dataset, val_dataset, epochs=30, batch_size=64, lr=2e-4)
    
#     # Evaluate
#     print("\nEvaluating on test set...")
#     y_pred = trainer.predict(test_dataset)
#     y_test = np.array(dataset_dict['test']['display_difficulty'])
    
#     # evaluator = Evaluation(dp)
#     # evaluator.get_scores(y_test, y_pred, "BERT Regressor")
#     # evaluator.plot_predictions(y_test, y_pred)


# if __name__ == "__main__":
#     train_and_evaluate()
