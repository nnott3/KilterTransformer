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
BERT-based difficulty prediction for Kilter Board routes.
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    )
from sklearn.model_selection import train_test_split
from .data_processing import DataPreprocessing
from .evaluation import Evaluation


class RouteRegressorConfig(BertConfig):
    """BERT config with regression head."""
    def __init__(self, num_labels=1, **kwargs):
        super().__init__(num_labels=num_labels, **kwargs)


class RouteRegressor(BertForSequenceClassification):
    """BERT model for difficulty regression."""  
    def __init__(self, config):
        super().__init__(config)
        # Replace classifier with regression head
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fn = nn.SmoothL1Loss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


class KilterBERTTrainer:
    """Trainer for BERT-based route difficulty prediction."""
    def __init__(self, tokenizer_path: str = None, hidden_dim: int = 128, 
                 num_layers: int = 4, num_heads: int = 4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load or create tokenizer
        if tokenizer_path:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        else:
            raise ValueError("Tokenizer path required. Train generation model first.")
        
        # Initialize model
        config = RouteRegressorConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            max_position_embeddings=128,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.model = RouteRegressor(config)
        self.model.to(self.device)
    
    def prepare_datasets(self, dataset_dict):
        """Tokenize datasets and add labels."""
        
        def tokenize_function(examples):
            # Tokenize frames
            tokenized = self.tokenizer(
                examples['frames'],
                padding='max_length',
                truncation=True,
                max_length=128,
            )
            # Add labels
            tokenized['labels'] = examples['display_difficulty']
            return tokenized
        
        # Tokenize both splits
        tokenized_datasets = dataset_dict.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_dict['train'].column_names
        )
        
        # Split train into train/val
        train_val = tokenized_datasets['train'].train_test_split(
            test_size=0.125, seed=42
        )
        
        # Set format to PyTorch tensors
        train_val['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        train_val['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_datasets['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return train_val['train'], train_val['test'], tokenized_datasets['test']
    
    def train(self, train_dataset, val_dataset, output_dir: str = 'models/bert_regressor',
              epochs: int = 30, batch_size: int = 64, lr: float = 1e-5):
        """Train the model."""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            eval_strategy='steps',
            save_strategy='best',
            save_total_limit=3,
            load_best_model_at_end=True,
            overwrite_output_dir=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            logging_steps=500,
            report_to='tensorboard',
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        
        trainer.train()
        self.model.save_pretrained(output_dir)
        print(f"\n✓ Model saved to {output_dir}")
    
    def predict(self, test_dataset) -> np.ndarray:
        """Predict difficulties."""
        self.model.eval()
        predictions = []
        
        batch_size = 64
        dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size
        )
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**batch)
                predictions.extend(outputs['logits'].cpu().numpy())
        
        return np.array(predictions)
    
    def load_model(self, path: str):
        """Load trained model."""
        self.model = RouteRegressor.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()


    
    


if __name__ == "__main__":
    # Load data using DataPreprocessing
    dp = DataPreprocessing()
    dataset_dict = dp.load_climbs().train_test_split(test_size=0.2, seed=42)
    print(f"Train: {len(dataset_dict['train'])}, Test: {len(dataset_dict['test'])}")
    
    # Initialize trainer (requires pre-trained tokenizer from generation model)
    tokenizer_path = 'models/climb_gpt/run_20251006_231840/checkpoint-4000'  # Directory, not .json file
    trainer = KilterBERTTrainer(
        tokenizer_path=tokenizer_path,
        hidden_dim=128,
        num_layers=4,
        num_heads=4
    )
    
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Device: {trainer.device}")
    
    # Prepare datasets (tokenize and split)
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(dataset_dict)
    
    # Train
    trainer.train(train_dataset, val_dataset, epochs=1, batch_size=64, lr=1e-5)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = trainer.predict(test_dataset)
    y_test = np.array(dataset_dict['test']['display_difficulty'])
    
    evaluator = Evaluation()
    evaluator.get_scores(y_test, y_pred, "BERT Regressor")
    evaluator.plot_predictions(y_test, y_pred)