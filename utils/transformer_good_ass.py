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

from utils.data_processing import HOLD_ID, DataPreprocessing
from utils.evaluation import Evaluation


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
        self.routes = routes_df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.routes)
    
    def __getitem__(self, idx):
        row = self.routes.iloc[idx]
        tokens = self._tokenize(row['holds_data'])
        angle = row['angle_y'] if not pd.isna(row['angle_y']) else 0.0
        difficulty = row['display_difficulty']
        
        # Pad tokens
        attentioxn_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(0)  # [PAD]
            attention_mask.append(0)
        
        return {
            'input_ids': torch.LongTensor(tokens[:self.max_length]),
            'angle': torch.FloatTensor([angle]),
            'attention_mask': torch.FloatTensor(attention_mask[:self.max_length]),
            'difficulty': torch.FloatTensor([difficulty])
        }
    
    def _tokenize(self, holds_data):
        """Convert holds to token sequence: [CLS] hold_tokens... [SEP]."""
        tokens = [1]  # [CLS]
        
        for hold in holds_data[:(self.max_length - 2)]:
            hold_id, func = list(hold.items())[0]
            
            hand_or_foot = 0 if func == 13 else 1 ##############
            token = self.vocab.get(f"{hold_id}_{hand_or_foot}", 0) ##############
            # token = self.vocab.get(f"{hold_id}_{func}", 0)
            tokens.append(token)
        
        tokens.append(2)  # [SEP]
        return tokens


class KilterEncoder:
    """HuggingFace BERT-based encoder with training capabilities."""
    
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
        for hold_id in HOLD_ID:
            # for func in [12, 13, 14, 15]:
            #     vocab[f"{hold_id}_{func}"] = token_id
            for hand_or_foot in [0, 1]:  # 0=foot, 1=hand/start/finish
                vocab[f"{hold_id}_{hand_or_foot}"] = token_id
                token_id += 1
        return vocab
    
    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                   epochs: int = 30, batch_size: int = 64, lr: float = 2e-4):
        """Train the transformer model."""
        train_dataset = BoulderDataset(train_df, self.vocab, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_df is not None:
            val_dataset = BoulderDataset(val_df, self.vocab, self.max_length)
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
            if val_df is not None:
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
                    self.save_model(f'{self.model_name}.pt')
                if epoch % 10 == 0 and epoch != 0:
                    self.save_model(f'{self.model_name}_{epoch}.pt')

                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
            else:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")
    
    def predict(self, routes_df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
        """Predict difficulties for routes."""
        dataset = BoulderDataset(routes_df, self.vocab, self.max_length)
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
    
    def save_model(self, name: str = "transformer_model"):
        """Save model weights."""
        torch.save(self.model.state_dict(), f'saved_models/{self.model_name}.pt')
    
    def load_model(self, path: str = None):
        """Load model weights."""
        if path is None:
            path = self.model_name
        self.model.load_state_dict(torch.load(f'saved_models/{path}', map_location=self.device))
        self.model.eval()


def train_and_evaluate():
    """Complete training and evaluation pipeline."""
    from sklearn.model_selection import train_test_split

    # Load data
    dp = DataPreprocessing()
    routes = dp.load_routes()
    # routes = dp.clean_routes(routes)
    print(f"Dataset: {len(routes)} routes")

    # Split data
    train_df, test_df = train_test_split(
        routes, test_size=0.15, random_state=42,
        stratify=routes['v_grade']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.125, random_state=42,  # 0.125 of 0.8 = 0.1 of total
        stratify=train_df['v_grade']
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train model
    encoder = KilterEncoder(hidden_dim=128, num_layers=4)
    print(f"Vocabulary size: {encoder.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in encoder.model.parameters()):,}")
    print(f"Device: {encoder.device}")

    # encoder.train_model(train_df, val_df, epochs=23, batch_size=64, lr=2e-4)

    # Load best model
    encoder.load_model('good_ass_transformer.pt')

    # Evaluate
    print("\nEvaluating on test set...")
    y_test = test_df['display_difficulty'].values
    y_pred = encoder.predict(test_df)

    evaluator = Evaluation(dp)
    scores = evaluator.get_scores(y_test, y_pred, "Transformer BERT")

    evaluator.plot_predictions(y_test, y_pred)    

if __name__ == "__main__":
    train_and_evaluate()

