"""
Feature encoding methods for route difficulty prediction.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from typing import Tuple

from .data_processing import HOLD_ID, HOLDCOORDINATES, DataPreprocessing
from .evaluation import Evaluation


class MatrixEncoder:
    """4-channel grid encoding (one per hold function)."""
    
    def __init__(self, grid_size=(48, 48)):
        self.grid_size = grid_size
        self.func_map = {12: 0, 13: 1, 14: 2, 15: 3}
    
    def encode(self, holds_data: list, angle: float) -> np.ndarray:
        """Encode route as 4-channel grid + angle."""
        grids = np.zeros((4, *self.grid_size), dtype=np.float32)
        
        for hold in holds_data:
            hold_id, func = list(hold.items())[0]
            if hold_id in HOLD_ID:
                idx = HOLD_ID.index(hold_id)
                x, y = HOLDCOORDINATES[idx]
                grid_x = min(int(x / 22.5), self.grid_size[0] - 1)
                grid_y = min(int(y / 24.4), self.grid_size[1] - 1)
                channel = self.func_map.get(func, 0)
                grids[channel, grid_x, grid_y] = 1
        
        return np.concatenate([grids.flatten(), [angle]])
    
    def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        angles = df['angle_y'].fillna(0).values
        return np.array([
            self.encode(row['holds_data'], angle)
            for (_, row), angle in zip(df.iterrows(), angles)
        ])


class MatrixStatsEncoder:
    """Combined matrix + statistical features encoding."""
    
    def __init__(self, grid_size=(48, 48)):
        self.matrix_encoder = MatrixEncoder(grid_size)
        self.stats_encoder = StatsEncoder()
    
    def encode(self, holds_data: list, angle: float) -> np.ndarray:
        """Combine matrix and stats features."""
        matrix_features = self.matrix_encoder.encode(holds_data, angle)
        stats_features = self.stats_encoder.encode(holds_data, angle)
        return np.concatenate([matrix_features, stats_features])
    
    def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        angles = df['angle_y'].fillna(0).values
        return np.array([
            self.encode(row['holds_data'], angle)
            for (_, row), angle in zip(df.iterrows(), angles)
        ])


class StatsEncoder:
    """Statistical feature extraction."""
    
    def encode(self, holds_data: list, angle: float) -> np.ndarray:
        if not holds_data:
            return np.zeros(18)
        
        coords, funcs = [], []
        for hold in holds_data:
            hold_id, func = list(hold.items())[0]
            if hold_id in HOLD_ID:
                idx = HOLD_ID.index(hold_id)
                coords.append(HOLDCOORDINATES[idx])
                funcs.append(func)
        
        if not coords:
            return np.zeros(18)
        
        coords = np.array(coords)
        x, y = coords[:, 0], coords[:, 1]
        
        return np.array([
            len(coords), np.mean(x), np.std(x), np.mean(y), np.std(y),
            np.min(x), np.max(x), np.min(y), np.max(y),
            funcs.count(13), funcs.count(12), funcs.count(15), funcs.count(14),
            np.mean(pdist(coords)) if len(coords) > 1 else 0,
            np.max(y) - np.min(y), np.max(x) - np.min(x),
            angle, np.sum(y) / len(coords)
        ])
    
    def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        angles = df['angle_y'].fillna(0).values
        return np.array([
            self.encode(row['holds_data'], angle)
            for (_, row), angle in zip(df.iterrows(), angles)
        ])


class SimpleGNN(nn.Module):
    """Simple GNN for graph encoding."""
    
    def __init__(self, node_features=5, hidden_dim=32, output_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pooling = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features, edge_index):
        h = self.encoder(node_features)
        messages = torch.zeros_like(h)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            messages[dst] += h[src]
        h = h + messages
        return self.pooling(torch.mean(h, dim=0))


class GraphEncoder:
    """Graph-based encoding with GNN."""
    
    def __init__(self, gnn_model=None):
        self.gnn = gnn_model or SimpleGNN()
    
    def encode_single(self, holds_data: list, angle: float) -> np.ndarray:
        if not holds_data:
            return np.zeros(16)
        
        node_features = []
        for hold in holds_data:
            hold_id, func = list(hold.items())[0]
            if hold_id in HOLD_ID:
                idx = HOLD_ID.index(hold_id)
                x, y = HOLDCOORDINATES[idx]
                node_features.append([hold_id, func, x, y, angle])
        
        if not node_features:
            return np.zeros(16)
        
        node_features = torch.FloatTensor(node_features)
        n = len(node_features)
        edge_index = torch.combinations(torch.arange(n), r=2).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        with torch.no_grad():
            embedding = self.gnn(node_features, edge_index)
        return embedding.cpu().detach().numpy()
    
    def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        angles = df['angle_y'].fillna(0).values
        return np.array([
            self.encode_single(row['holds_data'], angle)
            for (_, row), angle in zip(df.iterrows(), angles)
        ])


class SimpleLSTM(nn.Module):
    """Simple LSTM for sequence encoding."""
    
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n.squeeze(0))


class SequenceEncoder:
    """Sequence-based encoding with LSTM."""
    
    def __init__(self, max_holds=20, lstm_model=None):
        self.max_holds = max_holds
        self.lstm = lstm_model or SimpleLSTM(input_dim=3, hidden_dim=32, output_dim=16)
    
    def encode_single(self, holds_data: list, angle: float) -> np.ndarray:
        holds_with_coords = []
        for hold in holds_data:
            hold_id, func = list(hold.items())[0]
            if hold_id in HOLD_ID:
                idx = HOLD_ID.index(hold_id)
                x, y = HOLDCOORDINATES[idx]
                holds_with_coords.append([hold_id, func, x, y])
        
        if not holds_with_coords:
            return np.zeros(16)
        
        # Sort by y-coordinate (bottom to top)
        holds_with_coords.sort(key=lambda h: h[3])
        
        # Keep only hold_id and func, add angle
        sequence = [[h[0], h[1], angle] for h in holds_with_coords[:self.max_holds]]
        seq_len = len(sequence)
        
        # Pad to max_holds
        while len(sequence) < self.max_holds:
            sequence.append([0, 0, 0])
        
        sequence = torch.FloatTensor(sequence).unsqueeze(0)
        lengths = torch.tensor([seq_len])
        
        with torch.no_grad():
            embedding = self.lstm(sequence, lengths)
        return embedding.squeeze().cpu().detach().numpy()
    
    def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        angles = df['angle_y'].fillna(0).values
        return np.array([
            self.encode_single(row['holds_data'], angle)
            for (_, row), angle in zip(df.iterrows(), angles)
        ])


class ExperimentRunner:
    """Run encoding + model experiments."""
    
    def __init__(self, data_processor: DataPreprocessing = None, 
                 evaluator: Evaluation = None):
        self.dp = data_processor or DataPreprocessing()
        self.evaluator = evaluator or Evaluation(self.dp)
    
    def run(self, encoder, model, routes_df: pd.DataFrame, 
            experiment_name: str, test_size: float = 0.2) -> dict:
        """Run complete experiment."""
        import time
        
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        print("Encoding features...")
        X = encoder.encode_dataframe(routes_df)
        y = routes_df['display_difficulty'].values
        print(f"Feature shape: {X.shape}")
        
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=routes_df['v_grade']
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f}s")
        
        y_pred = model.predict(X_test)
        scores = self.evaluator.get_scores(y_test, y_pred, experiment_name)
        
        
        return {
            'name': experiment_name,
            'scores': scores,
            'train_time': train_time,
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }


if __name__ == "__main__":
    dp = DataPreprocessing()
    routes = dp.load_routes()
    routes = dp.clean_routes(routes)
    print(f"Dataset: {len(routes)} routes")
    
    runner = ExperimentRunner(dp)
    results = []
    
    # Experiment 1: Matrix + XGBoost
    results.append(runner.run(
        MatrixEncoder(grid_size=(48, 48)),
        XGBRegressor(n_estimators=100, random_state=42),
        routes,
        "Matrix + XGBoost"
    ))
    
    # Experiment 2: Stats + GradientBoosting
    results.append(runner.run(
        StatsEncoder(),
        XGBRegressor(n_estimators=100, random_state=42),
        routes,
        "Stats + XGBoost"
    ))
    
    # Experiment 3: Matrix + Stats + XGBoost
    results.append(runner.run(
        MatrixStatsEncoder(grid_size=(48, 48)),
        XGBRegressor(n_estimators=100, random_state=42),
        routes,
        "Matrix+Stats + XGBoost"
    ))
    
    # Experiment 4: Graph + XGBoost
    print("\nNote: GraphEncoder uses pretrained GNN. Training GNN from scratch for better results.")
    results.append(runner.run(
        GraphEncoder(),
        XGBRegressor(n_estimators=100, random_state=42),
        routes,
        "Graph + XGBoost"
    ))
    
    # Experiment 5: Sequence + XGBoost
    print("\nNote: SequenceEncoder uses pretrained LSTM. Training LSTM from scratch for better results.")
    results.append(runner.run(
        SequenceEncoder(max_holds=20),
        XGBRegressor(n_estimators=100, random_state=42),
        routes,
        "Sequence + XGBoost"
    ))
    
    # Compare all
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    comparison = pd.DataFrame([
        {
            'Method': r['name'],
            'R²': r['scores']['r2'],
            'MAE': r['scores']['mae'],
            'RMSE': r['scores']['rmse'],
            'Train Time (s)': r['train_time'],
        }
        for r in results
    ])
    print(comparison.to_string(index=False))
    
    # Find best method
    best = comparison.loc[comparison['R²'].idxmax()]
    print(f"\nBest Method: {best['Method']}")
    print(f"R²: {best['R²']:.4f}, MAE: {best['MAE']:.4f}, RMSE: {best['RMSE']:.4f}")
    print(f"Training Time: {best['Train Time (s)']:.2f}s")