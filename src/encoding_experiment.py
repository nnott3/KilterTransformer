# """
# Feature encoding methods for route difficulty prediction.
# """
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from scipy.spatial.distance import pdist

# from .data_processing import HOLD_ID, HOLDCOORDINATES, DataPreprocessing
# from .evaluation import Evaluation


# class MatrixEncoder:
#     """4-channel grid encoding (one per hold function)."""
    
#     def __init__(self, grid_size=(48, 48)):
#         self.grid_size = grid_size
#         self.func_map = {12: 0, 13: 1, 14: 2, 15: 3}
#         # ✅ Pre-compute hold_id to index mapping (O(1) lookup)
#         self.hold_to_idx = {hold_id: idx for idx, hold_id in enumerate(HOLD_ID)}
#         # ✅ Pre-compute grid coordinates for all holds
#         self.hold_coords = {
#             hold_id: (
#                 min(int(HOLDCOORDINATES[idx][0] / 22.5), self.grid_size[0] - 1),
#                 min(int(HOLDCOORDINATES[idx][1] / 24.4), self.grid_size[1] - 1)
#             )
#             for hold_id, idx in self.hold_to_idx.items()
#         }
    
#     def encode(self, holds_data: list, angle: float) -> np.ndarray:
#         grids = np.zeros((4, *self.grid_size), dtype=np.float32)
        
#         for hold in holds_data:
#             hold_id, func = list(hold.items())[0]
#             # ✅ O(1) lookup instead of O(n) search
#             if hold_id in self.hold_coords:
#                 grid_x, grid_y = self.hold_coords[hold_id]
#                 channel = self.func_map.get(func, 0)
#                 grids[channel, grid_x, grid_y] = 1
        
#         return np.concatenate([grids.flatten(), [angle]])
    
#     def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
#         # ✅ Avoid iterrows() - use .values directly
#         holds_list = df['holds_data'].values
#         angles = df['angle_y'].fillna(0).values
        
#         # ✅ Use list comprehension with zip (faster than iterrows)
#         return np.array([
#             self.encode(holds, angle)
#             for holds, angle in zip(holds_list, angles)
#         ])

# class MatrixStatsEncoder:
#     """Combined matrix + statistical features encoding."""
    
#     def __init__(self, grid_size=(48, 48)):
#         self.matrix_encoder = MatrixEncoder(grid_size)
#         self.stats_encoder = StatsEncoder()
    
#     def encode(self, holds_data: list, angle: float) -> np.ndarray:
#         """Combine matrix and stats features."""
#         matrix_features = self.matrix_encoder.encode(holds_data, angle)
#         stats_features = self.stats_encoder.encode(holds_data, angle)
#         return np.concatenate([matrix_features, stats_features])
    
#     def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
#         angles = df['angle_y'].fillna(0).values
#         return np.array([
#             self.encode(row['holds_data'], angle)
#             for (_, row), angle in zip(df.iterrows(), angles)
#         ])


# class StatsEncoder:
#     """Statistical feature extraction."""
    
#     def encode(self, holds_data: list, angle: float) -> np.ndarray:
#         if not holds_data:
#             return np.zeros(18)
        
#         coords, funcs = [], []
#         for hold in holds_data:
#             hold_id, func = list(hold.items())[0]
#             if hold_id in HOLD_ID:
#                 idx = HOLD_ID.index(hold_id)
#                 coords.append(HOLDCOORDINATES[idx])
#                 funcs.append(func)
        
#         if not coords:
#             return np.zeros(18)
        
#         coords = np.array(coords)
#         x, y = coords[:, 0], coords[:, 1]
        
#         return np.array([
#             len(coords), np.mean(x), np.std(x), np.mean(y), np.std(y),
#             np.min(x), np.max(x), np.min(y), np.max(y),
#             funcs.count(13), funcs.count(12), funcs.count(15), funcs.count(14),
#             np.mean(pdist(coords)) if len(coords) > 1 else 0,
#             np.max(y) - np.min(y), np.max(x) - np.min(x),
#             angle, np.sum(y) / len(coords)
#         ])
    
#     def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
#         angles = df['angle_y'].fillna(0).values
#         return np.array([
#             self.encode(row['holds_data'], angle)
#             for (_, row), angle in zip(df.iterrows(), angles)
#         ])


# class SimpleLSTM(nn.Module):
    
#     def __init__(self, input_dim=3, hidden_dim=32, output_dim=16):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x, lengths):
#         packed = nn.utils.rnn.pack_padded_sequence(
#             x, lengths, batch_first=True, enforce_sorted=False
#         )
#         _, (h_n, _) = self.lstm(packed)
#         return self.fc(h_n.squeeze(0))


# class SequenceEncoder:
    
#     def __init__(self, max_holds=20, lstm_model=None):
#         self.max_holds = max_holds
#         self.lstm = lstm_model or SimpleLSTM(input_dim=3, hidden_dim=32, output_dim=16)
#         self.hold_to_idx = {hold_id: idx for idx, hold_id in enumerate(HOLD_ID)}
#         self.func_map = {
#             12: 12, 13: 13, 14: 14, 15: 15,
#             'start': 12, 'finish': 14, 'foot': 15, 'hand': 13
#         }
    
#     def encode_single(self, holds_data: list, angle: float) -> np.ndarray:
#         holds_with_coords = []
#         for hold in holds_data:
#             hold_id, func = list(hold.items())[0]
#             if hold_id in HOLD_ID:
#                 idx = HOLD_ID.index(hold_id)
#                 x, y = HOLDCOORDINATES[idx]
#                 func_numeric = self.func_map.get(func, 0)
#                 holds_with_coords.append([float(idx), float(func_numeric), x, y])
        
#         if not holds_with_coords:
#             return np.zeros(16)
        
#         holds_with_coords.sort(key=lambda h: h[3])
        
#         sequence = [[h[0], h[1], float(angle)] for h in holds_with_coords[:self.max_holds]]
#         seq_len = len(sequence)
        
#         while len(sequence) < self.max_holds:
#             sequence.append([0.0, 0.0, 0.0])
        
#         sequence = torch.FloatTensor(sequence).unsqueeze(0)
#         lengths = torch.tensor([seq_len])
        
#         with torch.no_grad():
#             embedding = self.lstm(sequence, lengths)
#         return embedding.squeeze().cpu().detach().numpy()
    
#     def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
#         angles = df['angle_y'].fillna(0).values
#         return np.array([
#             self.encode_single(row['holds_data'], angle)
#             for (_, row), angle in zip(df.iterrows(), angles)
#         ])

# class NeuralNetRegressor:
    
#     def __init__(self, model, epochs=50, lr=0.001, batch_size=32):
#         self.model = model
#         self.epochs = epochs
#         self.lr = lr
#         self.batch_size = batch_size
#         self.device = torch.device('cpu')  # ✅ Force CPU
#         self.model = self.model.to(self.device)
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         self.criterion = nn.MSELoss()
    
#     def fit(self, X, y):
#         dataset = TensorDataset(
#             torch.FloatTensor(X), 
#             torch.FloatTensor(y).reshape(-1, 1)
#         )
#         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
#         self.model.train()
#         for epoch in range(self.epochs):
#             epoch_loss = 0
#             for batch_X, batch_y in loader:
#                 batch_X = batch_X.to(self.device)
#                 batch_y = batch_y.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(batch_X)
#                 loss = self.criterion(outputs, batch_y)
#                 loss.backward()
#                 self.optimizer.step()
#                 epoch_loss += loss.item()
            
#             if (epoch + 1) % 5 == 0:
#                 print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(loader):.4f}")
    
#     def predict(self, X):
#         self.model.eval()
#         predictions = []
        
#         # ✅ Predict in batches to avoid memory issues
#         with torch.no_grad():
#             for i in range(0, len(X), self.batch_size):
#                 batch = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)
#                 preds = self.model(batch)
#                 predictions.append(preds.cpu().numpy())
        
#         return np.concatenate(predictions).flatten()


# class CNNRegressor(NeuralNetRegressor):
    
#     def fit(self, X, y):
#         # ✅ Don't reshape all data at once
#         class CNNDataset(torch.utils.data.Dataset):
#             def __init__(self, X, y):
#                 self.X = X
#                 self.y = y
            
#             def __len__(self):
#                 return len(self.X)
            
#             def __getitem__(self, idx):
#                 x = self.X[idx, :-1].reshape(4, 48, 48)
#                 return torch.FloatTensor(x), torch.FloatTensor([self.y[idx]])
        
#         dataset = CNNDataset(X, y)
#         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
#         self.model.train()
#         for epoch in range(self.epochs):
#             epoch_loss = 0
#             for batch_X, batch_y in loader:
#                 batch_X = batch_X.to(self.device)
#                 batch_y = batch_y.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(batch_X)
#                 loss = self.criterion(outputs, batch_y)
#                 loss.backward()
#                 self.optimizer.step()
#                 epoch_loss += loss.item()
            
#             if (epoch + 1) % 5 == 0:
#                 print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(loader):.4f}")
    
#     def predict(self, X):
#         self.model.eval()
#         predictions = []
        
#         with torch.no_grad():
#             for i in range(0, len(X), self.batch_size):
#                 batch = X[i:i+self.batch_size, :-1].reshape(-1, 4, 48, 48)
#                 batch_tensor = torch.FloatTensor(batch).to(self.device)
#                 preds = self.model(batch_tensor)
#                 predictions.append(preds.cpu().numpy())
        
#         return np.concatenate(predictions).flatten()


# class SimpleCNN(nn.Module):
    
#     def __init__(self, input_channels=4):
#         super().__init__()
#         # ✅ Smaller model
#         self.conv1 = nn.Conv2d(input_channels, 8, 3, padding=1)
#         self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(16 * 12 * 12, 32)
#         self.fc2 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)


# class SimpleDense(nn.Module):
    
#     def __init__(self, input_dim):
#         super().__init__()
#         # ✅ Smaller model
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
    
#     def forward(self, x):
#         x = self.dropout(self.relu(self.fc1(x)))
#         x = self.dropout(self.relu(self.fc2(x)))
#         return self.fc3(x)
    
    
# class ExperimentRunner:
    
#     def __init__(self):
#         self.dp = DataPreprocessing()
#         self.evaluator = Evaluation()
    
#     def run(self, encoder, model, routes_df: pd.DataFrame, 
#             experiment_name: str, test_size: float = 0.2) -> dict:
#         import time
        
#         print(f"\n{'='*60}")
#         print(f"Experiment: {experiment_name}")
#         print(f"{'='*60}")
        
#         start_time = time.time()
        
#         print("Encoding features...")
#         X = encoder.encode_dataframe(routes_df)
#         y = routes_df['display_difficulty'].values
#         print(f"Feature shape: {X.shape}")
        
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42, stratify=routes_df['v_grade']
#         )
        
#         print("Training model...")
#         model.fit(X_train, y_train)
#         train_time = time.time() - start_time
#         print(f"Training time: {train_time:.2f}s")
        
#         y_pred = model.predict(X_test)
#         scores = self.evaluator.get_scores(y_test, y_pred, experiment_name)
        
#         return {
#             'name': experiment_name,
#             'scores': scores,
#             'train_time': train_time,
#             'model': model,
#             'X_test': X_test,
#             'y_test': y_test,
#             'y_pred': y_pred
#         }
        
# if __name__ == "__main__":
#     dp = DataPreprocessing()
#     routes = dp.load_climbs().to_pandas()
#     import ast
#     routes['holds_data'] = routes['holds_data'].apply(ast.literal_eval)
#     print(f"Dataset: {len(routes)} routes")
    
#     runner = ExperimentRunner()
#     results = []
    
#     # Experiment 1: Matrix + XGBoost
#     # results.append(runner.run(
#     #     MatrixEncoder(grid_size=(48, 48)),
#     #     XGBRegressor(n_estimators=100, random_state=42),
#     #     routes,
#     #     "Matrix + XGBoost"
#     #     ))
    
#     # Experiment 2: Stats + GradientBoosting
#     # results.append(runner.run(
#     #     StatsEncoder(),
#     #     XGBRegressor(n_estimators=100, random_state=42),
#     #     routes,
#     #     "Stats + XGBoost"
#     #     ))
    
#     # Experiment 3: Matrix + Stats + XGBoost
#     # results.append(runner.run(
#     #     MatrixStatsEncoder(grid_size=(48, 48)),
#     #     XGBRegressor(n_estimators=100, random_state=42),
#     #     routes,
#     #     "Matrix+Stats + XGBoost"
#     #     ))
    
    
#     # Experiment 4: Sequence + XGBoost
#     # results.append(runner.run(
#     #     SequenceEncoder(max_holds=20),
#     #     XGBRegressor(n_estimators=100, random_state=42),
#     #     routes,
#     #     "Sequence + XGBoost"
#     #     ))
    
#     # Experiment 5: Matrix + CNN
#     results.append(runner.run(
#         MatrixEncoder(grid_size=(48, 48)),
#         CNNRegressor(SimpleCNN(input_channels=4), epochs=30),
#         routes,
#         "Matrix + CNN"
#         ))
    
#     # Experiment 6: Stats + Dense NN
#     results.append(runner.run(
#         StatsEncoder(),
#         NeuralNetRegressor(SimpleDense(input_dim=18), epochs=50),
#         routes,
#         "Stats + Dense NN"
#         ))
    
#     # Compare all
#     print("\n" + "="*60)
#     print("RESULTS SUMMARY")
#     print("="*60)
#     comparison = pd.DataFrame([
#         {
#             'Method': r['name'],
#             'R²': r['scores']['r2'],
#             'MAE': r['scores']['mae'],
#             'RMSE': r['scores']['rmse'],
#             'Train Time (s)': r['train_time'],
#         }
#         for r in results
#     ])
#     print(comparison.to_string(index=False))
    
#     # Find best method
#     best = comparison.loc[comparison['R²'].idxmax()]
#     print(f"\nBest Method: {best['Method']}")
#     print(f"R²: {best['R²']:.4f}, MAE: {best['MAE']:.4f}, RMSE: {best['RMSE']:.4f}")
#     print(f"Training Time: {best['Train Time (s)']:.2f}s")


"""
Feature encoding methods for route difficulty prediction.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import pdist
import gc

from .data_processing import HOLD_ID, HOLDCOORDINATES, DataPreprocessing
from .evaluation import Evaluation


class MatrixEncoder:
    def __init__(self, grid_size=(48, 48)):
        self.grid_size = grid_size
        self.func_map = {12: 0, 13: 1, 14: 2, 15: 3}
        self.hold_coords = {
            hold_id: (
                min(int(HOLDCOORDINATES[idx][0] / 22.5), grid_size[0] - 1),
                min(int(HOLDCOORDINATES[idx][1] / 24.4), grid_size[1] - 1)
            )
            for idx, hold_id in enumerate(HOLD_ID)
        }
    
    def encode(self, holds_data: list, angle: float) -> np.ndarray:
        grids = np.zeros((4, *self.grid_size), dtype=np.float32)
        for hold in holds_data:
            hold_id, func = list(hold.items())[0]
            if hold_id in self.hold_coords:
                grid_x, grid_y = self.hold_coords[hold_id]
                grids[self.func_map.get(func, 0), grid_x, grid_y] = 1
        return np.concatenate([grids.flatten(), [angle]])


class StatsEncoder:
    def __init__(self):
        self.hold_to_idx = {hold_id: idx for idx, hold_id in enumerate(HOLD_ID)}
    
    def encode(self, holds_data: list, angle: float) -> np.ndarray:
        if not holds_data:
            return np.zeros(18)
        
        coords, funcs = [], []
        for hold in holds_data:
            hold_id, func = list(hold.items())[0]
            if hold_id in self.hold_to_idx:
                coords.append(HOLDCOORDINATES[self.hold_to_idx[hold_id]])
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
            np.max(y) - np.min(y), np.max(x) - np.min(x), angle, np.sum(y) / len(coords)
        ])


class RouteDataset(Dataset):
    def __init__(self, holds_list, angles, labels, encoder, is_cnn=False):
        self.holds_list = holds_list
        self.angles = angles
        self.labels = labels
        self.encoder = encoder
        self.is_cnn = is_cnn
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.encoder.encode(self.holds_list[idx], self.angles[idx])
        if self.is_cnn:
            x = x[:-1].reshape(4, 48, 48)
        return torch.FloatTensor(x), torch.FloatTensor([self.labels[idx]])


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.regressor(self.features(x))


class SimpleDense(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class Trainer:
    def __init__(self, model, encoder, is_cnn=False, epochs=20, batch_size=256, lr=0.001):
        self.model = model
        self.encoder = encoder
        self.is_cnn = is_cnn
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    def fit(self, train_holds, train_angles, train_labels, val_holds=None, val_angles=None, val_labels=None):
        train_dataset = RouteDataset(train_holds, train_angles, train_labels, self.encoder, self.is_cnn)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                 num_workers=2, pin_memory=True)
        
        val_loader = None
        if val_holds is not None:
            val_dataset = RouteDataset(val_holds, val_angles, val_labels, self.encoder, self.is_cnn)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=2, pin_memory=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader:
                val_loss = self._validate(val_loader)
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}")
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                val_loss += self.criterion(outputs, batch_y).item()
        return val_loss / len(val_loader)
    
    def predict(self, holds_list, angles):
        self.model.eval()
        dataset = RouteDataset(holds_list, angles, np.zeros(len(holds_list)), self.encoder, self.is_cnn)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                preds = self.model(batch_X)
                predictions.append(preds.cpu().numpy())
        
        return np.concatenate(predictions).flatten()


class ExperimentRunner:
    def __init__(self):
        self.evaluator = Evaluation()
    
    def run(self, encoder, model, routes_df: pd.DataFrame, experiment_name: str, 
            test_size=0.2, val_size=0.1, max_samples=None) -> dict:
        import time
        
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*60}")
        
        if max_samples and len(routes_df) > max_samples:
            routes_df = routes_df.sample(n=max_samples, random_state=42)
            print(f"Subsampled to {max_samples} routes")
        
        start_time = time.time()
        
        holds = routes_df['holds_data'].values
        angles = routes_df['angle_y'].fillna(0).values
        labels = routes_df['display_difficulty'].values
        
        # Split: train/test, then train/val
        train_idx, test_idx = train_test_split(
            range(len(routes_df)), test_size=test_size, random_state=42,
            stratify=routes_df['v_grade']
        )
        
        if isinstance(model, Trainer):
            train_idx, val_idx = train_test_split(
                train_idx, test_size=val_size, random_state=42
            )
            
            print(f"Training with validation ({len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test)")
            model.fit(
                holds[train_idx], angles[train_idx], labels[train_idx],
                holds[val_idx], angles[val_idx], labels[val_idx]
            )
            y_pred = model.predict(holds[test_idx], angles[test_idx])
            y_test = labels[test_idx]
        else:
            print("Encoding features...")
            X = np.array([encoder.encode(h, a) for h, a in zip(holds, angles)])
            print(f"Feature shape: {X.shape}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            print("Training model...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            del X, X_train, X_test
            gc.collect()
        
        train_time = time.time() - start_time
        scores = self.evaluator.get_scores(y_test, y_pred, experiment_name)
        
        print(f"Training time: {train_time:.2f}s")
        print(f"R²: {scores['r2']:.4f}, MAE: {scores['mae']:.4f}, RMSE: {scores['rmse']:.4f}")
        
        return {'name': experiment_name, 'scores': scores, 'train_time': train_time}


if __name__ == "__main__":
    dp = DataPreprocessing()
    routes = dp.load_climbs().to_pandas()
    import ast
    routes['holds_data'] = routes['holds_data'].apply(ast.literal_eval)
    print(f"Dataset: {len(routes)} routes")
    
    runner = ExperimentRunner()
    results = []
    
    results.append(runner.run(
        MatrixEncoder(),
        XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        routes,
        "Matrix + XGBoost"
    ))
    
    results.append(runner.run(
        StatsEncoder(),
        XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        routes,
        "Stats + XGBoost"
    ))
    
    matrix_enc = MatrixEncoder()
    results.append(runner.run(
        matrix_enc,
        Trainer(SimpleCNN(), matrix_enc, is_cnn=True, epochs=30, batch_size=128),
        routes,
        "Matrix + CNN",
        max_samples=30000
    ))
    
    stats_enc = StatsEncoder()
    results.append(runner.run(
        stats_enc,
        Trainer(SimpleDense(18), stats_enc, epochs=30, batch_size=256),
        routes,
        "Stats + Dense NN",
        max_samples=30000
    ))
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    comparison = pd.DataFrame([{
        'Method': r['name'],
        'R²': r['scores']['r2'],
        'MAE': r['scores']['mae'],
        'RMSE': r['scores']['rmse'],
        'Time (s)': r['train_time']
    } for r in results])
    print(comparison.to_string(index=False))
    
    best = comparison.loc[comparison['R²'].idxmax()]
    print(f"\nBest: {best['Method']} (R²={best['R²']:.4f}, MAE={best['MAE']:.4f})")