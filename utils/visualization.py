# """
# Visualization utilities for Kilter Board routes.
# """
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from PIL import Image
# from collections import defaultdict
# from sklearn.decomposition import PCA
# from .data_processing import HOLD_ID, HOLDCOORDINATES
# from .data_processing import DataPreprocessing

# HOLD_COLORS = {12: "#00DD00", 13: "#00FFFF", 14: "#FF00FF", 15: "#FFA500"}
# CIRC_RADIUS = 30
# CIRC_STROKE = 2


# class Visualization:
#     """Visualization methods for boulder routes."""
    
#     def __init__(self, board_img_path: str = "src/full_board_commercial.png"):
#         self.board_img = Image.open(board_img_path).convert("RGBA")
    
#     def plot_hold_ids(self, show_ids: bool = True):
#         """Display hold IDs on board."""
#         fig, ax = plt.subplots(figsize=(10, 12))
#         ax.imshow(self.board_img)
        
#         for hold_id in HOLD_ID:
#             idx = HOLD_ID.index(hold_id)
#             x, y = HOLDCOORDINATES[idx]
#             if show_ids:
#                 ax.text(x, y, f"{hold_id}", color="#000", fontsize=6, ha="center", va="center",
#                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
#             else:
#                 ax.plot(x, y, 'ro', markersize=3)
        
#         ax.axis("off")
#         plt.tight_layout()
#         plt.show()
    
#     def plot_heatmap(self, hold_values: dict, title: str = "", cmap: str = 'Reds',
#                      hold_type: str = 'all', alpha: float = 0.4):
#         """Overlay heatmap values on board."""
#         fig, ax = plt.subplots(figsize=(5, 6))
#         ax.imshow(self.board_img)
        
#         holds_to_plot = HOLD_ID
#         if hold_type == 'main':
#             holds_to_plot = [h for h in HOLD_ID if h < 1447]
#         elif hold_type == 'auxiliary':
#             holds_to_plot = [h for h in HOLD_ID if h >= 1447]
        
#         values = [hold_values.get(h, 0) for h in holds_to_plot]
#         norm = plt.Normalize(vmin=min(values), vmax=max(values))
#         cm = plt.colormaps[cmap]
#         cell_size = CIRC_RADIUS * 2
        
#         for hold_id in holds_to_plot:
#             idx = HOLD_ID.index(hold_id)
#             x, y = HOLDCOORDINATES[idx]
#             color = cm(norm(hold_values.get(hold_id, 0)))
#             rect = plt.Rectangle((x - cell_size/2, y - cell_size/2), cell_size, cell_size,
#                                facecolor=color, edgecolor="none", alpha=alpha)
#             ax.add_patch(rect)
        
#         sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
#         sm.set_array([])
#         cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
#         cbar.set_label(title, rotation=270, labelpad=15)
        
#         ax.axis("off")
#         ax.set_title(title)
#         plt.tight_layout()
#         plt.show()
    
#     def plot_boulder(self, holds_data: list, name: str = "", v_grade: str = ""):
#         """Visualize specific route."""
#         fig, ax = plt.subplots(figsize=(5, 6))
#         ax.imshow(self.board_img)
        
#         for hold in holds_data:
#             hold_id, func = list(hold.items())[0]
#             if hold_id in HOLD_ID:
#                 idx = HOLD_ID.index(hold_id)
#                 x, y = HOLDCOORDINATES[idx]
#                 color = HOLD_COLORS.get(func, "#000000")
#                 circle = plt.Circle((x, y), radius=CIRC_RADIUS, facecolor='none',
#                                   linestyle='-', edgecolor=color, linewidth=CIRC_STROKE)
#                 ax.add_patch(circle)
        
#         ax.axis("off")
#         title = f"{name} ({v_grade})" if name and v_grade else name or v_grade
#         ax.set_title(title)
#         plt.tight_layout()
#         plt.show()
    
#     def plot_clustering(self, X: np.ndarray, routes_df, cluster_col: str = 'style_cluster'):
#         """Visualize clusters in PCA space."""
#         X_2d = PCA(n_components=2).fit_transform(X)
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
#         sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=routes_df[cluster_col],
#                        palette="Set2", s=50, alpha=0.7, ax=ax1)
#         ax1.set_title("Route Clusters by Style")
#         ax1.set_xlabel("PCA Component 1")
#         ax1.set_ylabel("PCA Component 2")
        
#         scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=routes_df['display_difficulty'],
#                             s=routes_df['display_difficulty'] * 5, alpha=0.5,
#                             cmap='viridis', linewidth=0.5)
#         ax2.set_title("Routes by Difficulty")
#         ax2.set_xlabel("PCA Component 1")
#         ax2.set_ylabel("PCA Component 2")
#         plt.colorbar(scatter, ax=ax2, label='Difficulty Grade')
        
#         plt.tight_layout()
#         plt.show()


# if __name__ == "__main__":
    
#     dp = DataPreprocessing()
#     routes = dp.load_routes()
    
#     viz = Visualization()
    
#     # Hold frequency heatmap
#     hold_freq = defaultdict(int)
#     for holds in routes['holds_data']:
#         for hold in holds:
#             hold_freq[list(hold.keys())[0]] += 1
    
#     viz.plot_heatmap(dict(hold_freq), title="Hold Frequency", hold_type='main')
    
#     # Plot sample boulder
#     sample = routes.iloc[0]
#     viz.plot_boulder(sample['holds_data'], sample['name'], sample['v_grade'])

"""
Visualization utilities for Kilter Board routes.
Includes embedding analysis and route similarity visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from PIL import Image
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple
from torch.utils.data import DataLoader

from .data_processing import HOLD_ID, HOLDCOORDINATES, DataPreprocessing

HOLD_COLORS = {12: "#00DD00", 13: "#00FFFF", 14: "#FF00FF", 15: "#FFA500"}
CIRC_RADIUS = 30
CIRC_STROKE = 2

class Visualization:
    """Visualization methods for boulder routes and model embeddings."""
    
    def __init__(self, board_img_path: str = "src/full_board_commercial.png"):
        self.board_img = Image.open(board_img_path).convert("RGBA")
    
    def plot_difficulty_quality_analysis(self, routes_df: pd.DataFrame, dp: DataPreprocessing):
        """
        Plot difficulty distribution and quality vs difficulty relationship with V-grade labels.
        
        Args:
            routes_df: DataFrame with 'display_difficulty' and 'quality_average' columns
            dp: DataPreprocessing instance for grade conversion
        """
        # Pre-build V-grade mapping once (FAST)
        v_grades_map = {}
        for diff_id, grade_str in dp.grade_dict.items():
            v_grade = grade_str.split('/')[0]  # Extract V-grade from "V5/6B"
            v_grades_map[diff_id] = v_grade
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Difficulty histogram ---
        diff_data = routes_df['display_difficulty'].dropna()
        ax1.hist(diff_data, bins=np.arange(diff_data.min(), diff_data.max()+2)-0.5,
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Difficulty Grade')
        ax1.set_ylabel('Route Count')
        ax1.set_title('Difficulty Distribution')
        ax1.grid(True, alpha=0.3)

        # --- Quality vs Difficulty ---
        quality_data = routes_df[['display_difficulty', 'quality_average']].dropna()
        ax2.scatter(quality_data['display_difficulty'], quality_data['quality_average'], alpha=0.5)
        ax2.set_xlabel('Difficulty Grade')
        ax2.set_ylabel('Quality Rating')
        ax2.set_title('Quality vs Difficulty')
        ax2.grid(True, alpha=0.3)

        # --- Apply V-grade labels to both axes ---
        def apply_v_labels(ax, data):
            ticks = sorted(data['display_difficulty'].round().astype(int).unique())
            labels = [v_grades_map.get(int(t), f"V{int(t)}") for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        apply_v_labels(ax1, routes_df[['display_difficulty']].dropna())
        apply_v_labels(ax2, quality_data)

        plt.tight_layout()
        plt.show()
        plt.savefig('figs/difficulty_quality_analysis.png', dpi=150)
 
    def plot_hold_ids(self, show_ids: bool = True):
        """Display hold IDs on board."""
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(self.board_img)
        
        for hold_id in HOLD_ID:
            idx = HOLD_ID.index(hold_id)
            x, y = HOLDCOORDINATES[idx]
            if show_ids:
                ax.text(x, y, f"{hold_id}", color="#000", fontsize=6, ha="center", va="center",
                       bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
            else:
                ax.plot(x, y, 'ro', markersize=3)
        
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        plt.savefig('figs/hold_ids.png', dpi=150)
    
    def plot_heatmap(self, hold_values: dict, title: str = "", cmap: str = 'Reds',
                     hold_type: str = 'all', alpha: float = 0.4):
        """Overlay heatmap values on board."""
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.imshow(self.board_img)
        
        holds_to_plot = HOLD_ID
        if hold_type == 'main':
            holds_to_plot = [h for h in HOLD_ID if h < 1447]
        elif hold_type == 'auxiliary':
            holds_to_plot = [h for h in HOLD_ID if h >= 1447]
        
        values = [hold_values.get(h, 0) for h in holds_to_plot]
        
        norm = plt.Normalize(vmin=min(values), vmax=max(values))
        cm = plt.colormaps[cmap]
        cell_size = CIRC_RADIUS * 2
        
        for hold_id in holds_to_plot:
            idx = HOLD_ID.index(hold_id)
            x, y = HOLDCOORDINATES[idx]
            color = cm(norm(hold_values.get(hold_id, 0)))
            rect = plt.Rectangle((x - cell_size/2, y - cell_size/2), cell_size, cell_size,
                               facecolor=color, edgecolor="none", alpha=alpha)
            ax.add_patch(rect)
        
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title, rotation=270, labelpad=15)
        
        ax.axis("off")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'figs/{title}_{hold_type}_heatmap.png', dpi=150)
    
    def plot_boulder(self, holds_data: list, name: str = "", v_grade: str = "", predicted_v_grade: str = ""):
        """Visualize specific route."""
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.imshow(self.board_img)
        
        for hold in holds_data:
            hold_id, func = list(hold.items())[0]
            if hold_id in HOLD_ID:
                idx = HOLD_ID.index(hold_id)
                x, y = HOLDCOORDINATES[idx]
                color = HOLD_COLORS.get(func, "#000000")
                circle = plt.Circle((x, y), radius=CIRC_RADIUS, facecolor='none',
                                  linestyle='-', edgecolor=color, linewidth=CIRC_STROKE)
                ax.add_patch(circle)
        
        ax.axis("off")
        title = f"{name} | Actual: {v_grade} | Predicted: {predicted_v_grade}" if predicted_v_grade else\
                f"{name} ({v_grade})"
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'figs/{name}_{v_grade}.png', dpi=150)
        
    
    def plot_clustering(self, X: pd.DataFrame, routes_df: pd.DataFrame, 
                   cluster_col: str = 'style_cluster', n_representatives: int = None,
                   plot_boulders: bool = False):
        """
        Visualize clustering results with optional representative routes.
        
        Args:
            X: Feature matrix
            routes_df: Routes dataframe with cluster assignments
            cluster_col: Column name containing cluster labels
            n_representatives: Number of cluster representatives to show (None = all)
            plot_boulders: Whether to visualize representative routes on board
        """
        
        # PCA 
        X_2d = PCA(n_components=2).fit_transform(X)
        n_clusters = routes_df[cluster_col].nunique()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cluster colored
        sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=routes_df[cluster_col],
                    palette="Set2", s=50, alpha=0.7, ax=ax1)
        ax1.set_title(f"Route Clusters by Style (k={n_clusters})")
        ax1.set_xlabel("PCA Component 1")
        ax1.set_ylabel("PCA Component 2")
        
        # Difficulty colored
        scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=routes_df['display_difficulty'],
                            s=50, alpha=0.5, cmap='viridis')
        ax2.set_title("Routes by Difficulty")
        ax2.set_xlabel("PCA Component 1")
        ax2.set_ylabel("PCA Component 2")
        plt.colorbar(scatter, ax=ax2, label='Difficulty')
        
        plt.tight_layout()
        plt.savefig(f'figs/clustering_k{n_clusters}.png', dpi=150)
        plt.show()
        
        # Find representatives
        clusters_to_show = range(n_clusters) if n_representatives is None else range(min(n_representatives, n_clusters))
        
        for cluster_id in clusters_to_show:
            cluster_mask = routes_df[cluster_col] == cluster_id
            cluster_routes = routes_df[cluster_mask]
            
            # Pick median difficulty route as representative
            median_diff = cluster_routes['display_difficulty'].median()
            rep_idx = (cluster_routes['display_difficulty'] - median_diff).abs().idxmin()
            rep = routes_df.loc[rep_idx]
            
            print(f"\nCluster {cluster_id}: {rep['name']} | V{rep['v_grade']} | {rep['angle_y']:.0f}°")
            
            if plot_boulders:
                self.plot_boulder(rep['holds_data'], 
                                name=f"Cluster {cluster_id}: {rep['name']}", 
                                v_grade=f"V{rep['v_grade']}")
        

    
    def plot_correlation(self, df: pd.DataFrame, save_dir: str = 'figs'):
        """Full correlation heatmap between all features."""
        
        key_features = [
            'display_difficulty', 'angle_y', 'num_holds', 'num_hand_holds',
            'num_foot_holds', 'hand_foot_ratio', 'avg_reach', 'max_reach',
            'route_area', 'hold_density', 'ascensionist_count', 'quality_average',
            'popularity_score', 'angle_x_holds', 'density_x_angle'
        ]
        key_features = [f for f in key_features if f in df.columns]
        df_clean = df[key_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        corr_matrix = df_clean.corr()
        
        plt.figure(figsize=(16, 9))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1,
                    cbar_kws={"label": "Correlation"}, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return corr_matrix


    def plot_corr_with_difficulty(self, df: pd.DataFrame, save_dir: str = 'figs'):
        """Feature correlations with difficulty target variable."""
        
        key_features = [
            'angle_y', 'num_holds', 'num_hand_holds', 'num_foot_holds', 
            'hand_foot_ratio', 'avg_reach', 'max_reach', 'route_area', 
            'hold_density', 'ascensionist_count', 'quality_average',
            'popularity_score', 'angle_x_holds', 'density_x_angle'
        ]
        key_features = [f for f in key_features if f in df.columns]
        df_clean = df[key_features + ['display_difficulty']].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        diff_corr = df_clean.corr()['display_difficulty'].drop('display_difficulty').sort_values()
        
        plt.figure(figsize=(10, 8))
        colors = ['crimson' if x < 0 else 'steelblue' for x in diff_corr.values]
        bars = plt.barh(diff_corr.index, diff_corr.values, color=colors, edgecolor='black')
        plt.axvline(0, color='black', linewidth=0.8)
        plt.xlabel('Correlation with Difficulty', fontsize=12)
        plt.title('Feature Importance for Difficulty Prediction', fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, diff_corr.values)):
            plt.text(val + (0.02 if val > 0 else -0.02), i, f'{val:.2f}',
                    va='center', ha='left' if val > 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/difficulty_correlation.png', dpi=150, bbox_inches='tight')
        plt.show()


    def plot_distribution(self, df: pd.DataFrame, save_dir: str = 'figs'):
        """Feature distributions by grade and quality-popularity relationship."""
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        
        # Box plots: grade vs. features
        box_features = ['angle_y', 'num_holds', 'ascensionist_count', 
                        'quality_average', 'avg_reach']
        
        # Sort grades naturally (V1, V2, ..., V11)
        
        grade_order = sorted(df['v_grade'].unique(), key=lambda x: int(str(x).replace('V', '').replace('+', '').replace('-', '')))
        
        for idx, feat in enumerate(box_features):
            if feat in df.columns:
                ax = axes[idx]
                data = df[[feat, 'v_grade']].dropna()
                sns.boxplot(data=data, x='v_grade', y=feat, ax=ax, 
                        order=grade_order, palette='viridis', showfliers=False)
                ax.set_title(f'{feat} by Grade', fontsize=11, fontweight='bold')
                ax.set_xlabel('Grade', fontsize=9)
                ax.set_ylabel(feat, fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(axis='y', alpha=0.3)
        
        # Scatter: quality_average (x) vs ascensionist_count (y, log scale)
        if 'ascensionist_count' in df.columns and 'quality_average' in df.columns:
            ax = axes[5]
            scatter_data = df[['ascensionist_count', 'quality_average']].dropna()
            scatter_data = scatter_data[scatter_data['ascensionist_count'] > 0]
            
            ax.scatter(scatter_data['quality_average'], 
                    scatter_data['ascensionist_count'],
                    alpha=0.4, s=15, c='steelblue', edgecolors='none')
            ax.set_yscale('log')
            ax.set_xlabel('Quality Average', fontsize=10)
            ax.set_ylabel('Ascensionist Count (log scale)', fontsize=10)
            ax.set_title('Quality vs Popularity', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/distributions.png', dpi=150, bbox_inches='tight')
        plt.show()


    
    
    
    
    # ===== NEW EMBEDDING VISUALIZATION METHODS =====
    
    def get_embeddings(self, encoder, routes_df: pd.DataFrame, 
                      dataset_class, layer: str = 'combined') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from trained model.
        
        Args:
            encoder: Trained encoder instance
            routes_df: DataFrame with routes
            dataset_class: Dataset class to use (ImprovedBoulderDataset)
            layer: 'cls', 'combined', or 'attention'
        
        Returns:
            embeddings: (n_routes, hidden_dim) array
            difficulties: (n_routes,) array
        """
        dataset = dataset_class(routes_df, encoder.vocab, encoder.max_length)
        loader = DataLoader(dataset, batch_size=64)
        
        encoder.model.eval()
        embeddings = []
        difficulties = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(encoder.device)
                pos_x = batch['position_x'].to(encoder.device)
                pos_y = batch['position_y'].to(encoder.device)
                metadata = batch['metadata'].to(encoder.device)
                mask = batch['attention_mask'].to(encoder.device)
                
                outputs = encoder.model.bert(
                    input_ids=input_ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                if layer == 'cls':
                    emb = outputs.last_hidden_state[:, 0]
                elif layer == 'combined':
                    # Multi-scale pooling + metadata
                    hidden_states = outputs.hidden_states[-4:]
                    cls_tokens = torch.stack([h[:, 0] for h in hidden_states], dim=1)
                    pooled_cls = cls_tokens.mean(dim=1)
                    
                    # Add positional info
                    pos_x_emb = encoder.model.pos_x_embed(pos_x).mean(dim=1)
                    pos_y_emb = encoder.model.pos_y_embed(pos_y).mean(dim=1)
                    pos_emb = torch.cat([pos_x_emb, pos_y_emb], dim=-1)
                    pooled_cls = pooled_cls + pos_emb
                    
                    meta_emb = encoder.model.metadata_mlp(metadata)
                    emb = torch.cat([pooled_cls, meta_emb], dim=1)
                elif layer == 'attention':
                    mask_expanded = mask.unsqueeze(-1)
                    emb = (outputs.last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1)
                
                embeddings.append(emb.cpu().numpy())
                difficulties.extend(batch['difficulty'].squeeze().numpy().tolist())
        
        return np.vstack(embeddings), np.array(difficulties)
    
    def plot_embedding_space(self, encoder, routes_df: pd.DataFrame, dataset_class,
                            method: str = 'pca', layer: str = 'combined',
                            color_by: str = 'difficulty'):
        """Visualize embedding space in 2D."""
        embeddings, difficulties = self.get_embeddings(encoder, routes_df, dataset_class, layer)
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
            title = f"PCA of Boulder Embeddings ({layer})"
            explained_var = reducer.explained_variance_ratio_
            xlabel = f"PC1 ({explained_var[0]:.1%} variance)"
            ylabel = f"PC2 ({explained_var[1]:.1%} variance)"
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = reducer.fit_transform(embeddings)
            title = f"t-SNE of Boulder Embeddings ({layer})"
            xlabel, ylabel = "t-SNE 1", "t-SNE 2"
        
        plt.figure(figsize=(12, 8))
        
        if color_by == 'difficulty':
            scatter = plt.scatter(coords[:, 0], coords[:, 1], c=difficulties, 
                                cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Difficulty')
        else:
            v_grades = routes_df['v_grade'].values
            unique_grades = sorted(np.unique(v_grades))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_grades)))
            
            for i, grade in enumerate(unique_grades):
                mask = v_grades == grade
                plt.scatter(coords[mask, 0], coords[mask, 1], 
                          c=[colors[i]], label=f'V{grade}', alpha=0.6, s=50)
            plt.legend(title='V-Grade', bbox_to_anchor=(1.15, 1))
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'figs/embeddings_{method}_{layer}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_embedding_heatmap(self, encoder, routes_df: pd.DataFrame, dataset_class,
                              n_samples: int = 20, layer: str = 'combined'):
        """Show heatmap of embeddings for sample routes."""
        sample_df = routes_df.groupby('v_grade').apply(
            lambda x: x.sample(min(3, len(x)), random_state=42)
        ).reset_index(drop=True).head(n_samples)
        
        embeddings, _ = self.get_embeddings(encoder, sample_df, dataset_class, layer)
        
        labels = [f"{row['name'][:15]}... (V{row['v_grade']}, {row['display_difficulty']:.1f})" 
                 for _, row in sample_df.iterrows()]
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(embeddings, cmap='coolwarm', center=0, yticklabels=labels,
                   cbar_kws={'label': 'Activation'}, xticklabels=False)
        plt.xlabel('Hidden Dimensions', fontsize=12)
        plt.ylabel('Boulder Routes', fontsize=12)
        plt.title(f'Boulder Embeddings Heatmap ({layer})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'figs/embeddings_heatmap_{layer}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_hold_embeddings(self, encoder):
        """Show learned embeddings for hold tokens."""
        embedding_layer = encoder.model.bert.embeddings.word_embeddings
        hold_tokens = list(range(3, len(encoder.vocab)))
        hold_embeddings = embedding_layer.weight[hold_tokens].detach().cpu().numpy()
        
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(hold_embeddings)
        
        vocab_items = [(k, v) for k, v in encoder.vocab.items() if v >= 3]
        vocab_items.sort(key=lambda x: x[1])
        
        hand_or_foot = [int(k.split('_')[1]) for k, _ in vocab_items]
        
        plt.figure(figsize=(14, 10))
        
        colors = {0: 'blue', 1: 'red'}
        labels = {0: 'Foot', 1: 'Hand/Start/Finish'}
        
        for func, color in colors.items():
            mask = np.array(hand_or_foot) == func
            plt.scatter(coords[mask, 0], coords[mask, 1], 
                       c=color, label=labels[func], alpha=0.6, s=30)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Learned Hold Token Embeddings', fontsize=14, fontweight='bold')
        plt.legend(title='Hold Type')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figs/hold_embeddings.png', dpi=150)
        plt.show()
        
        print("\nEmbedding Statistics:")
        print(f"Embedding dimension: {hold_embeddings.shape[1]}")
        print(f"Number of hold tokens: {len(hold_tokens)}")
        print(f"Mean activation: {hold_embeddings.mean():.4f}")
        print(f"Std activation: {hold_embeddings.std():.4f}")
    
    def find_similar_routes(self, encoder, routes_df: pd.DataFrame, dataset_class,
                           route_idx: int, n_similar: int = 5, layer: str = 'combined'):
        """Find and display routes with similar embeddings."""
        embeddings, _ = self.get_embeddings(encoder, routes_df, dataset_class, layer)
        
        target_embedding = embeddings[route_idx:route_idx+1]
        similarities = cosine_similarity(target_embedding, embeddings)[0]
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        target_route = routes_df.iloc[route_idx]
        print(f"\n{'='*70}")
        print(f"TARGET ROUTE")
        print(f"{'='*70}")
        print(f"Name: {target_route['name']}")
        print(f"Grade: V{target_route['v_grade']} ({target_route['display_difficulty']:.1f})")
        print(f"Angle: {target_route.get('angle_y', 'N/A')}°")
        
        print(f"\n{'='*70}")
        print(f"MOST SIMILAR ROUTES")
        print(f"{'='*70}")
        
        results = []
        for rank, idx in enumerate(similar_indices, 1):
            route = routes_df.iloc[idx]
            sim = similarities[idx]
            print(f"{rank}. {route['name'][:40]:40} | V{route['v_grade']:2} | "
                  f"Diff: {route['display_difficulty']:4.1f} | Sim: {sim:.3f}")
            results.append({
                'rank': rank,
                'name': route['name'],
                'v_grade': route['v_grade'],
                'difficulty': route['display_difficulty'],
                'similarity': sim
            })
        
        return pd.DataFrame(results)
    
    def visualize_complete_analysis(self, encoder, routes_df: pd.DataFrame, 
                                   dataset_class, sample_route_idx: int = None):
        """Run complete visualization suite."""
        print("=" * 70)
        print("BOULDER ENCODING VISUALIZATION")
        print("=" * 70)
        
        print("\n1. Visualizing embedding space with PCA...")
        self.plot_embedding_space(encoder, routes_df, dataset_class, 
                                 method='pca', layer='combined', color_by='difficulty')
        
        print("\n2. Visualizing embedding space with t-SNE...")
        self.plot_embedding_space(encoder, routes_df, dataset_class,
                                 method='tsne', layer='combined', color_by='v_grade')
        
        print("\n3. Plotting embedding heatmap...")
        self.plot_embedding_heatmap(encoder, routes_df, dataset_class, 
                                   n_samples=20, layer='combined')
        
        print("\n4. Analyzing learned hold embeddings...")
        self.analyze_hold_embeddings(encoder)
        
        if sample_route_idx is None:
            sample_route_idx = len(routes_df) // 2
        
        print("\n5. Finding similar routes...")
        self.find_similar_routes(encoder, routes_df, dataset_class,
                                sample_route_idx, n_similar=5, layer='combined')
         
        print("\n" + "=" * 70)
        print("Visualization complete! Check saved PNG files.")
        print("=" * 70)


if __name__ == "__main__":
    from .data_processing import DataPreprocessing
    
    #  Load model and data
    ####################### ERROR STILL #######################
    encoder = ImprovedTransformerEncoder(hidden_dim=128, num_layers=4)
    encoder.load_model('best_improved_20251001_162708.pt')

    dp = DataPreprocessing()
    routes = dp.load_routes()
    test_df = routes.sample(1000)  # Sample for faster visualization

    # Run complete analysis
    viz = Visualization()
    viz.visualize_complete_analysis(encoder, test_df, ImprovedBoulderDataset)

    # Or run individual visualizations
    viz.plot_embedding_space(encoder, test_df, ImprovedBoulderDataset, method='tsne')
    similar_df = viz.find_similar_routes(encoder, test_df, ImprovedBoulderDataset, 
                                        route_idx=50, n_similar=10)