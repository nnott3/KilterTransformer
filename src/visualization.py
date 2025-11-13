"""
Visualization utilities for Kilter Board routes.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
from typing import List
import os
import re
from src.data_processing import DataPreprocessing, HOLD_ID, HOLDCOORDINATES, HOLD_COLORS

class Visualization:
    """Visualization for Kilter Board routes."""
    
    def __init__(self, board_img_path="figs/full_board_commercial.png"):
        try:
            self.board_img = Image.open(board_img_path).convert("RGBA")
        except:  
            print(f"Warning: Could not load board image from {board_img_path}")
            self.board_img = None
    
    
    def plot_difficulty_quality_analysis(self, routes_df, dp, save_fig=False):
        """Plot difficulty distribution and quality vs difficulty."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Difficulty histogram
        diff_data = routes_df['display_difficulty'].dropna()
        ax1.hist(diff_data, bins=np.arange(diff_data.min(), diff_data.max()+2)-0.5,
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Difficulty Grade')
        ax1.set_ylabel('Route Count')
        ax1.set_title('Difficulty Distribution')
        ax1.grid(True, alpha=0.3)

        # Quality vs Difficulty
        quality_data = routes_df[['display_difficulty', 'quality_average']].dropna()
        ax2.scatter(quality_data['display_difficulty'], quality_data['quality_average'], alpha=0.5)
        ax2.set_xlabel('Difficulty Grade')
        ax2.set_ylabel('Quality Rating')
        ax2.set_title('Quality vs Difficulty')
        ax2.grid(True, alpha=0.3)

        # V-grade labels
        def apply_v_labels(ax, data):
            v_grades_map = {}
            for diff_id, grade_str in dp.grade_dict.items():
                v_grade = grade_str.split('/')[0]
                v_grades_map[diff_id] = v_grade
            ticks = sorted(data['display_difficulty'].round().astype(int).unique())
            labels = [v_grades_map.get(int(t), f"V{int(t)}") for t in ticks] 
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        apply_v_labels(ax1, routes_df[['display_difficulty']].dropna())
        apply_v_labels(ax2, quality_data)

        plt.tight_layout()
        if save_fig:
            plt.savefig('figs/difficulty_quality_analysis.png', dpi=150)
        plt.show()
    
    def plot_hold_ids(self, hold_ids: List[int]=[], all_holds=True, show_ids=True, save_fig=False):
        """Display hold IDs on board."""
        if self.board_img is None:
            return
        
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(self.board_img)
        if all_holds:
            title = "All holds"
            for hold_id in HOLD_ID:
                idx = HOLD_ID.index(hold_id)
                x, y = HOLDCOORDINATES[idx]
                if show_ids:
                    ax.text(x, y, f"{hold_id}", color="#000", fontsize=6, ha="center", va="center",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
                else:
                    ax.plot(x, y, 'ro', markersize=3)
        elif not all_holds and len(hold_ids) > 0:
            title = f"{len(hold_ids)} holds : {' '.join([str(h) for h in hold_ids][:5])}..."
            for hold_id in hold_ids:
                idx = HOLD_ID.index(hold_id)
                x, y = HOLDCOORDINATES[idx]
                if show_ids:
                    ax.text(x, y, f"{hold_id}", color="#000", fontsize=6, ha="center", va="center",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
                else:
                    ax.plot(x, y, 'ro', markersize=3)
            
        
        ax.axis("off")
        ax.set_title(title)
        plt.tight_layout()
        if save_fig:
            plt.savefig('figs/hold_ids.png', dpi=150)
        plt.show()
    
    def plot_heatmap(self, hold_values, title="", cmap='Reds', hold_type='all', alpha=0.4, save_fig=False):
        """Overlay heatmap values on board."""
        if self.board_img is None:
            return
        
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
        cell_size = 60
        
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
        if save_fig:
            plt.savefig(f'figs/{title}_{hold_type}_heatmap.png', dpi=150)
        plt.show()
    
    def plot_boulder(self, data, name="", angle="", v_grade="", predicted_v_grade="", save_fig=False):
        """
        Visualize route from string
        - angle40_grade18_start1111_hand2222_feet3333_finish4444
        Holds_data list of dict
        - [{'1078': 15}, {'1153': 12}, {'1184': 13}]
        """
        if self.board_img is None:
            return
        if isinstance(data, list):
            # no angle/grade
            holds_data = data
        elif isinstance(data, str):
            angle = int(re.search(r"angle(\d+)", data).group(1)) if re.search(r"angle(\d+)", data) else None
            grade = int(re.search(r"grade(\d+)", data).group(1)) if re.search(r"grade(\d+)", data) else None
            
            dp = DataPreprocessing()
            v_grade = dp.difficulty_to_v_grade(grade) if grade else None
            letter_grade = dp.difficulty_to_letter_grade(grade) if grade else None
            holds_data = dp.str_to_holds_data(data) #holds_data        
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(self.board_img)
        

        for hold in holds_data:
            for hold_id_str, role in hold.items():
                hold_id = int(hold_id_str)
                if hold_id in HOLD_ID and role is not None:
                    idx = HOLD_ID.index(hold_id)
                    x, y = HOLDCOORDINATES[idx]
                    color = HOLD_COLORS.get(role, "#000000")
                    circle = plt.Circle((x, y), radius=30, facecolor='none',
                                    linestyle='-', edgecolor=color, linewidth=2)
                    ax.add_patch(circle)
        
        ax.axis("off")

        if predicted_v_grade:
            title = f"{name} | Angle: {angle} | Actual: V{v_grade} | Predicted: V{predicted_v_grade}"
        elif name:
            title = f"{name} | Angle: {angle} | V{v_grade} | {letter_grade}"
        else:
            title = f"Angle:{angle} | V{v_grade} | {letter_grade}"
        
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'figs/{name}_V{v_grade}.png', dpi=150)
        
        plt.show()
    
    def plot_correlation(self, df, save_dir='figs', save_fig=False):
        """Full correlation heatmap between all features."""
        key_features = [
            'display_difficulty', 'angle_y', 'num_holds', 'num_hand', 
            'num_foot', 'hand_foot_ratio', 'avg_reach', 'max_reach',
            'route_area', 'hold_density', 'ascensionist_count', 
            'quality_average', 'popularity_score', 'angle_x_holds', 
            'density_x_angle'
        ]
        key_features = [f for f in key_features if f in df.columns]
        df_clean = df[key_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        corr_matrix = df_clean.corr()
        
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1,
                    cbar_kws={"label": "Correlation"}, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'{save_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def plot_corr_with_difficulty(self, df, save_dir='figs', save_fig=False):
        """Feature correlations with difficulty target variable."""
        key_features = [
            'angle_y', 'num_holds', 'num_hand', 'num_foot',
            'hand_foot_ratio', 'avg_reach', 'max_reach', 
            'route_area', 'hold_density', 'ascensionist_count', 
            'quality_average', 'popularity_score', 'angle_x_holds', 
            'density_x_angle'
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
        if save_fig:
            plt.savefig(f'{save_dir}/difficulty_correlation.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_distribution(self, df, save_dir='figs', save_fig=False):
        """Feature distributions by grade and quality-popularity relationship."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        
        # Box plots
        box_features = ['angle_y', 'num_holds', 'ascensionist_count', 'quality_average', 'avg_reach']
        grade_order = sorted(df['v_grade'].dropna().unique())
        
        for idx, feat in enumerate(box_features):
            if feat in df.columns:
                ax = axes[idx]
                data = df[[feat, 'v_grade']].dropna()
                sns.boxplot(data=data, x='v_grade', y=feat, hue='v_grade', ax=ax, 
                        order=grade_order, palette='viridis', legend=False, showfliers=False)
                ax.set_title(f'{feat} by Grade', fontsize=11, fontweight='bold')
                ax.set_xlabel('V-Grade', fontsize=9)
                ax.set_ylabel(feat, fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(axis='y', alpha=0.3)
        
        # Scatter: quality vs popularity
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
        if save_fig:
            plt.savefig(f'{save_dir}/distributions.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_clustering(self, X: pd.DataFrame, routes_df: pd.DataFrame, 
                    cluster_col: str = 'style_cluster', n_representatives: int = None,
                    plot_boulders: bool = False):
        from sklearn.decomposition import PCA
        """
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
            

if __name__ == "__main__":
    
    os.makedirs('figs', exist_ok=True)
    
    dp = DataPreprocessing()
    dataset = dp.load_climbs()
    df = dataset.to_pandas()
    
    
    viz = Visualization()
    
    print("Generating visualizations...")
    viz.plot_hold_ids()
    viz.plot_difficulty_quality_analysis(df, dp, save_fig=True)
    viz.plot_correlation(df, save_fig=True)
    viz.plot_corr_with_difficulty(df, save_fig=True)
    viz.plot_distribution(df, save_fig=True)
    
    # Sample climbs
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        viz.plot_boulder(
            sample['holds_data'],
            name=sample['name'][:20],
            v_grade=sample['v_grade'],
            save_fig=True
        )
    