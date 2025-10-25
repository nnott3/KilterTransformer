"""
Model evaluation utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from .data_processing import DataPreprocessing


class Evaluation:
    """Model evaluation and visualization."""
    
    def __init__(self, data_processor: DataPreprocessing = None):
        self.dp = data_processor or DataPreprocessing()
    
    def get_scores(self, y_true, y_pred, model_name: str = "Model") -> dict:
        """Calculate and print regression metrics."""
        # Convert difficulty IDs to V-grade numbers
        y_true_v = np.array([
            int(self.dp.difficulty_to_grades(d)[1].replace('V', '')) 
            for d in np.round(y_true).astype(int)
        ])
        y_pred_v = np.array([
            int(self.dp.difficulty_to_grades(d)[1].replace('V', '')) 
            for d in np.round(y_pred).astype(int)
        ])
        
        # Calculate V-grade difference
        within_grade = 1
        grade_diff = np.abs(y_true_v - y_pred_v)
        within_tolerance = np.mean(grade_diff <= within_grade) * 100
       
        scores = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'within_±{within_grade}_grade': within_tolerance
        }
        print(f"\n{model_name} Performance:")
        print(f"  R²:   {scores['r2']:.4f}")
        print(f"  MAE:  {scores['mae']:.4f}")
        print(f"  RMSE: {scores['rmse']:.4f}")
        print(f"  Within ±{within_grade} V-grade: {scores[f'within_±{within_grade}_grade']:.2f}%")
        return scores
    
    def plot_predictions(self, y_true, y_pred):
        """Plot actual vs predicted with perfect prediction line."""
        plt.figure(figsize=(8, 6))
        sns.histplot(x=y_true, y=y_pred, bins=30, pthresh=0.1, cmap="mako", cbar=True)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        
        # Get unique grades and convert to labels
        unique_grades = sorted(set(np.round(y_true).astype(int)))
        
        # Group grades by their letter grade to avoid duplicates
        grade_label_map = {}
        for g in unique_grades:
            label = self.dp.difficulty_to_grades(g)[0] or f"D{g}"
            if label not in grade_label_map.values():
                grade_label_map[g] = label
        
        # Only show ticks for grades with unique labels
        tick_positions = sorted(grade_label_map.keys())
        tick_labels = [grade_label_map[pos] for pos in tick_positions]
        
        # Set tick positions and labels
        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
        plt.yticks(tick_positions, tick_labels)
        
        plt.xlabel("Actual Grade")
        plt.ylabel("Predicted Grade")
        plt.title("Actual vs Predicted Difficulty")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true, y_pred):
        """Plot residual distribution."""
        residuals = y_pred - y_true
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Difficulty')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residual Plot')
        
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.randint(12, 30, 100)
    y_pred = y_true + np.random.normal(0, 2, 100)
    
    evaluator = Evaluation()
    scores = evaluator.get_scores(y_true, y_pred, "Example Model")
    evaluator.plot_predictions(y_true, y_pred)
    evaluator.plot_residuals(y_true, y_pred)
