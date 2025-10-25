"""
Feature engineering for climbing route difficulty prediction.
"""

import re
import pandas as pd
import numpy as np
import ast
from typing import List, Tuple
from utils.data_processing import HOLD_ID, HOLDCOORDINATES


class FeatureEng:
    """Feature engineering class for climbing route data."""
    
    def __init__(self):
        self.hold_id = HOLD_ID
        self.hold_coords = HOLDCOORDINATES
        
    def load_data(self, filepath: str = 'src/cleaned_routes.csv') -> pd.DataFrame:
        """
        Load cleaned routes data.
        
        Args:
            filepath: Path to cleaned routes CSV
            
        Returns:
            DataFrame with routes
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Parse holds_data from string to list of dicts
        if 'holds_data' in df.columns:
            df['holds_data'] = df['holds_data'].apply(self._parse_holds_data)
        
        print(f"Loaded {len(df)} routes")
        return df
    
    def _parse_holds_data(self, holds_str: str) -> List[dict]:
        """
        Parse holds_data string to list of dictionaries.
        
        Args:
            holds_str: String representation of holds data
            
        Returns:
            List of hold dictionaries
        """
        try:
            if pd.isna(holds_str) or holds_str == '':
                return []
            # Use ast.literal_eval to safely parse string
            holds = ast.literal_eval(holds_str)
            return holds if isinstance(holds, list) else []
        except (ValueError, SyntaxError):
            return []
    
    def add_hold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add hold-related features.
        
        Features:
        - num_holds: Total number of holds
        - num_hand_holds: Number of hand holds
        - num_foot_holds: Number of foot holds
        - hand_foot_ratio: Ratio of hand to foot holds
        - has_start: Whether route has start hold (14)
        - has_finish: Whether route has finish hold (15)
        """
        print("Engineering hold features...")
        
        df['num_holds'] = df['holds_data'].apply(len)
        
        df['num_hand_holds'] = df['holds_data'].apply(
            lambda x: sum(1 for h in x if list(h.values())[0] != 13)
        )
        
        df['num_foot_holds'] = df['holds_data'].apply(
            lambda x: sum(1 for h in x if list(h.values())[0] == 13)
        )
        
        df['hand_foot_ratio'] = df.apply(
            lambda row: row['num_hand_holds'] / row['num_foot_holds'] 
            if row['num_foot_holds'] > 0 else row['num_hand_holds'],
            axis=1
        )
        
        df['has_start'] = df['holds_data'].apply(
            lambda x: any(list(h.values())[0] == 14 for h in x)
        )
        
        df['has_finish'] = df['holds_data'].apply(
            lambda x: any(list(h.values())[0] == 15 for h in x)
        )
        
        return df
    
    def add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add spatial/geometric features based on hold positions.
        
        Features:
        - route_width: Horizontal span (max_x - min_x)
        - route_height: Vertical span (max_y - min_y)
        - route_area: Width * Height
        - hold_density: Number of holds / area
        - avg_reach: Average distance between consecutive holds
        - max_reach: Maximum distance between consecutive holds
        - total_distance: Sum of all consecutive hold distances
        """
        print("Engineering spatial features...")
        
        def compute_spatial_features(holds_data):
            """Compute spatial features for a single route."""
            if not holds_data or len(holds_data) == 0:
                return {
                    'route_width': 0, 'route_height': 0, 'route_area': 0,
                    'hold_density': 0, 'avg_reach': 0, 'max_reach': 0, 
                    'total_distance': 0
                }
            
            # Get coordinates for all holds
            coords = []
            for hold in holds_data:
                hold_id = list(hold.keys())[0]
                try:
                    idx = self.hold_id.index(hold_id)
                    x, y = self.hold_coords[idx]
                    coords.append((x, y))
                except (ValueError, IndexError):
                    continue
            
            if not coords:
                return {
                    'route_width': 0, 'route_height': 0, 'route_area': 0,
                    'hold_density': 0, 'avg_reach': 0, 'max_reach': 0,
                    'total_distance': 0
                }
            
            # Calculate spatial dimensions
            xs, ys = zip(*coords)
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            area = max(width * height, 1)  # Avoid division by zero
            
            # Calculate reaches between consecutive holds
            reaches = []
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                reach = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if not np.isnan(reach) and not np.isinf(reach):
                    reaches.append(reach)
            
            return {
                'route_width': width,
                'route_height': height,
                'route_area': area,
                'hold_density': len(holds_data) / area,
                'avg_reach': np.mean(reaches) if reaches else 0,
                'max_reach': max(reaches) if reaches else 0,
                'total_distance': sum(reaches) if reaches else 0
            }
        
        # Apply to all routes
        spatial = df['holds_data'].apply(compute_spatial_features)
        spatial_df = pd.DataFrame(spatial.tolist())
        
        # Add to main dataframe
        for col in spatial_df.columns:
            df[col] = spatial_df[col]
        
        return df
    
    def add_difficulty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add difficulty-related derived features.
        
        Features:
        - v_grade_numeric: Numeric version of v_grade (V3->3, V10->10)
        - is_plus_grade: Whether grade has '+' (e.g., V5+)
        - difficulty_normalized: Min-max normalized difficulty [0,1]
        - popularity_score: Log-scaled ascensionist count
        """
        print("Engineering difficulty features...")
        
        # Extract numeric grade
        def extract_numeric_grade(grade_str):
            """Extract numeric value from V-grade string."""
            
            match = re.search(r'\d+', str(grade_str))
            return int(match.group()) if match else 0
        
        df['v_grade_numeric'] = df['v_grade'].apply(extract_numeric_grade)
        
        # Check for plus grades
        df['is_plus_grade'] = df['v_grade'].astype(str).str.contains(r'\+', regex=True)
        
        # Normalize difficulty
        if 'display_difficulty' in df.columns:
            min_diff = df['display_difficulty'].min()
            max_diff = df['display_difficulty'].max()
            df['difficulty_normalized'] = (df['display_difficulty'] - min_diff) / (max_diff - min_diff)
        
        # Popularity score (log scale to handle outliers)
        df['popularity_score'] = np.log1p(df['ascensionist_count'])
        
        return df
    
    def add_angle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wall angle categorical features.
        
        Features:
        - angle_sin: sin(angle) for cyclic encoding
        - angle_cos: cos(angle) for cyclic encoding
        """
        print("Engineering angle features...")
        
        if 'angle_y' not in df.columns:
            print("Warning: angle_y column not found")
            return df
           
        df['angle_sin'] = np.sin(np.radians(df['angle_y']))
        df['angle_cos'] = np.cos(np.radians(df['angle_y']))
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features:
        - angle_x_holds: Angle * number of holds
        - angle_x_reach: Angle * average reach
        - density_x_angle: Hold density * angle
        """
        print("Engineering interaction features...")
        
        df['angle_x_holds'] = df['angle_y'] * df['num_holds']
        df['angle_x_reach'] = df['angle_y'] * df['avg_reach']
        df['density_x_angle'] = df['hold_density'] * df['angle_y']
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature engineering steps.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        df = self.add_hold_features(df)
        df = self.add_spatial_features(df)
        df = self.add_difficulty_features(df)
        df = self.add_angle_features(df)
        df = self.add_interaction_features(df)
        
        print("\n" + "="*70)
        print(f" Feature engineering complete!")
        print(f"Total columns: {len(df.columns)}")
        print("="*70)
        
        return df
    
    def save_features(self, df: pd.DataFrame, output_path: str = 'src/route_features.csv'):
        """
        Save dataframe with engineered features.
        
        Args:
            df: DataFrame with features
            output_path: Path to save CSV
        """
        
        df_save = df.copy()
        df_save['holds_data'] = df_save['holds_data'].apply(str)
        
        df_save.to_csv(output_path, index=False)
        print(f"Shape: {df_save.shape}")


def main():
    """Example usage of FeatureEng class."""
    
    # Initialize feature engineer
    fe = FeatureEng()
    
    # Load data
    routes = fe.load_data('src/cleaned_routes.csv')
    
    # Engineer all features
    routes_with_features = fe.engineer_all_features(routes)
    
    # Display sample
    print("\nSample of new features:")
    print(routes_with_features[['v_grade', 'num_holds', 'avg_reach', 'route_area', 
                                 'hand_foot_ratio']].head(3))
    
    # Save
    fe.save_features(routes_with_features, 'src/route_features.csv')
    
    # Feature summary
    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    print(f"{'Feature':<25} {'Mean':<12} {'Std':<12} {'Min':<10} {'Max':<10}")
    print("-"*70)
    
    numeric_features = routes_with_features.select_dtypes(include=[np.number]).columns
    for feat in sorted(numeric_features):
        if feat not in ['uuid', 'layout_id', 'setter_id']:  # Skip IDs
            mean = routes_with_features[feat].mean()
            std = routes_with_features[feat].std()
            min_val = routes_with_features[feat].min()
            max_val = routes_with_features[feat].max()
            print(f"{feat:<25} {mean:<12.2f} {std:<12.2f} {min_val:<10.2f} {max_val:<10.2f}")


if __name__ == "__main__":
    main()