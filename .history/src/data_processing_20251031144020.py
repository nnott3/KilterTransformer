"""
Clean data processing for Kilter Board climbs.
Complete implementation with frame parsing, feature engineering, and dataset creation.
"""

import pandas as pd
import numpy as np
import re
import os
import ast
from datasets import Dataset

# Constants
HOLD_ID = list(range(1073, 1396)) + list(range(1447, 1600))

HOLDCOORDINATES = [[1020, 1225], [960, 1225], [900, 1225], [840, 1225], [780, 1225], [720, 1225], [660, 1225], [600, 1225], [540, 1225], [480, 1225], [420, 1225], [360, 1225], [300, 1225], [240, 1225], [180, 1225], [120, 1225], [60, 1225], [60, 1165], [120, 1165], [180, 1165], [240, 1165], [300, 1165], [360, 1165], [420, 1165], [480, 1165], [540, 1165], [600, 1165], [660, 1165], [720, 1165], [780, 1165], [840, 1165], [900, 1165], [960, 1165], [1020, 1165], [60, 1105], [120, 1105], [180, 1105], [240, 1105], [300, 1105], [360, 1105], [420, 1105], [480, 1105], [540, 1105], [600, 1105], [660, 1105], [720, 1105], [780, 1105], [840, 1105], [900, 1105], [960, 1105], [1020, 1105], [60, 1045], [120, 1045], [180, 1045], [240, 1045], [300, 1045], [360, 1045], [420, 1045], [480, 1045], [540, 1045], [600, 1045], [660, 1045], [720, 1045], [780, 1045], [840, 1045], [900, 1045], [960, 1045], [1020, 1045], [60, 985], [120, 985], [180, 985], [240, 985], [300, 985], [360, 985], [420, 985], [480, 985], [540, 985], [600, 985], [660, 985], [720, 985], [780, 985], [840, 985], [900, 985], [960, 985], [1020, 985], [60, 925], [120, 925], [180, 925], [240, 925], [300, 925], [360, 925], [420, 925], [480, 925], [540, 925], [600, 925], [660, 925], [720, 925], [780, 925], [840, 925], [900, 925], [960, 925], [1020, 925], [60, 865], [120, 865], [180, 865], [240, 865], [300, 865], [360, 865], [420, 865], [480, 865], [540, 865], [600, 865], [660, 865], [720, 865], [780, 865], [840, 865], [900, 865], [960, 865], [1020, 865], [60, 805], [120, 805], [180, 805], [240, 805], [300, 805], [360, 805], [420, 805], [480, 805], [540, 805], [600, 805], [660, 805], [720, 805], [780, 805], [840, 805], [900, 805], [960, 805], [1020, 805], [60, 745], [120, 745], [180, 745], [240, 745], [300, 745], [360, 745], [420, 745], [480, 745], [540, 745], [600, 745], [660, 745], [720, 745], [780, 745], [840, 745], [900, 745], [960, 745], [1020, 745], [60, 685], [120, 685], [180, 685], [240, 685], [300, 685], [360, 685], [420, 685], [480, 685], [540, 685], [600, 685], [660, 685], [720, 685], [780, 685], [840, 685], [900, 685], [960, 685], [1020, 685], [60, 625], [120, 625], [180, 625], [240, 625], [300, 625], [360, 625], [420, 625], [480, 625], [540, 625], [600, 625], [660, 625], [720, 625], [780, 625], [840, 625], [900, 625], [960, 625], [1020, 625], [60, 565], [120, 565], [180, 565], [240, 565], [300, 565], [360, 565], [420, 565], [480, 565], [540, 565], [600, 565], [660, 565], [720, 565], [780, 565], [840, 565], [900, 565], [960, 565], [1020, 565], [60, 505], [120, 505], [180, 505], [240, 505], [300, 505], [360, 505], [420, 505], [480, 505], [540, 505], [600, 505], [660, 505], [720, 505], [780, 505], [840, 505], [900, 505], [960, 505], [1020, 505], [60, 445], [120, 445], [180, 445], [240, 445], [300, 445], [360, 445], [420, 445], [480, 445], [540, 445], [600, 445], [660, 445], [720, 445], [780, 445], [840, 445], [900, 445], [960, 445], [1020, 445], [60, 385], [120, 385], [180, 385], [240, 385], [300, 385], [360, 385], [420, 385], [480, 385], [540, 385], [600, 385], [660, 385], [720, 385], [780, 385], [840, 385], [900, 385], [960, 385], [1020, 385], [60, 325], [120, 325], [180, 325], [240, 325], [300, 325], [360, 325], [420, 325], [480, 325], [540, 325], [600, 325], [660, 325], [720, 325], [780, 325], [840, 325], [900, 325], [960, 325], [1020, 325], [60, 265], [120, 265], [180, 265], [240, 265], [300, 265], [360, 265], [420, 265], [480, 265], [540, 265], [600, 265], [660, 265], [720, 265], [780, 265], [840, 265], [900, 265], [960, 265], [1020, 265], [60, 205], [120, 205], [180, 205], [240, 205], [300, 205], [360, 205], [420, 205], [480, 205], [540, 205], [600, 205], [660, 205], [720, 205], [780, 205], [840, 205], [900, 205], [960, 205], [1020, 205], [60, 145], [120, 145], [180, 145], [240, 145], [300, 145], [360, 145], [420, 145], [480, 145], [540, 145], [600, 145], [660, 145], [720, 145], [780, 145], [840, 145], [900, 145], [960, 145], [1020, 145], [1050, 1255], [990, 1255], [930, 1255], [870, 1255], [810, 1255], [750, 1255], [690, 1255], [630, 1255], [570, 1255], [510, 1255], [450, 1255], [390, 1255], [330, 1255], [270, 1255], [210, 1255], [150, 1255], [90, 1255], [30, 1255], [30, 1135], [150, 1135], [270, 1135], [390, 1135], [510, 1135], [630, 1135], [750, 1135], [870, 1135], [990, 1135], [90, 1075], [210, 1075], [330, 1075], [450, 1075], [570, 1075], [690, 1075], [810, 1075], [930, 1075], [1050, 1075], [30, 1015], [150, 1015], [270, 1015], [390, 1015], [510, 1015], [630, 1015], [750, 1015], [870, 1015], [990, 1015], [90, 955], [210, 955], [330, 955], [450, 955], [570, 955], [690, 955], [810, 955], [930, 955], [1050, 955], [30, 895], [150, 895], [270, 895], [390, 895], [510, 895], [630, 895], [750, 895], [870, 895], [990, 895], [90, 835], [210, 835], [330, 835], [450, 835], [570, 835], [690, 835], [810, 835], [930, 835], [1050, 835], [30, 775], [150, 775], [270, 775], [390, 775], [510, 775], [630, 775], [750, 775], [870, 775], [990, 775], [90, 715], [210, 715], [330, 715], [450, 715], [570, 715], [690, 715], [810, 715], [930, 715], [1050, 715], [30, 655], [150, 655], [270, 655], [390, 655], [510, 655], [630, 655], [750, 655], [870, 655], [990, 655], [90, 595], [210, 595], [330, 595], [450, 595], [570, 595], [690, 595], [810, 595], [930, 595], [1050, 595], [30, 535], [150, 535], [270, 535], [390, 535], [510, 535], [630, 535], [750, 535], [870, 535], [990, 535], [90, 475], [210, 475], [330, 475], [450, 475], [570, 475], [690, 475], [810, 475], [930, 475], [1050, 475], [30, 415], [150, 415], [270, 415], [390, 415], [510, 415], [630, 415], [750, 415], [870, 415], [990, 415], [90, 355], [210, 355], [330, 355], [450, 355], [570, 355], [690, 355], [810, 355], [930, 355], [1050, 355], [30, 295], [150, 295], [270, 295], [390, 295], [510, 295], [630, 295], [750, 295], [870, 295], [990, 295]]
HOLDCOORDINATES = [[a, b-112] for a, b in HOLDCOORDINATES]

HOLD_COLORS = {
    'start': "#00DD00",  # Start - Green
    'hand': "#00FFFF",  # Hand - Cyan
    'finish': "#FF00FF",  # Finish - Magenta
    'feet': "#FFA500"   # Foot - Orange
}

ROLE_MAP = {12: "start", 13: "hand", 14: "finish", 15: "feet"}



class DataPreprocessing:
    def __init__(self, csv_dir="csv_exports"):
        self.csv_dir = csv_dir
        self._grade_dict = None

    @property
    def grade_dict(self):
        if self._grade_dict is None:
            try:
                df = pd.read_csv(f"{self.csv_dir}/difficulty_grades.csv")
                self._grade_dict = dict(zip(df["difficulty"], df["boulder_name"]))
            except:
                self._grade_dict = {1: '1a/V0', 2: '1b/V0', 3: '1c/V0', 
                                    4: '2a/V0',5: '2b/V0', 6: '2c/V0', 
                                    7: '3a/V0', 8: '3b/V0', 9: '3c/V0', 
                                    10: '4a/V0', 11: '4b/V0', 12: '4c/V0',
                                    13: '5a/V1', 14: '5b/V1', 15: '5c/V2',
                                    16: '6a/V3', 17: '6a+/V3',18: '6b/V4', 19: '6b+/V4', 20: '6c/V5', 21: '6c+/V5',
                                    22: '7a/V6', 23: '7a+/V7', 24: '7b/V8', 25: '7b+/V8', 26: '7c/V9', 27: '7c+/V10',
                                    28: '8a/V11', 29: '8a+/V12', 30: '8b/V13', 31: '8b+/V14', 32: '8c/V15', 33: '8c+/V16',
                                    34: '9a/V17', 35: '9a+/V18', 36: '9b/V19', 37: '9b+/V20', 38: '9c/V21', 39: '9c+/V22'
                                    }
        return self._grade_dict

    def difficulty_to_v_grade(self, diff_id):
        if pd.isna(diff_id):
            return None
        return int(self.grade_dict.get(int(diff_id), "/?").split('/')[1].split('V')[1])

    def difficulty_to_letter_grade(self, diff_id):
        if pd.isna(diff_id):
            return None
        return self.grade_dict.get(int(diff_id), "/?").split('/')[0]

    def parse_frames(seld, frames, **kwargs): #angle+grade is optional,
        # Transform frames: p1595r12p1596r15 -> start1595_feet1596
        parts = []
        
        for hold_id, func in re.findall(r"p(\d+)r(\d+)", frames):
            role = ROLE_MAP.get(int(func), "")
            parts.append(f"{role}{hold_id}")
        
        frames_str = "_".join(parts)
        return_str = ""
        if 'angle' in kwargs:
            return_str += f"angle{int(kwargs['angle'])}_"
        if 'grade' in kwargs:
            return_str += f"grade{round(kwargs['grade'])}_"
        return_str += frames_str
        return return_str
        
    def add_holds_data(self, df):
        """Parse frames and add hold-related columns."""
        pattern = r"p(\d+)r(\d+)"
        matches = df["frames"].astype(str).apply(lambda s: re.findall(pattern, s))


        df["holds_data"] = matches.apply(lambda m: [{int(f): ROLE_MAP.get(int(h))} for f, h in m])
        df["num_holds"] = df["holds_data"].str.len()

        funcs = matches.apply(lambda m: [int(f) for _, f in m])
        df["num_start"] = funcs.apply(lambda x: x.count(12))
        df["num_finish"] = funcs.apply(lambda x: x.count(14))
        df["num_footonly"] = funcs.apply(lambda x: x.count(15))
        df["num_hand"] = funcs.apply(lambda x: x.count(13))

        df["v_grade"] = df["display_difficulty"].apply(self.difficulty_to_v_grade)
        df["letter_grade"] = df["display_difficulty"].apply(self.difficulty_to_letter_grade)
  
        df["frames"] = df.apply(lambda x: self.parse_frames(x["frames"], angle=x["angle_y"], grade=x["display_difficulty"]), axis=1)

        return df

    def add_engineered_features(self, df):
        """Add derived features for analysis."""
        # Parse holds_data if string
        if isinstance(df['holds_data'].iloc[0], str):
            df['holds_data'] = df['holds_data'].apply(ast.literal_eval)
        
        # Ratios
        df['hand_foot_ratio'] = df['num_hand'] / (df['num_footonly'] + 1)
        
        # Spatial features
        def compute_reach_stats(holds):
            if len(holds) < 2:
                return 0, 0
            reaches = []
            for i in range(len(holds) - 1):
                h1_id = int(list(holds[i].keys())[0])
                h2_id = int(list(holds[i+1].keys())[0])
                if h1_id in HOLD_ID and h2_id in HOLD_ID:
                    idx1, idx2 = HOLD_ID.index(h1_id), HOLD_ID.index(h2_id)
                    x1, y1 = HOLDCOORDINATES[idx1]
                    x2, y2 = HOLDCOORDINATES[idx2]
                    reaches.append(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            return (np.mean(reaches) if reaches else 0, 
                    max(reaches) if reaches else 0)
        
        reach_stats = df['holds_data'].apply(compute_reach_stats)
        df['avg_reach'] = reach_stats.apply(lambda x: x[0])
        df['max_reach'] = reach_stats.apply(lambda x: x[1])
        
        # Route area (bounding box)
        def compute_area(holds):
            if len(holds) < 3:
                return 0
            coords = []
            for h in holds:
                h_id = int(list(h.keys())[0])
                if h_id in HOLD_ID:
                    idx = HOLD_ID.index(h_id)
                    coords.append(HOLDCOORDINATES[idx])
            
            if len(coords) < 3:
                return 0
            
            coords = np.array(coords)
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            return x_range * y_range
        
        df['route_area'] = df['holds_data'].apply(compute_area)
        df['hold_density'] = df['num_holds'] / (df['route_area'] + 1)
        
        # Interaction features
        df['popularity_score'] = df['ascensionist_count'] * df['quality_average']
        df['angle_x_holds'] = df['angle_y'] * df['num_holds']
        df['density_x_angle'] = df['hold_density'] * df['angle_y']
        
        return df

    def clean_routes(self, df):
        """Filter to valid climbs."""
        mask = (
            (df['ascensionist_count'] > 2) &
            df['display_difficulty'].between(12, 28) &
            (df['is_listed'] == 1) &
            (df['layout_id'] == 1) &
            (~df['angle_y'].isna()) &
            (df['angle_y'].between(20, 60)) &
            (~df['display_difficulty'].isna()) &
            (df['num_holds'] < 20) &
            (df['num_holds'] > 0) &
            df['num_start'].between(1, 2) &
            df['num_finish'].between(1, 2) &
            (df['quality_average'] > 2)
        )
        
        df_filtered = df[mask].copy()
        
        # Check valid hold IDs
        df_filtered = df_filtered[
            df_filtered['holds_data'].apply(
                lambda h: all(int(list(x.keys())[0]) in HOLD_ID for x in h) if len(h) > 0 else False
            )
        ]
        
        return df_filtered

    def load_climbs(self, cache_path="data/climbs_cleaned.csv"):
        """Load climbs as HuggingFace Dataset."""
        
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            print(f"Loaded {len(df)} routes from cache {cache_path}")
            # Parse holds_data and add engineered features
            # df['holds_data'] = df['holds_data'].apply(ast.literal_eval)
            # df = self.add_engineered_features(df)
            
            # Select columns for dataset
            selected_cols = ['name', 'frames', 'holds_data', 'display_difficulty', 
                           'angle_y', 'quality_average', 'ascensionist_count', 
                           'v_grade', 'letter_grade', 'num_holds', 'num_hand',
                           'num_footonly', 'hand_foot_ratio', 'avg_reach', 
                           'max_reach', 'route_area', 'hold_density', 
                           'popularity_score', 'angle_x_holds', 'density_x_angle']
            selected_cols = [c for c in selected_cols if c in df.columns]
            
            dataset = Dataset.from_pandas(df[selected_cols])
            # return dataset.train_test_split(test_size=0.15, seed=42)
            return dataset

        # First-time processing
        print("Processing routes from CSV exports...")
        routes = pd.read_csv(f"{self.csv_dir}/climbs.csv")
        stats = pd.read_csv(f"{self.csv_dir}/climb_stats.csv")

        # Merge
        routes = routes.merge(
            stats[['climb_uuid', 'angle', 'display_difficulty',
                   'ascensionist_count', 'difficulty_average', 'quality_average']],
            left_on='uuid', right_on='climb_uuid', how='left'
        ).drop('climb_uuid', axis=1)

        # Clean numeric columns
        for col in ['angle_y', 'ascensionist_count']:
            routes[col] = routes[col].fillna(0).replace([np.inf, -np.inf], 0).astype(int)

        print(f"Initial: {len(routes)} routes")
        
        # Basic filter
        routes = routes[
            (routes['layout_id'] == 1) & 
            (routes['ascensionist_count'] > 2) &
            (routes['is_listed'] == 1)
        ]
        print(f"After basic filter: {len(routes)} routes")
        
        # Parse frames and add features
        routes = self.add_holds_data(routes)
        routes = self.add_engineered_features(routes)
        routes = self.clean_routes(routes)
        
        print(f"After cleaning: {len(routes)} routes")

        # Save (convert holds_data to string for CSV)
        # routes['holds_data'] = routes['holds_data'].apply(str)
        
        os.makedirs('data', exist_ok=True)
        routes.to_csv(cache_path, index=False)
        print(f"Saved to {cache_path}")

        # Create dataset
        selected_cols = ['name', 'frames', 'display_difficulty', 
                       'angle_y', 'quality_average', 'ascensionist_count', 
                       'v_grade', 'letter_grade', 'num_holds', 'num_hand',
                       'num_footonly', 'hand_foot_ratio', 'avg_reach', 
                       'max_reach', 'route_area', 'hold_density', 
                       'popularity_score', 'angle_x_holds', 'density_x_angle']
        
        dataset = Dataset.from_pandas(routes[selected_cols])
        # return dataset.train_test_split(test_size=0.15, seed=42)
        return dataset

    def tokenize_dataset(self, dataset, tokenizer):
        """Tokenize frames column."""
        return dataset.map(lambda example: tokenizer(example["frames"]), batched=True)

    def preprocess_datasets(self, datasets, tokenizer):
        """Tokenize and remove original columns."""
        for name in ("train", "test"):
            col_names = datasets[name].column_names
            datasets[name] = self.tokenize_dataset(datasets[name], tokenizer).remove_columns(col_names)
        return datasets
    
    # add more utils
    def holds_data_to_str(self, holds_data):
        """Convert holds_data to string.
        [{'1078': 15}, {'1153': 12}, {'1184': 13}] to 
        angle40_grade18_start1111_hand2222_feet3333_finish4444
        """
        role_map = {12: "start", 13: "hand", 14: "finish", 15: "feet"}
        hold_str = []
        for hold_dict in holds_data:
            for hold_id, func in hold_dict.items():
                role = role_map.get(int(func), "")
                hold_str.append(f"{role}{hold_id}")
        return "_".join(hold_str)
    
    def str_to_holds_data(self, holds_str):
        # excluding angle and grade, process it elsewhere
        pattern = r"(start|hand|finish|feet)(\d+)"
        holds_data = []
        for role, hold_id in re.findall(pattern, holds_str):
            holds_data.append({hold_id: role})
        return holds_data


if __name__ == "__main__":
    dp = DataPreprocessing()
    datasets = dp.load_climbs()
    
    print(f"\nTrain: {len(datasets['train'])}, Test: {len(datasets['test'])}")
    
    sample = datasets['train'][0]
    print(f"\nSample columns: {list(sample.keys())}")
    print(f"\nSample data:")
    for k in ['name', 'v_grade', 'angle_y', 'num_holds', 'frames']:
        if k in sample:
            print(f"  {k}: {sample[k]}")