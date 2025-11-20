"""
Utility functions for KilterGPT
"""
import torch
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Optional, List, Dict, Tuple
import time
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

from src.data_processing import HOLD_ID, HOLDCOORDINATES, HOLD_COLORS

hold_cycle = ["start", "hand", "feet", "finish"]

def grade_to_display_text(grade_value: int, dp) -> str:
    """Convert grade value to display text"""
    letter_grade = dp.difficulty_to_letter_grade(grade_value)
    v_grade = dp.difficulty_to_v_grade(grade_value)
    return f"{letter_grade} / V{v_grade}"

def find_nearest_hold(x: int, y: int, threshold: int = 40) -> Optional[int]:
    """Find nearest hold to clicked position"""
    min_dist = float('inf')
    nearest_hold = None
    for hold_id in HOLD_ID:
        idx = HOLD_ID.index(hold_id)
        hold_x, hold_y = HOLDCOORDINATES[idx]
        dist = np.sqrt((x - hold_x)**2 + (y - hold_y)**2)
        if dist < threshold and dist < min_dist:
            min_dist = dist
            nearest_hold = hold_id
    return nearest_hold

def draw_board_with_holds(board_img: Image.Image,
                          selected_holds_list: List[Tuple[int, str]], 
                          holds_data: List[Dict] = None,
                          candidate_holds: List[int] = None,
                          title: str = "",
                          font_path: str = None,
                          show_ids=False) -> Image.Image:
    """Draw board with holds, candidates, or generated route"""
    img = board_img.copy()
    draw = ImageDraw.Draw(img)
    radius, width = 30, 4
    
    # Draw candidate holds (grey circles during animation)
    if candidate_holds:
        for hold_id in candidate_holds:
            idx = HOLD_ID.index(hold_id)
            x, y = HOLDCOORDINATES[idx]
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                        outline='#DDDDDD', width=width)

    # Draw generated route
    if holds_data:
        for hold in holds_data:
            for hold_id_str, role in hold.items():
                hold_id = int(hold_id_str)
                idx = HOLD_ID.index(hold_id)
                x, y = HOLDCOORDINATES[idx]
                color = HOLD_COLORS.get(role, "#FFFFFF")
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                           outline=color, width=width)
                if show_ids and font_path:
                    font = ImageFont.truetype(font_path, 25)
                    draw.rectangle([(x - radius, y - 5), (x + radius, y + radius)], fill=(255, 255, 255, 170))
                    draw.text((x-30, y-15), str(hold_id), fill="#FFF", font=font, align="left")
                    
    else:
        # Draw selected holds (input)
        for hold_id, role in selected_holds_list:
            idx = HOLD_ID.index(hold_id)
            x, y = HOLDCOORDINATES[idx]
            color = HOLD_COLORS.get(role, "#FFFFFF")
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                       outline=color, width=width)
            if show_ids and font_path:
                font = ImageFont.truetype(font_path, 25)
                draw.text((x-30, y-15), str(hold_id), fill="#FFF", font=font, align="left")
    
    # Add title if provided
    if title and font_path:
        font = ImageFont.truetype(font_path, 28)
        bbox = draw.textbbox((0, 0), title, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = img.width - w - 20, 10
        draw.text((x + 8, y + 8), title, fill="black", font=font)
    
    return img

def calculate_similarity_perfect_match(route1_dict: Dict, route2_dict: Dict) -> Tuple[int, int]:
    """Calculate perfect match and hold-only match"""
    route1_holds = set(route1_dict.keys())
    route2_holds = set(route2_dict.keys())
    
    hold_matches = len(route1_holds & route2_holds)
    perfect_matches = sum(1 for hold_id in route1_holds & route2_holds 
                         if route1_dict[hold_id] == route2_dict[hold_id])
    
    return perfect_matches, hold_matches

def search_similar_boulders(generated_holds: List[Dict], angle: int, grade: int,
                           database_df, dp,
                           sort_by: str = "similarity", 
                           match_type: str = "perfect (correct holds + roles)"):
    """Search database for similar boulders"""
    if database_df.empty:
        return database_df, []
    
    gen_dict = {int(list(h.keys())[0]): list(h.values())[0] for h in generated_holds if h}
    
    results = []
    for _, row in database_df.iterrows():
        perfect_matches, hold_matches = calculate_similarity_perfect_match(gen_dict, row['hold_set'])
        results.append({
            'Name': row.get('name', 'Unknown'),
            'Setter': row.get('setter_username', 'Anonymous')[:15],
            'Angle': f"{int(row.get('angle_y', 0))}°",
            'Grade': grade_to_display_text(int(row.get('display_difficulty', 18)), dp),
            'Perfect': perfect_matches,
            'Holds': hold_matches,
            'perfect_matches': perfect_matches,
            'hold_matches': hold_matches,
            'holds_data': row.get('holds_data', []),
            'angle_y': row.get('angle_y', 40),
            'display_difficulty': row.get('display_difficulty', 18)
        })
    
    results = sorted(results, key=lambda x: (x['perfect_matches'], x['hold_matches']), reverse=True)
    
    import pandas as pd
    top_df = pd.DataFrame(results)
    top_df = top_df[top_df['Angle'].str.replace('°','').astype(int) == angle].head(10)
    return top_df[['Name', 'Setter', 'Angle', 'Grade', 'Perfect', 'Holds']], results

def generate_route_animated(angle: int, grade: int, num_holds: int,
                           temperature: float, top_p: float,
                           selected_holds_state: List,
                           use_constraints: bool,
                           animate: bool,
                           model_name: str,
                           load_model_fn,
                           dp,
                           board_img,
                           font_path: str,
                           stats: dict,
                           routes_db: list,
                           save_stats_fn):
    """Generate route with animation"""
    gen_model, gen_tokenizer = load_model_fn(model_name)
    if gen_model is None:
        yield None, f"Total: {stats['total']}", f"👍 {stats['likes']} | 👎 {stats['dislikes']}"
        return
    
    try:
        prompt_parts = [f"angle{angle}", f"grade{grade}"]
        for hold_id, role in selected_holds_state:
            prompt_parts.append(f"{role}{hold_id}")
        prompt = "_".join(prompt_parts)
        
        generated_holds = []
        animation_frames = []
        selected_hold_ids = set(hold_id for hold_id, _ in selected_holds_state)
        
        def animation_callback(top_tokens, top_tokens_ids):
            nonlocal animation_frames
            candidate_holds, candidate_tokens = [], []
            
            for tok, tok_id in zip(top_tokens, top_tokens_ids.squeeze().tolist()):
                match = re.search(r'(start|hand|feet|finish)(\d+)', tok[0])
                if match:
                    hold_id = int(match.group(2))
                    if hold_id not in selected_hold_ids:
                        candidate_holds.append(hold_id)
                        candidate_tokens.append(tok_id)
            
            animation_frames.append({
                'candidates_hold_id': candidate_holds[:8],
                'candidates_token_id': candidate_tokens[:8],
            })
        
        if use_constraints:
            generated_text = gen_model.generate_with_constraint(
                prompt=prompt, max_length=num_holds * 2 + 10,
                temperature=temperature, top_k=8, repetition_penalty=1.2,
                do_sample=True, min_holds=max(4, num_holds - 2),
                max_holds=num_holds + 2, animate=animate,
                animation_callback=animation_callback if animate else None
            )
        else:
            generated_text = gen_model.generate(
                prompt=prompt, max_length=num_holds * 2 + 10,
                temperature=temperature, top_k=8, repetition_penalty=1.2,
                do_sample=True, animate=animate,
                animation_callback=animation_callback if animate else None
            )
        
        generated_holds = dp.str_to_holds_data(generated_text)
        if len(generated_holds) > num_holds:
            generated_holds = generated_holds[:num_holds]
        
        # Animation
        if animate and animation_frames:
            for i, frame in enumerate(animation_frames):
                if i >= len(selected_holds_state):
                    partial_holds = generated_holds[:i]
                    img = draw_board_with_holds(board_img, [], holds_data=partial_holds,
                                              candidate_holds=animation_frames[i-len(selected_holds_state)]['candidates_hold_id'],
                                              font_path=font_path)
                    yield img, f"Total: {stats['total']}", f"👍 {stats['likes']} | 👎 {stats['dislikes']}"
                    time.sleep(0.5)
                    
                    if i < len(generated_holds):
                        partial_holds = generated_holds[:i+1]
                        img = draw_board_with_holds(board_img, [], holds_data=partial_holds, font_path=font_path)
                        yield img, f"Total: {stats['total']}", f"👍 {stats['likes']} | 👎 {stats['dislikes']}"
                        time.sleep(0.5)
        
        # Final route
        title = f"{angle}° | {grade_to_display_text(grade, dp)}"
        img = draw_board_with_holds(board_img, [], holds_data=generated_holds, title=title, font_path=font_path)
        
        routes_db.append({
            'holds_data': generated_holds, 'angle': angle, 'grade': grade,
            'timestamp': time.time(), 'liked': None, 'model': model_name
        })
        stats['total'] += 1
        save_stats_fn(stats, routes_db)
        
        yield img, f"Total: {stats['total']}", f"👍 {stats['likes']} | 👎 {stats['dislikes']}"
        
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        img = draw_board_with_holds(board_img, selected_holds_state, font_path=font_path)
        yield img, f"Total: {stats['total']}", f"👍 {stats['likes']} | 👎 {stats['dislikes']}"

def create_hold_frequency_heatmap(database_df):
    """Hold frequency visualization"""
    if database_df.empty:
        return go.Figure()
    
    hold_freq = defaultdict(int)
    for _, row in database_df.iterrows():
        for hold in row.get('holds_data', []):
            if hold:
                hold_freq[list(hold.keys())[0]] += 1
    
    fig = go.Figure(data=go.Scatter(
        x=list(hold_freq.keys())[:50],
        y=list(hold_freq.values())[:50],
        mode='markers',
        marker=dict(size=10, color=list(hold_freq.values())[:50], colorscale='Viridis')
    ))
    fig.update_layout(title="Hold Frequency (Top 50)", xaxis_title="Hold ID", yaxis_title="Frequency")
    return fig

def create_grade_angle_distribution(database_df):
    """Grade vs Angle distribution"""
    if database_df.empty:
        return go.Figure()
    
    fig = px.scatter(database_df.head(500), x='angle_y', y='display_difficulty',
                     color='quality_average', title="Grade vs Angle Distribution",
                     labels={'angle_y': 'Angle', 'display_difficulty': 'Grade'})
    return fig

def create_generation_stats(stats):
    """Stats from generated routes"""
    likes = stats['likes']
    dislikes = stats['dislikes']
    
    fig = go.Figure(data=[
        go.Bar(x=['Likes', 'Dislikes'], y=[likes, dislikes],
               marker_color=['green', 'red'])
    ])
    fig.update_layout(title="User Feedback Stats", yaxis_title="Count")
    return fig