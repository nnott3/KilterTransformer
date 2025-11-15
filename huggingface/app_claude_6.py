"""
KilterGPT Gradio Web Interface - Enhanced Version with Fixes
"""
import gradio as gr
import torch
import re
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
import functools
import pandas as pd
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import ast

# Setup paths
project_root = Path(__file__).resolve().parent.parent 
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from src.data_processing import DataPreprocessing, HOLD_ID, HOLDCOORDINATES, HOLD_COLORS
from src.temp_gpt import KilterGPT

# Configuration
MODEL_CONFIGS = {
    "KilterGPT Base": "nottreepat/testtest",
    "KilterGPT (with data-augment)": "nottreepat/testtest",  # Mock with same model
    "KilterGPT (with order-invariant loss)": "nottreepat/testtest"  # Mock with same model
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOARD_IMG_URL = "https://huggingface.co/datasets/nottreepat/climbs_cleaned/resolve/main/full_board_commercial.png"
DATABASE_URL = "https://huggingface.co/datasets/nottreepat/climbs_cleaned/resolve/main/climbs_cleaned.csv"
STATS_FILE = "huggingface/generation_stats.json"
GENERATED_ROUTES_FILE = "huggingface/generated_routes.json"
FONT_PATH = "huggingface/fonts/Arial Bold.ttf"

hold_cycle = ["start", "hand", "feet", "finish"]
dp = DataPreprocessing()

PRESET_EXAMPLES = {
    "Beginner": {"angle": 30, "grade": 15, "selected_holds": [(1133, "start"), (1184, "start")]},
    "Intermediate": {"angle": 40, "grade": 25, "selected_holds": [(1100, "start"), (1150, "start")]},
}

def load_stats():
    stats, generated_routes = {'total': 0, 'likes': 0, 'dislikes': 0}, []
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
        except: pass
    if os.path.exists(GENERATED_ROUTES_FILE):
        try:
            with open(GENERATED_ROUTES_FILE, 'r') as f:
                generated_routes = json.load(f)
        except: pass
    return stats, generated_routes

def save_stats(stats, generared_routes_db):
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
        with open(GENERATED_ROUTES_FILE, 'w') as f:
            json.dump(generared_routes_db, f)
    except Exception as e:
        print(f"Failed to save stats: {e}")

STATS, GENERATED_ROUTES_DB = load_stats()

# Load database with preprocessing for faster search
@functools.lru_cache(maxsize=1)
def load_database():
    try:
        df = pd.read_csv(DATABASE_URL)
        df = df[(df['quality_average'] >= 2.5) & (df['angle_y'] >= 20) & (df['angle_y'] <= 60)]
        
        if 'frames' in df.columns:
            df['holds_data'] = df['frames'].apply(
                lambda x: dp.str_to_holds_data(x) if isinstance(x, str) else []
            )
        
        # Precompute hold sets for faster similarity search
        df['hold_set'] = df['holds_data'].apply(
            lambda holds: {int(list(h.keys())[0]): list(h.values())[0] for h in holds if h}
        )
        
        print(f"✅ Loaded {len(df)} routes")
        return df
    except Exception as e:
        print(f"⚠️ Could not load database: {e}")
        return pd.DataFrame()

database_df = load_database()

# Model cache - now supports multiple models
model_cache = {}

def load_model_for_config(model_name: str):
    """Load model based on selection"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    print(f"Loading {model_name}...")
    try:
        model_id = MODEL_CONFIGS[model_name]
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        model = KilterGPT(
            tokenizer=tokenizer, n_embd=256, n_head=4, n_layer=6,
            n_positions=128, dropout=0.1, use_custom_loss=False, device=DEVICE
        )
        
        checkpoint_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        
        if list(state_dict.keys())[0].startswith("model."):
            model.load_state_dict(state_dict, strict=False)
        else:
            model.model.load_state_dict(state_dict, strict=True)
        
        model.to(DEVICE)
        model.eval()
        
        model_cache[model_name] = (model, tokenizer)
        print(f"✅ {model_name} loaded")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None

# Load default model
current_model_name = "KilterGPT Base"
model, tokenizer = load_model_for_config(current_model_name)

@functools.lru_cache(maxsize=1)
def load_board_image():
    try:
        response = requests.get(BOARD_IMG_URL)
        board_img = Image.open(BytesIO(response.content)).convert("RGBA")
        return board_img
    except:
        return Image.new('RGBA', (1080, 1343), color=(52, 62, 80, 255))

board_img = load_board_image()

def grade_to_display_text(grade_value: int) -> str:
    letter_grade = dp.difficulty_to_letter_grade(grade_value)
    v_grade = dp.difficulty_to_v_grade(grade_value)
    return f"{letter_grade} / V{v_grade}"

def find_nearest_hold(x: int, y: int, threshold: int = 40) -> Optional[int]:
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

def draw_board_with_holds(selected_holds_list: List[Tuple[int, str]], 
                          holds_data: List[Dict] = None,
                          candidate_holds: List[int] = None,
                          title: str = "") -> Image.Image:
    img = board_img.copy()
    draw = ImageDraw.Draw(img)
    radius, width = 30, 4
    
    if candidate_holds:
        for hold_id in candidate_holds:
            idx = HOLD_ID.index(hold_id)
            x, y = HOLDCOORDINATES[idx]
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                        outline='#DDDDDD', width=width)

    if holds_data:
        for hold in holds_data:
            for hold_id_str, role in hold.items():
                hold_id = int(hold_id_str)
                idx = HOLD_ID.index(hold_id)
                x, y = HOLDCOORDINATES[idx]
                color = HOLD_COLORS.get(role, "#FFFFFF")
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                           outline=color, width=width)
    else:
        for hold_id, role in selected_holds_list:
            idx = HOLD_ID.index(hold_id)
            x, y = HOLDCOORDINATES[idx]
            color = HOLD_COLORS.get(role, "#FFFFFF")
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                       outline=color, width=width)
    
    if title:
        font = ImageFont.truetype(FONT_PATH, 28)
        bbox = draw.textbbox((0, 0), title, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = img.width - w - 20, 10
        draw.rectangle([(x, y), (x + w + 20, y + h + 20)], fill=(255, 255, 255, 170))
        draw.text((x + 8, y + 8), title, fill="black", font=font)
    
    return img

def calculate_similarity_perfect_match(route1_dict: Dict, route2_dict: Dict) -> Tuple[int, int]:
    """Optimized similarity calculation using precomputed dicts"""
    route1_holds = set(route1_dict.keys())
    route2_holds = set(route2_dict.keys())
    
    hold_matches = len(route1_holds & route2_holds)
    perfect_matches = sum(1 for hold_id in route1_holds & route2_holds 
                         if route1_dict[hold_id] == route2_dict[hold_id])
    
    return perfect_matches, hold_matches

def search_similar_boulders(generated_holds: List[Dict], angle: int, grade: int, 
                           sort_by: str = "similarity", match_type: str = "perfect (correct holds + roles)") -> Tuple[pd.DataFrame, List[Dict]]:
    """Faster search using precomputed hold sets"""
    if database_df.empty:
        return pd.DataFrame(), []
    
    gen_dict = {int(list(h.keys())[0]): list(h.values())[0] for h in generated_holds if h}
    
    # Vectorized similarity computation
    results = []
    for _, row in database_df.iterrows():
        perfect_matches, hold_matches = calculate_similarity_perfect_match(gen_dict, row['hold_set'])
        results.append({
            'Name': row.get('name', 'Unknown'),
            'Setter': row.get('setter_username', 'Anonymous')[:15],
            'Angle': f"{int(row.get('angle_y', 0))}°",
            'Grade': grade_to_display_text(int(row.get('display_difficulty', 18))),
            'Perfect': perfect_matches,
            'Holds': hold_matches,
            'perfect_matches': perfect_matches,
            'hold_matches': hold_matches,
            'holds_data': row.get('holds_data', []),
            'angle_y': row.get('angle_y', 40),
            'display_difficulty': row.get('display_difficulty', 18)
        })
    
    results = sorted(results, key=lambda x: (x['perfect_matches'], x['hold_matches']), reverse=True)
    top_df = pd.DataFrame(results)
    top_df = top_df[top_df['Angle'].str.replace('°','').astype(int) == angle].head(10)
    return top_df[['Name', 'Setter', 'Angle', 'Grade', 'Perfect', 'Holds']], results

def handle_board_click(selected_holds_state: List, evt: gr.SelectData) -> Tuple[Image.Image, List]:
    x, y = evt.index
    hold_id = find_nearest_hold(x, y)
    
    if hold_id is None:
        return gr.update(), selected_holds_state
    
    existing_idx = None
    for i, (h_id, _) in enumerate(selected_holds_state):
        if h_id == hold_id:
            existing_idx = i
            break
    
    if existing_idx is not None:
        current_role = selected_holds_state[existing_idx][1]
        current_idx = hold_cycle.index(current_role)
        next_role = hold_cycle[(current_idx + 1) % len(hold_cycle)]
        selected_holds_state[existing_idx] = (hold_id, next_role)
    else:
        selected_holds_state.append((hold_id, "start"))
    
    img = draw_board_with_holds(selected_holds_state)
    return img, selected_holds_state

def clear_selected_holds() -> Tuple[Image.Image, List]:
    return draw_board_with_holds([]), []

def switch_model(model_name: str):
    """Switch between models"""
    global model, tokenizer, current_model_name
    model, tokenizer = load_model_for_config(model_name)
    current_model_name = model_name
    return f"Switched to {model_name}"

def generate_route_animated(angle: int, grade: int, num_holds: int, 
                           temperature: float, top_p: float, 
                           selected_holds_state: List,
                           use_constraints: bool,
                           animate: bool,
                           model_name: str):
    """Generate with selected model"""
    global GENERATED_ROUTES_DB, STATS
    
    # Switch model if needed
    gen_model, gen_tokenizer = load_model_for_config(model_name)
    if gen_model is None:
        yield draw_board_with_holds(selected_holds_state), f"Total: {STATS['total']}", f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}"
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
            nonlocal animation_frames, generated_holds
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
        
        if animate and animation_frames:
            for i, frame in enumerate(animation_frames):
                if i >= len(selected_holds_state):
                    partial_holds = generated_holds[:i]
                    img = draw_board_with_holds([], holds_data=partial_holds, 
                                            candidate_holds=animation_frames[i-len(selected_holds_state)]['candidates_hold_id'])
                    yield img, f"Total: {STATS['total']}", f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}"
                    time.sleep(0.5)
                    
                    if i < len(generated_holds):
                        partial_holds = generated_holds[:i+1]
                        img = draw_board_with_holds([], holds_data=partial_holds)
                        yield img, f"Total: {STATS['total']}", f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}"
                        time.sleep(0.5)
        
        title = f"{angle}° | {grade_to_display_text(grade)}"
        img = draw_board_with_holds([], holds_data=generated_holds, title=title)
        
        GENERATED_ROUTES_DB.append({
            'holds_data': generated_holds, 'angle': angle, 'grade': grade,
            'timestamp': time.time(), 'liked': None, 'model': model_name
        })
        STATS['total'] += 1
        save_stats(STATS, GENERATED_ROUTES_DB)
        
        yield img, f"Total: {STATS['total']}", f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}"
        
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        yield draw_board_with_holds(selected_holds_state), f"Total: {STATS['total']}", f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}"

def load_preset(preset_name: str):
    preset = PRESET_EXAMPLES[preset_name]
    return (preset["angle"], preset["grade"], preset["selected_holds"],
            draw_board_with_holds(preset["selected_holds"]))

def handle_search(angle: int, grade: int, selected_holds: List, sort_by: str, match_type: str):
    holds_data = [{str(hold_id): role} for hold_id, role in selected_holds]
    results_df, full_data = search_similar_boulders(holds_data, angle, grade, sort_by, match_type)
    return results_df, full_data

def reveal_selected_boulder(evt: gr.SelectData, full_results):
    if not full_results or len(full_results) == 0:
        return None
    
    selected_idx = evt.index[0]
    if selected_idx >= len(full_results):
        return None
    
    boulder_data = full_results[selected_idx]
    holds_data = boulder_data.get('holds_data', [])
    angle = boulder_data.get('angle_y', 40)
    grade = boulder_data.get('display_difficulty', 18)
    
    title = f"{int(angle)}° | {grade_to_display_text(int(grade))}"
    img = draw_board_with_holds([], holds_data=holds_data, title=title)
    return img

def like_route():
    global GENERATED_ROUTES_DB, STATS
    if GENERATED_ROUTES_DB:
        if GENERATED_ROUTES_DB[-1]['liked'] != True:
            if GENERATED_ROUTES_DB[-1]['liked'] == False:
                STATS['dislikes'] -= 1
            GENERATED_ROUTES_DB[-1]['liked'] = True
            STATS['likes'] += 1
            save_stats(STATS, GENERATED_ROUTES_DB)
    return f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}"

def dislike_route():
    global GENERATED_ROUTES_DB, STATS
    if GENERATED_ROUTES_DB:
        if GENERATED_ROUTES_DB[-1]['liked'] != False:
            if GENERATED_ROUTES_DB[-1]['liked'] == True:
                STATS['likes'] -= 1
            GENERATED_ROUTES_DB[-1]['liked'] = False
            STATS['dislikes'] += 1
            save_stats(STATS, GENERATED_ROUTES_DB)
    return f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}"

# Exploration visualizations
def create_hold_frequency_heatmap():
    """Mock hold frequency visualization"""
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

def create_grade_angle_distribution():
    """Grade vs Angle distribution"""
    if database_df.empty:
        return go.Figure()
    
    fig = px.scatter(database_df.head(500), x='angle_y', y='display_difficulty',
                     color='quality_average', title="Grade vs Angle Distribution",
                     labels={'angle_y': 'Angle', 'display_difficulty': 'Grade'})
    return fig

def create_generation_stats():
    """Stats from generated routes"""
    if not GENERATED_ROUTES_DB:
        return go.Figure()
    
    likes = STATS['likes']
    dislikes = STATS['dislikes']
    
    fig = go.Figure(data=[
        go.Bar(x=['Likes', 'Dislikes'], y=[likes, dislikes],
               marker_color=['green', 'red'])
    ])
    fig.update_layout(title="User Feedback Stats", yaxis_title="Count")
    return fig

# Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    title="KilterGPT",
    css="""
    .gradio-container {max-width: 1800px !important; font-size: 18px !important}
    .board-image {cursor: crosshair !important; height: 700px !important}
    h1 {font-size: 3em !important}
    """
) as demo:
    
    gr.Markdown(f"# 🧗 KilterGPT - AI Climbing Route Generator")
    gr.Markdown("### Generate custom boulder routes for the Kilter Board using GPT-based AI!")
    
    selected_holds_state = gr.State([])
    search_results_full = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🎛️ Parameters")            
            angle_input = gr.Slider(20, 60, value=40, step=5, label=f"Angle: {40}")
            grade_input = gr.Slider(12, 29, value=18, step=1, label=f"Grade: {grade_to_display_text(18)}")
            num_holds_input = gr.Slider(4, 20, value=10, step=1, label=f"Num Holds: {10}")
            
            with gr.Accordion("⚙️ Advanced", open=True):
                temperature = gr.Slider(0.1, 1, value=0.3, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                use_constraints = gr.Checkbox(label="Constrained Generation", value=True)
                animate_gen = gr.Checkbox(label="Animated Generation", value=True)
            
            with gr.Row():
                beginner_btn = gr.Button("Example 1: Easy", size="sm", scale=1)
                intermediate_btn = gr.Button("Example 2: Hard", size="sm", scale=1)
            
            generate_btn = gr.Button("🎲 Generate", variant="stop", size="lg")
        
        with gr.Column(scale=8):
            model_selector = gr.Radio(
                list(MODEL_CONFIGS.keys()),
                value="KilterGPT Base",
                show_label=False,
            )
            
            board_display = gr.Image(
                value=draw_board_with_holds([]),
                type="pil",
                interactive=True,
                height=700,
                show_label=False,
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Feedbacks")
            clear_btn = gr.Button("🗑️ Clear", size="lg")
            like_btn = gr.Button("👍 Like", variant="secondary", size="lg", interactive=False)
            dislike_btn = gr.Button("👎 Dislike", variant="secondary", size="lg", interactive=False)
            
            gr.Markdown("#### 📊 Stats")
            with gr.Row():
                total_label = gr.Markdown(f"Total: {STATS['total']}")
                feedback_label = gr.Markdown(f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}")
            
            search_btn = gr.Button("🔍 Similar boulders", size="lg")
    
    gr.Markdown("## 🔍 Similar Boulders")
    with gr.Row():
        with gr.Column(scale=2):
            sort_dropdown = gr.Dropdown(choices=["similarity", "grade", "angle"], value="similarity", label="Sort by")
            match_type_radio = gr.Radio(
                choices=["perfect (correct holds + roles)", "partial (only same holds)"], 
                value="perfect (correct holds + roles)", 
                show_label=False,
            )
            search_results = gr.Dataframe(label="Results")
        with gr.Column(scale=2):
            similar_boulder_img = gr.Image(label="Boulder", type="pil", height=700)
    
    gr.Markdown("## 🔍 Explore More")
    with gr.Tabs():
        with gr.TabItem("Hold Frequency"):
            hold_freq_plot = gr.Plot(label="Hold Usage")
            gr.Button("🔄 Refresh").click(fn=create_hold_frequency_heatmap, outputs=[hold_freq_plot])
        
        with gr.TabItem("Grade Distribution"):
            grade_plot = gr.Plot(label="Grade vs Angle")
            gr.Button("🔄 Refresh").click(fn=create_grade_angle_distribution, outputs=[grade_plot])
        
        with gr.TabItem("Generation Stats"):
            stats_plot = gr.Plot(label="Feedback Stats")
            gr.Button("🔄 Refresh").click(fn=create_generation_stats, outputs=[stats_plot])
    
    # Event handlers
    grade_input.change(fn=lambda g: gr.update(label=f"Grade: {grade_to_display_text(g)}"),
                      inputs=[grade_input], outputs=[grade_input])
    angle_input.change(fn=lambda a: gr.update(label=f"Angle: {a}"),
                      inputs=[angle_input], outputs=[angle_input])
    num_holds_input.change(fn=lambda n: gr.update(label=f"Num Holds: {n}"),
                          inputs=[num_holds_input], outputs=[num_holds_input])
    
    beginner_btn.click(fn=lambda: load_preset("Beginner"),
                      outputs=[angle_input, grade_input, selected_holds_state, board_display])
    intermediate_btn.click(fn=lambda: load_preset("Intermediate"),
                          outputs=[angle_input, grade_input, selected_holds_state, board_display])
    
    board_display.select(fn=handle_board_click,
                        inputs=[selected_holds_state],
                        outputs=[board_display, selected_holds_state])
    
    clear_btn.click(fn=clear_selected_holds, outputs=[board_display, selected_holds_state])
    
    model_selector.change(fn=switch_model, inputs=[model_selector], outputs=[])
    
    generate_btn.click(
        fn=generate_route_animated,
        inputs=[angle_input, grade_input, num_holds_input, temperature, top_p,
               selected_holds_state, use_constraints, animate_gen, model_selector],
        outputs=[board_display, total_label, feedback_label]
    ).then(lambda: gr.update(interactive=True), None, [like_btn])\
    .then(lambda: gr.update(interactive=True), None, [dislike_btn])
    
    like_btn.click(fn=like_route, outputs=[feedback_label])
    dislike_btn.click(fn=dislike_route, outputs=[feedback_label])
    
    search_btn.click(fn=handle_search,
                    inputs=[angle_input, grade_input, selected_holds_state, sort_dropdown, match_type_radio],
                    outputs=[search_results, search_results_full])
    
    search_results.select(fn=reveal_selected_boulder,
                         inputs=[search_results_full],
                         outputs=[similar_boulder_img])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)