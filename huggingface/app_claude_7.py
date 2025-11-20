"""
KilterGPT Gradio Web Interface
"""
import gradio as gr
import torch
import requests
from PIL import Image
from io import BytesIO
import os
import sys
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
import functools
import pandas as pd
import json

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import DataPreprocessing
from src.temp_gpt import KilterGPT
from utils import (
    grade_to_display_text, find_nearest_hold, draw_board_with_holds,
    search_similar_boulders, generate_route_animated, hold_cycle,
    create_hold_frequency_heatmap, create_grade_angle_distribution, create_generation_stats, HOLD_ID
)



# Configuration
MODEL_CONFIGS = {
    "KilterGPT Base": "nottreepat/testtest",
    "KilterGPT (with data-augment)": "nottreepat/testtest",
    "KilterGPT (with order-invariant loss)": "nottreepat/testtest"
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOARD_IMG_URL = "https://huggingface.co/datasets/nottreepat/climbs_cleaned/resolve/main/full_board_commercial.png"
DATABASE_URL = "https://huggingface.co/datasets/nottreepat/climbs_cleaned/resolve/main/climbs_cleaned.csv"
STATS_FILE = "huggingface/generation_stats.json"
GENERATED_ROUTES_FILE = "huggingface/generated_routes.json"
FONT_PATH = "huggingface/font/Arial Bold.ttf"
TECHNICAL_BLOG_FILE = "huggingface/technical_blog.md"

dp = DataPreprocessing()

PRESET_EXAMPLES = {
    "Beginner": {"angle": 30, "grade": 15, "selected_holds": [(1133, "start"), (1184, "start")]},
    "Intermediate": {"angle": 40, "grade": 25, "selected_holds": [(1100, "start"), (1150, "start")]},
}
PRESET_EXAMPLE = "angle40_grade18_feet1169_feet1183_feet1198_start1234_start1236_hand1268_hand1284_hand1317_hand1353_finish1387"

# Load/save stats
def load_stats():
    stats, generated_routes = {'total': 0, 'likes': 0, 'dislikes': 0}, []
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            print(f"✅ Loaded STATS_FILE: {stats}")
        except: pass
    if os.path.exists(GENERATED_ROUTES_FILE):
        try:
            with open(GENERATED_ROUTES_FILE, 'r') as f:
                generated_routes = json.load(f)
            print(f"✅ Loaded GENERATED_ROUTES_FILE: {generated_routes[-1]}")
        except: pass
    return stats, generated_routes

def save_stats(stats, routes_db):
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
        with open(GENERATED_ROUTES_FILE, 'w') as f:
            json.dump(routes_db, f)
    except Exception as e:
        print(f"Failed to save stats: {e}")

STATS, GENERATED_ROUTES_DB = load_stats()

# Load database
@functools.lru_cache(maxsize=1)
def load_database():
    try:
        df = pd.read_csv(DATABASE_URL)
        df = df[(df['quality_average'] >= 2.5) & (df['angle_y'] >= 20) & (df['angle_y'] <= 60)]
        
        if 'frames' in df.columns:
            df['holds_data'] = df['frames'].apply(
                lambda x: dp.str_to_holds_data(x) if isinstance(x, str) else []
            )
        
        df['hold_set'] = df['holds_data'].apply(
            lambda holds: {int(list(h.keys())[0]): list(h.values())[0] for h in holds if h}
        )
        
        print(f"✅ Loaded {len(df)} routes")
        return df
    except Exception as e:
        print(f"⚠️ Could not load database: {e}")
        return pd.DataFrame()

database_df = load_database()

# Model cache
model_cache = {}

def load_model_for_config(model_name: str):
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

# Load board image
@functools.lru_cache(maxsize=1)
def load_board_image():
    try:
        response = requests.get(BOARD_IMG_URL)
        return Image.open(BytesIO(response.content)).convert("RGBA")
    except:
        return Image.new('RGBA', (1080, 1343), color=(52, 62, 80, 255))

board_img = load_board_image()
model, tokenizer = load_model_for_config("KilterGPT Base")

# Handler functions
def handle_board_click(selected_holds_state, evt: gr.SelectData):
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
    
    img = draw_board_with_holds(board_img, selected_holds_state, font_path=FONT_PATH)
    return img, selected_holds_state

def clear_selected_holds():
    return draw_board_with_holds(board_img, [], font_path=FONT_PATH), []

def switch_model(model_name: str):
    load_model_for_config(model_name)
    return f"Switched to {model_name}"

def load_preset(preset_name: str):
    preset = PRESET_EXAMPLES[preset_name]
    return (preset["angle"], preset["grade"], preset["selected_holds"],
            draw_board_with_holds(board_img, preset["selected_holds"], font_path=FONT_PATH))

def handle_search(angle, grade, selected_holds, sort_by, match_type):
    holds_data = [{str(hold_id): role} for hold_id, role in selected_holds]
    results_df, full_data = search_similar_boulders(holds_data, angle, grade, database_df, dp, sort_by, match_type)
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
    
    title = f"{int(angle)}° | {grade_to_display_text(int(grade), dp)}"
    img = draw_board_with_holds(board_img, [], holds_data=holds_data, title=title, font_path=FONT_PATH)
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

def generate_wrapper(*args):
    """Wrapper to pass dependencies to generate function"""
    yield from generate_route_animated(
        *args,
        load_model_fn=load_model_for_config,
        dp=dp,
        board_img=board_img,
        font_path=FONT_PATH,
        stats=STATS,
        routes_db=GENERATED_ROUTES_DB,
        save_stats_fn=save_stats
    )

# def handle_tokenized_click(tokenized_selected_holds_state, evt: gr.SelectData):
#     # Reuse your existing click logic
#     img, new_state = handle_board_click(tokenized_selected_holds_state, evt)
#     print(new_state)
#     # Convert hold-state format: [(hold_id, role), ...] → text for tokenizer
#     hold_strings = [f"{role}{h_id}" for h_id, role in new_state]
    

#     # Tokenize using your existing tokenizer
#     token_ids = tokenizer.encode("_".join(hold_strings), return_tensors="pt", add_special_tokens=True)

#     # Display text + tokens
#     decoded_input = tokenizer.decode(token_ids[0], skip_special_tokens=False)
    
    
#     tokens_text = f"Tokenized Input: {token_ids[0]}"
#     holds_text =  f"When Decoded back: {decoded_input}"
    

#     return img, new_state, holds_text, tokens_text


# ============================================================================
# Section 1: Tokenization
# ============================================================================

def frames_to_original(frames_str: str) -> str:
    """Convert frames back to original format (without angle/grade, role as number)"""
    if not frames_str:
        return ""
    
    parts = frames_str.split("_")
    result = []
    
    role_map = {"start": "12", "hand": "13", "finish": "14", "feet": "15"}
    
    for part in parts:
        if part.startswith(("angle", "grade")):
            continue
        
        for role, num in role_map.items():
            if part.startswith(role):
                hold_id = part[len(role):]
                result.append(f"p{hold_id}r{num}")
                break
    
    return "".join(result)

def frames_to_converted(selected_holds, angle, grade) -> str:
    """Convert selected holds to frames format with angle/grade"""
    parts = [f"angle{angle}", f"grade{grade}"]
    parts.extend([f"{role}{hold_id}" for hold_id, role in selected_holds])
    return "_".join(parts)

def tokenize_frames(frames_str: str, tokenizer) -> tuple:
    """Tokenize frames and return token_ids and decoded tokens"""
    token_ids = tokenizer.encode(frames_str, return_tensors="pt", add_special_tokens=True)
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids[0]]
    return token_ids[0].tolist(), decoded_tokens

def load_preset_example(tokenizer):
    """Load preset example for tokenization demo"""
    parts = PRESET_EXAMPLE.split("_")
    angle = int(parts[0].replace("angle", ""))
    grade = int(parts[1].replace("grade", ""))
    
    selected_holds = []
    for part in parts[2:]:
        for role in ["start", "hand", "feet", "finish"]:
            if part.startswith(role):
                hold_id = int(part[len(role):])
                selected_holds.append((hold_id, role))
                break

    img = draw_board_with_holds(board_img, selected_holds, font_path=FONT_PATH, show_ids=True)
    
    original_frames = frames_to_original(PRESET_EXAMPLE)
    converted_frames = frames_to_converted(selected_holds, angle, grade)
    token_ids, decoded_tokens = tokenize_frames(converted_frames, tokenizer)
    
    return (
        img, selected_holds, angle, grade,
        f"{original_frames}",
        f"{converted_frames}",
        f"{token_ids}",
        f"{decoded_tokens}"
    )

def update_tokenization_display(selected_holds, angle, grade, tokenizer):
    """Update all tokenization displays based on current state"""
    if not selected_holds:
        return "", "", "", ""
    
    converted_frames = frames_to_converted(selected_holds, angle, grade)
    original_frames = frames_to_original(converted_frames)
    token_ids, decoded_tokens = tokenize_frames(converted_frames, tokenizer)
    
    return (
        f"{original_frames}",
        f"{converted_frames}",
        f"{token_ids}",
        f"{decoded_tokens}"
    )
    
def clear_tokenization_board(tokenizer):
    """Clear board for tokenization section"""
    img = draw_board_with_holds(board_img, [], font_path=FONT_PATH, show_ids=True)
    return img, [], "", "", "", ""

# ============================================================================
# Section 3: Order-Invariant Loss
# ============================================================================
 
def clear_loss_board():
    """Clear board for tokenization section"""
    img = draw_board_with_holds(board_img, [], font_path=FONT_PATH, show_ids=True)
    return img, [], "", 5, ""

def update_loss_display(selected_holds, angle, grade, position, tokenizer, model):
    if not selected_holds:
        return "", "", "", []

    converted_frames = frames_to_converted(selected_holds, angle, grade)
    token_ids = tokenizer.encode(converted_frames, return_tensors="pt", add_special_tokens=True).to(DEVICE)

    token_position = position + 1
    seq_len = token_ids.shape[1]

    if token_position >= seq_len or token_position < 1:
        return converted_frames, "", f"Position {position} is out of range.", []

    with torch.no_grad():
        logits = model.model(token_ids).logits[0, token_position - 1]

    eos_idx = token_ids[0].tolist().index(tokenizer.eos_token_id)
    remaining_tokens = token_ids[0, token_position:eos_idx].tolist()

    if not remaining_tokens:
        return converted_frames, "", f"No remaining tokens at position {position}", []

    probs = F.softmax(logits, dim=-1)

    # --- NEW: split displays cleanly ---
    prompt_text = "_".join(converted_frames.split("_")[:position])

    remaining_text = tokenizer.decode(remaining_tokens, add_special_tokens=True).split()
    remaining_text = f"Any of : {remaining_text}"

    # --- NEW: small dataframe with token, prob, log_prob ---
    table = []
    for token_id in remaining_tokens[:5]:
        tok = tokenizer.decode([token_id])
        p = probs[token_id].item()
        lp = torch.log(probs[token_id]).item()
        table.append([tok, round(p, 6), round(lp, 6)])

    return converted_frames, prompt_text, remaining_text, table

def load_preset_example_for_loss(tokenizer, model, position):
    parts = PRESET_EXAMPLE.split("_")
    angle = int(parts[0][5:])
    grade = int(parts[1][5:])

    selected_holds = []
    for part in parts[2:]:
        for role in ["start", "hand", "feet", "finish"]:
            if part.startswith(role):
                selected_holds.append((int(part[len(role):]), role))
                break

    img = draw_board_with_holds(board_img, selected_holds, font_path=FONT_PATH, show_ids=True)

    conv, prompt, remaining, table = update_loss_display(selected_holds, angle, grade, position, tokenizer, model)

    return img, selected_holds, angle, grade, conv, prompt, remaining, table


# ============================================================================
# Section 3: interface utils
# ============================================================================

 # Create wrapper functions that have access to tokenizer
def handle_tok_click(selected_holds, angle, grade, evt: gr.SelectData):
    x, y = evt.index
    hold_id = find_nearest_hold(x, y)
    
    if hold_id is None:
        return gr.update(), selected_holds, *update_tokenization_display(selected_holds, angle, grade, tokenizer)
    
    existing_idx = None
    for i, (h_id, _) in enumerate(selected_holds):
        if h_id == hold_id:
            existing_idx = i
            break
    
    if existing_idx is not None:
        current_role = selected_holds[existing_idx][1]
        current_idx = hold_cycle.index(current_role)
        next_role = hold_cycle[(current_idx + 1) % len(hold_cycle)]
        selected_holds[existing_idx] = (hold_id, next_role)
    else:
        selected_holds.append((hold_id, "start"))
    
    img = draw_board_with_holds(board_img, selected_holds, font_path=FONT_PATH, show_ids=True)
    return img, selected_holds, *update_tokenization_display(selected_holds, angle, grade, tokenizer)
     

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
    
    gr.Markdown("# 🧗 KilterTransformer - AI Climbing Route Generator")
    gr.Markdown("### Generate boulder for the Kilter Board using GPT-based AI. Trained from 70,000+ real boulders")
    
    selected_holds_state = gr.State([])
    search_results_full = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🎛️ Parameters")
            angle_input = gr.Slider(20, 60, value=40, step=5, label=f"Angle: {40}")
            grade_input = gr.Slider(12, 29, value=18, step=1, label=f"Grade: {grade_to_display_text(18, dp)}")
            num_holds_input = gr.Slider(4, 20, value=10, step=1, label=f"Num Holds: {10}")
            
            with gr.Accordion("⚙️ Advanced", open=True):
                temperature = gr.Slider(0.1, 1, value=0.3, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                use_constraints = gr.Checkbox(label="Constrained Generation", value=True)
                animate_gen = gr.Checkbox(label="Animated Generation", value=True)
        
        with gr.Column(scale=8):
            model_selector = gr.Radio(
                list(MODEL_CONFIGS.keys()),
                value="KilterGPT Base",
                show_label=False,
            )
            
            board_display = gr.Image(
                value=draw_board_with_holds(board_img, [], font_path=FONT_PATH),
                type="pil",
                interactive=True,
                height=700,
                show_label=False,
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🧗 Generate Boulders")
            with gr.Row():
                beginner_btn = gr.Button("Example 1: Easy", size="md", scale=1)
                intermediate_btn = gr.Button("Example 2: Hard", size="md", scale=1)
            
            generate_btn = gr.Button("🎲 Generate", variant="primary", size="lg")
            
            gr.Markdown("### 🎛️ Feedbacks")
            clear_btn = gr.Button("🗑️ Clear", size="lg")
            like_btn = gr.Button("👍 Like", variant="secondary", size="lg", interactive=False)
            dislike_btn = gr.Button("👎 Dislike", variant="secondary", size="lg", interactive=False)
            
            gr.Markdown("#### 📊 Stats")
            with gr.Row():
                total_label = gr.Markdown(f"Total: {STATS['total']}")
                feedback_label = gr.Markdown(f"👍 {STATS['likes']} | 👎 {STATS['dislikes']}")
            
            search_btn = gr.Button("🔍 Similar boulders", size="lg")
    
    with gr.Accordion("## 🔍 Similar Boulders", open=False) as search_accordion:
        with gr.Row():
            with gr.Column(scale=2):
                sort_dropdown = gr.Dropdown(choices=["similarity", "grade", "angle"], value="similarity", label="Sort by")
                match_type_radio = gr.Radio(
                    choices=["perfect (correct holds + roles)", "partial (only same holds)"],
                    value="perfect (correct holds + roles)",
                    show_label=False,
                )
                search_results = gr.Dataframe(
                    value=database_df.head(10)[['name', 'setter_username', 'angle_y', 'display_difficulty']].rename(
                        columns={'name':'Name', 'setter_username':'Setter','angle_y':'Angle','display_difficulty':'Grade'}
                    ) if not database_df.empty else pd.DataFrame(),
                    label="Results"
                )
            
            with gr.Column(scale=2):
                similar_boulder_img = gr.Image(label="Boulder", type="pil", height=700)
    
    
    # Event handlers
    grade_input.change(fn=lambda g: gr.update(label=f"Grade: {grade_to_display_text(g, dp)}"),
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
        fn=generate_wrapper,
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
    
    

    
    gr.Markdown("## 🧗 KilterGPT Technical Deep Dive")

    # ====================================================================
    # Section 0: Holds Encoding
    # ====================================================================
    holds_encoding_file = Path(__file__).parent / "blog_texts" / "holds_encoding.md"
    gr.Markdown(holds_encoding_file.read_text())
    
    
    # ====================================================================
    # Section 1: Tokenization
    # ====================================================================
    with gr.Accordion("1 Tokenization of Boulders", open=True):  
        
        tok_state = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1, min_width=120):
                with gr.Row():
                    tok_angle = gr.Number(value=40, label="Angle", minimum=20, maximum=60, step=5)
                    tok_grade = gr.Number(value=18, label="Grade", minimum=12, maximum=29, step=1)
                tok_preset_btn = gr.Button("📋 Load Example", variant="primary", size="md")
                tok_clear_btn = gr.Button("🗑️ Clear", size="md")
            
            with gr.Column(scale=2):
                tok_board = gr.Image(
                    value=draw_board_with_holds(board_img, [], font_path=FONT_PATH, show_ids=True),
                    type="pil", interactive=True, height=500, label="Select holds"
                )
        
            with gr.Column(scale=3):
                tok_original = gr.Textbox(label="1. Original Frames (Dataset Format)", lines=3, max_lines=3)
                tok_converted = gr.Textbox(label="2. Converted Frames (With Angle/Grade)", lines=3, max_lines=3)
                tok_tokenized = gr.Textbox(label="3. Token IDs", lines=3, max_lines=3)
                tok_decoded = gr.Textbox(label="4. Decoded Tokens", lines=3, max_lines=3)
        
        
        # Event handlers
        tok_preset_btn.click(
            fn=lambda: load_preset_example(tokenizer),
            outputs=[tok_board, tok_state, tok_angle, tok_grade, 
                    tok_original, tok_converted, tok_tokenized, tok_decoded]
        )
        
        tok_board.select(
            fn=handle_tok_click,
            inputs=[tok_state, tok_angle, tok_grade],
            outputs=[tok_board, tok_state, tok_original, tok_converted, tok_tokenized, tok_decoded]
        )
        
        tok_clear_btn.click(
            fn=lambda: clear_tokenization_board(tokenizer),
            outputs=[tok_board, tok_state, tok_original, tok_converted, tok_tokenized, tok_decoded]
        )
        
        tok_angle.change(
            fn=lambda s, a, g: update_tokenization_display(s, a, g, tokenizer),
            inputs=[tok_state, tok_angle, tok_grade],
            outputs=[tok_original, tok_converted, tok_tokenized, tok_decoded]
        )
        
        tok_grade.change(
            fn=lambda s, a, g: update_tokenization_display(s, a, g, tokenizer),
            inputs=[tok_state, tok_angle, tok_grade],
            outputs=[tok_original, tok_converted, tok_tokenized, tok_decoded]
        )
        
    # ====================================================================
    # Section 2: Model Architecture
    # ====================================================================
    with gr.Accordion("2 GPT Model Architecture", open=True):
        model_arch_file = Path(__file__).parent / "blog_texts" / "model_arch.md"
        gr.Markdown(model_arch_file.read_text())
    
        
        # gr.Image("huggingface/gpt_architecture.png", label="GPT Architecture Diagram")
    
    
    # ====================================================================
    # Section 3: Order-Invariant Loss
    # ====================================================================
    
    with gr.Accordion("3 Order-Invariant Loss Function", open=True):
        order_loss_file = Path(__file__).parent / "blog_texts" / "order_loss.md"
        gr.Markdown(order_loss_file.read_text())
        



    # ====================================================================
    # Section 4: Data Augmentation
    # ====================================================================
    with gr.Accordion("4 Data Augmentation: Shuffling Hold Sequences", open=True):
        gr.Markdown("""
        ### Why Shuffle Holds?
        
        This method in training further amplifies the goal that we want the model to perform:
        
        1. **Bidirectional Generation**: Generate from middle → start + end, not just left-to-right
        2. **Order Invariance**: Learn that different orders = same route


        """)
        
        
        # add eval loss + test loss instead
            
    


if __name__ == "__main__":
    demo.launch(share=False)
    

"""
x push to huggingface spaces
[] automate deployment
[] fold the search section, only reveal when there's search button click
[] in search section, auto show the first boulder in results_df, show the boulder img even without clicking of generate button
- faster search/retrieval
[] clustering with style/type e.g. crimps, pinches, large/small moves, explosive/compression moves
[] Trend Explorer: Visualize how grades/angles/styles cluster, Which grades have the widest spread? Add time dimension → show evolution of style.
x implement models selection (mock with same model for all three buttons for now)

TECHNICAL BLOG
[] tokenization: show boulder with hold ids -> frame, holds data, (+angle +grade)tokenized data, (add BOS, EOS, PAD)
    explain why we add angle+grade as tokens (prompting)
[] GPT model architcture + Language Modeling head (insert pic)
[] order-invariant loss: recall the boulder above -> show original tokenized data
    given context, it's fine for the model to predict any of the remaining holds, regardless of order
    show that two valid but different next-token predictions yeild the same boulder/loss
[] Data Augmentation: shuffling the holds sequence
    the model should learn to generate by not just completing from the starting holds, but bidrectionally, from the middle to the start+end
    grade and angle are still mandatory tokens
    show augmentaiton examples with pic
[] GPT logits/probs of top_N tokens in the vocab size
[] Datasets Stats
    - hold_difficulty heatmap; hover-info be : [hold_id, avg_difficulty(vgrade/lettergrade), num_climbs using this hold], 
    two plots side-by-side (one with main holds, the other with auxiliary holds)
    - plots of various params vs. grade : [angle, num_holds, ascentionist count, quality, quality, avr_reach]
    - popularity vs quality
    - plot correlation of variables in datasets_df
    - clustering with k means ; hover-info be [name, grade, angle, num_holds, ascentionist count, avr_reach]
[] GPT Training
    - GPT-2 Style Transformer
    - eval/train loss of differnt models from wandb
[] Difficulty Prediction Models
    - BERT-based Classifier
        Input: Hold sequence embeddings
        Accuracy: 78% (±1 V-grade)
        R²: 0.82
    - XGBoost Regressor
        Features: 15 engineered features (angle, holds, reach, density, etc.)
        Within ±1 grade: 85%
        MAE: 1.2 grades


more important stuffs
[] add more data on the harder grades (maybe discrimate remove some climbs (with bad quality) on the popular grades)
[] actaully use model selection; retrain GPT models, and host with diff. training methods (loss+data aug)

Not really importnat but fun
[] difficulty or quality predictor using BERT and/or XGBoost

x Connect dataset climbs_cleaned.csv via huggingface datasets url and use in search results: https://huggingface.co/datasets/nottreepat/climbs_cleaned/blob/main/climbs_cleaned.csv
x shorter/smaller image (go back to older ver.) so that we can see gereate buttons without scrolling
x bigger fonts
x remove info from the search result markdown
x remove info for sliders (e.g. 20° = Slight, 60° = Steep)
x remove this markdown: "Click to select holds. Repeated clicks cycle: 🟢 Start → 🔵 Hand → 🟠 Feet → 🟣 Finish"
x ERROR: similar board img not showing
x like/dislike the generated route -> store all generated routes in global database that we can append every session (not refresh/re-init every session), also show  how many climbs have been generated so far (totally)
x larger text on top of generared img including Angle | Grade (lettergrade, vgrade)
x inference animation works, but need to modify generate and generate_with_constraints function to accept animation=True(default as False) to display top_k tokens as we generate them
x make preset examples (3 of them) align horizontally in the same line
x add animation of logits/generation process e.g. with constrained holds (if any), show next possible holds/tokens (maybe 5) 
all circles colored black, pause for 0.5 sec, thne model picks next hold (maybe from top_p top_k or whatever), change the color to its role (start, finish, hand, feet),
pause for another 0.5 sec, repeat until all holds are shown
"""