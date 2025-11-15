# KilterTransformer

GPT-Architecture Decoder transformers for climbing route generations on the Kilter Board.

## Overview

All data are fetched from boardlib library. Standardized boards are chosen because of their ease of hold encoding, without image processing burden. Though, that could be another interesting idea. Kilter board is chosen for its popularity and availability of data. Solution could work with moonboard or tensionboard with minor tweaks since no spatial encoding is done.

<div align="center">
  <img src="figs/plot_boulder_prediction.png" alt="Kilter Board Route Example" width="400"/>
  
</div>


**Dataset:** ~70,000 cleaned boulders from the `boardlib` library.

**Custom GPT Implementation:**
- Lightweight GPT2 architecture with Linear Modeling head  (~7M parameters) built using PyTorch and HuggingFace
- Custom tokenizer and vocabulary dict (~2000 tokens)
- Trained with order-invariant loss and data augmentation, specific to climbing domain
- Host model and datasets on huggingface's spaces, interface built with `Gradio`
   

**Custom BERT Implementation:**
- 4 layers, 8 attention heads (~1M parameters) built using PyTorch and HuggingFace's `BertConfig`
- Custom tokenizer and vocabulary dict (~2000 tokens)
    - Hold encoding: `hold_id, hand_or_foot` (binary: foot vs hand/start/finish)
- 2D positional embeddings for spatial awareness
- Metadata integration (angle, density, reach)

**Performance:**
<p align="center">
  <table>
    <tr><th>Model</th><th>RMSE</th><th>±1 V-grade Accuracy</th></tr>
    <tr><td>XGBoost</td><td>1.9</td><td>76%</td></tr>
    <tr><td><b>BERT Encoder</b></td><td><b>1.6</b></td><td><b>82%</b></td></tr>
  </table>
</p>



<div align="center">
    <img src="figs/plot_prediction_good_ass.png" alt="Predictions" width="400"/>
</div>

*"That last 6%, it doesn't sound like a lot, but it's tremendous"* - Gale Boetticher, Breaking Bad (probably)

*"82% will do just fine"* - Gus Fring (probably


## Project Structure
```
project-root/
├── src/
│   ├── data_processing.py      
│   ├── evaluation.py           
│   ├── features_eng.py         
│   ├── visualizations.py       # for all plots
│   ├── bert_ass.py             # BERT with tokenizer and model from scratch
│   └── temp_gpt.py             # KilterGPT class and tokenizer modules
├── data/
│   ├── climbs_cleaned.csv      # Raw data
├── notebooks/
│   ├── EDA.ipynb               # data processing, simple modeling
│   ├── bert.ipynb              # BERT for difficulty prediction
│   ├── bert_improved.ipynb     # BERT but more structured
│   ├── gpt_new.ipynb (*)       # Training notebook for final GPT
│   └── gpt_shuffle_me.ipynb    # dratf GPT 
├── saved_models/               # Checkpoints for predictions models
├── models/                     # Checkpoints for GPT models
├── hugginface/
│   ├── app.py                  
│   ├── create_model_config.py
│   ├── create_model_config.py
│   ├── upload_model_to_hf.py
│   ├── deploy_to_hf.sh         # syncing this repo with huggingface/spaces repo
│   └── fonts/
├── figs/                    
```

## Installation
```bash
# Clone repository
git clone https://github.com/nnott3/KilterTransformer.git
cd KilterTransformer

# Install dependencies
uv sync

jupyter notebook notebooks/EDA.ipynb
# run first block to read sqlite3 
# explore all the functionalities
jupyter notebook notebooks/bert.ipynb
jupyter notebook notebooks/gpt_new.ipynb
```

TODO
- [x] Add similarity search
- [ ] Make Clustering/Generation based on style+grade (powerful, crimpy, shouldery, technical)
- [x] Route Generation
 