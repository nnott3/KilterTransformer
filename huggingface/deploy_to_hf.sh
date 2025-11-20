#!/bin/bash
# Automated deployment script for KilterGPT to Hugging Face Spaces

set -e  # Exit on error

echo "🚀 KilterGPT Deployment Script"
echo "================================"

# Configuration
HF_USERNAME="nottreepat"
read -p "Enter Space name (default: kilter-gpt-app): " SPACE_NAME
SPACE_NAME=${SPACE_NAME:-kilter-gpt-app}

read -p "Enter commit message for this deployment: " COMMIT_MSG
COMMIT_MSG=${COMMIT_MSG:-"Deploy KilterGPT app"}

SPACE_REPO="https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
TEMP_DIR="./hf_space_temp"

echo ""
echo "📋 Configuration:"
echo "   Username: $HF_USERNAME"
echo "   Space: $SPACE_NAME"
echo "   URL: $SPACE_REPO"
echo ""

# Step 1: Check if logged in
echo "🔐 Step 1: Checking HuggingFace login..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    exit 1
fi
echo "✅ Logged in as: $(huggingface-cli whoami)"

# Step 2: Create Space (if doesn't exist)
echo ""
echo "📦 Step 2: Creating Space..."
huggingface-cli repo create --type space --space_sdk gradio $SPACE_NAME || echo "Space might already exist, continuing..."

# Step 3: Clone Space
echo ""
echo "📥 Step 3: Cloning Space repository..."
rm -rf $TEMP_DIR
git clone $SPACE_REPO $TEMP_DIR
cd $TEMP_DIR

# Step 4: Copy files
echo ""
echo "📂 Step 4: Copying files..."
mkdir -p src figs

echo "   Copying app.py..."
cp ../huggingface/app.py . 2>/dev/null || echo "⚠️  app.py not found in parent directory"

echo "   Copying source files..."
cp ../src/data_processing.py src/ 2>/dev/null || echo "⚠️  data_processing.py not found"
cp ../src/temp_gpt.py src/ 2>/dev/null || echo "⚠️  temp_gpt.py not found"
cp ../src/visualization.py src/ 2>/dev/null || echo "⚠️  visualization.py not found"
cp ../src/tokenizer.py src/ 2>/dev/null || echo "⚠️  tokenizer.py not found (optional)"

echo "   Creating src/__init__.py..."
touch src/__init__.py


# Step 6: README.md
echo ""
echo "📄 Step 6: Creating README.md..."
cat > README.md << EOF
---
title: KilterGPT Route Generator
emoji: 🧗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# KilterGPT - AI Climbing Route Generator 🧗

Generate custom climbing routes for the Kilter Board using AI!

## Features
- 🎯 Generate routes based on angle, grade, and number of holds
- 🎨 Visual route representation on the Kilter Board
- 🔧 Adjustable generation parameters
- 💾 Specify required starting holds
- 🔒 Structural constraints for realistic routes

## Model
- Architecture: Custom GPT-2 (6 layers, 4 heads, 256 dim)
- Training: 70,000+ real Kilter Board routes
- Custom order-invariant loss function

---
Deployed by: $HF_USERNAME
EOF

# Step 7: Git operations
echo ""
echo "📤 Step 7: Committing and pushing..."
git add .
git commit -m "$COMMIT_MSG" || echo "No changes to commit"
git push

cd ..
rm -rf $TEMP_DIR

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your app should be live in 2-5 minutes at:"
echo "   $SPACE_REPO"
echo ""
echo "📊 Monitor build progress:"
echo "   https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME/logs"
echo ""
