#!/bin/bash

# Vast.ai Parakeet ASR Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🚀 Starting Parakeet ASR deployment on Vast.ai..."

# Update system
apt update -y
apt install -y python3 python3-pip python3-venv python3-dev build-essential git wget curl htop nano tmux nvidia-cuda-toolkit

# Create project directory
PROJECT_DIR="/opt/parakeet-asr"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Create virtual environment
python3 -m venv parakeet-env
source parakeet-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install nemo_toolkit[asr] websockets silero-vad aiohttp transformers numpy scipy librosa soundfile jiwer

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core dependencies
torch>=2.0.0
nemo_toolkit[asr]>=1.20.0
websockets>=11.0.0
silero-vad>=4.0.0
aiohttp>=3.8.0
transformers>=4.30.0
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.9.0
soundfile>=0.12.0
jiwer>=3.0.0
python-dotenv>=1.0.0
EOF

# Install from requirements.txt
pip install -r requirements.txt

# Download the actual server file from your Git repository
print_status "Downloading Parakeet WebSocket server from Git repository..."

# Set your repository URL here
REPO_URL="https://raw.githubusercontent.com/spanDevOps/Parakeet/main/parakeet_websocket_server.py"

# Download the server file
print_status "Downloading parakeet_websocket_server.py from: $REPO_URL"
wget -O parakeet_websocket_server.py "$REPO_URL"

# Verify the file was downloaded
if [ -f "parakeet_websocket_server.py" ]; then
    print_success "✅ Server file downloaded successfully!"
    print_status "📊 File size: $(wc -l < parakeet_websocket_server.py) lines"
else
    print_error "❌ Failed to download server file"
    print_status "📁 Please check your repository URL and try again"
    exit 1
fi

chmod +x parakeet_websocket_server.py

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import nemo.collections.asr as nemo_asr
import websockets
import numpy as np
print('✅ All imports successful')
print(f'🚀 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎯 GPU: {torch.cuda.get_device_name()}')
"

# Start server
echo "🚀 Starting Parakeet ASR server..."
echo "🌐 WebSocket: ws://0.0.0.0:8080"
echo "📊 Monitor with: htop"
echo "🔄 Restart with: cd $PROJECT_DIR && source parakeet-env/bin/activate && python parakeet_websocket_server.py"

# Start in background
cd $PROJECT_DIR
source parakeet-env/bin/activate
nohup python parakeet_websocket_server.py > /var/log/parakeet-asr.log 2>&1 &

echo "🎉 Parakeet ASR server is running!"
echo "📝 Check logs: tail -f /var/log/parakeet-asr.log"
echo "🔍 Monitor: htop"