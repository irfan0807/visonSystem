#!/bin/bash
# AI Vision Monitoring System - Setup Script
# One-click setup for development and production

set -e

echo "🔧 AI Vision Monitoring System Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher is required${NC}"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi
echo -e "${GREEN}Python version: $PYTHON_VERSION ✓${NC}"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created ✓${NC}"
else
    echo -e "${GREEN}Virtual environment already exists ✓${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check for PyAudio dependencies (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "\n${YELLOW}Checking PyAudio dependencies (Linux)...${NC}"
    if ! dpkg -s portaudio19-dev > /dev/null 2>&1; then
        echo -e "${YELLOW}Installing portaudio19-dev...${NC}"
        sudo apt-get update && sudo apt-get install -y portaudio19-dev
    fi
fi

# Create directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p data/{events,clips,logs}
mkdir -p models
mkdir -p static/{css,js}
echo -e "${GREEN}Directories created ✓${NC}"

# Create default config if not exists
if [ ! -f "config.yaml" ]; then
    echo -e "\n${YELLOW}Creating default configuration...${NC}"
    cat > config.yaml << 'EOF'
# AI Vision Monitoring System Configuration

app_name: "AI Vision Monitor"
debug: false
demo_mode: true

# Performance
target_fps: 30
max_latency_ms: 150
max_memory_gb: 4.0

# Paths
data_dir: "data"
models_dir: "models"
logs_dir: "data/logs"
clips_dir: "data/clips"
events_dir: "data/events"

# Camera configuration
cameras:
  - id: 0
    name: "Main Camera"
    source: "0"
    width: 1280
    height: 720
    fps: 30
    enabled: true

# Audio configuration
audio:
  sample_rate: 16000
  chunk_size: 1024
  channels: 1
  device_index: null
  vad_mode: 2
  noise_suppression: true

# Alert configuration
alerts:
  email_enabled: false
  email_smtp_server: ""
  email_smtp_port: 587
  email_sender: ""
  email_recipients: []
  sms_enabled: false
  desktop_enabled: true
  sound_enabled: true

# Anomaly detection
anomaly:
  motion_threshold: 0.3
  anomaly_threshold: 0.75
  motion_weight: 0.4
  mask_change_weight: 0.3
  object_change_weight: 0.3
  scream_threshold: 0.85
  glass_break_threshold: 0.80
  gunshot_threshold: 0.90
  door_slam_threshold: 0.75
  speech_threshold: 0.60

# AI configuration
ai:
  openai_api_key: ""  # Set via OPENAI_API_KEY env var
  openai_model: "gpt-4o-mini"
  scene_summary_interval: 30
  sam_model_type: "vit_b"
  use_gpu: true
EOF
    echo -e "${GREEN}Configuration created ✓${NC}"
fi

# Train audio model (optional)
echo -e "\n${YELLOW}Training audio classification model...${NC}"
if [ ! -f "models/audio_classifier.pkl" ]; then
    python audio_train.py --synthetic --samples-per-class 50 --output models/audio_classifier.pkl
    echo -e "${GREEN}Audio model trained ✓${NC}"
else
    echo -e "${GREEN}Audio model already exists ✓${NC}"
fi

# Check GPU availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not configured for GPU"

# Final setup summary
echo -e "\n${GREEN}======================================"
echo "Setup Complete!"
echo "======================================${NC}"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the Streamlit app:"
echo "     streamlit run app.py --server.port 8501"
echo ""
echo "  3. Open in browser:"
echo "     http://localhost:8501"
echo ""
echo "Optional:"
echo "  - Train custom audio model:"
echo "    python audio_train.py --data-dir /path/to/audio/data"
echo ""
echo "  - Download SAM model checkpoint:"
echo "    wget -P models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
echo ""
echo -e "${GREEN}Happy monitoring! 🎥🔊${NC}"
