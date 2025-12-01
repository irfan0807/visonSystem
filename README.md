# AI Vision Monitoring System

🔴 **Real-time AI-powered video and audio monitoring with anomaly detection**

A comprehensive security monitoring solution featuring:
- 🎥 Live video feed with SAM 3 object segmentation
- 🎤 Real-time audio analysis and classification
- 🎯 Motion-based anomaly detection
- 🚨 Multi-channel alerting (Email, SMS, Desktop)
- ✨ AI-powered scene summarization

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [API Reference](#-api-reference)
- [Docker](#-docker)
- [Development](#-development)
- [Testing](#-testing)

## 🎯 Features

### Video Processing
- **Live webcam capture** at configurable FPS (up to 4K @ 30FPS)
- **SAM 3 integration** for real-time object segmentation
- **Low-light enhancement** using CLAHE
- **Multi-camera support** via configuration
- **Auto-reconnect** on camera disconnect

### Audio Processing
- **Real-time microphone capture** with PyAudio
- **Feature extraction**: MFCC, Spectral Centroid, Zero Crossing Rate
- **Sound event classification** (Scream, Glass Break, Gunshot, etc.)
- **Voice Activity Detection** (WebRTC VAD)
- **Noise suppression**

### Anomaly Detection
```python
# Anomaly score formula
anomaly_score = motion_magnitude * (1 - iou_prev_mask) * object_count_change
```

| Trigger | Threshold |
|---------|-----------|
| High Anomaly | >0.75 |
| Audio Event | >0.85 |
| Person Detected | True |

### Alert System
- 🔔 **Desktop notifications** - Immediate
- 📧 **Email alerts** - HTML reports with clips
- 📱 **SMS alerts** - Twilio integration
- 📊 **Event logging** - Structured JSON logs

### Web UI (Streamlit)
- Live video feed with mask overlay
- Audio waveform visualization
- Anomaly score gauge (0-100)
- Interactive event timeline
- Alert history table
- Dashboard metrics

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/irfan0807/visonSystem.git
cd visonSystem/monitoring_app

# Run setup script
chmod +x setup.sh
./setup.sh

# Launch the application
streamlit run app.py --server.port 8501
```

Open your browser at http://localhost:8501

## 📦 Installation

### Prerequisites

- Python 3.9+
- pip
- (Optional) NVIDIA GPU with CUDA for SAM acceleration

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{events,clips,logs} models

# Train audio model (optional)
python audio_train.py --synthetic

# Run the app
streamlit run app.py
```

### System Dependencies (Linux)

```bash
# For PyAudio
sudo apt-get install portaudio19-dev

# For OpenCV
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

## 🎮 Usage

### Demo Mode

The application starts in demo mode by default, showing simulated data:

```bash
streamlit run app.py
```

### Live Monitoring

1. Connect a webcam
2. Disable "Demo Mode" in the sidebar
3. Click "Start" to begin monitoring

### Command Line Options

```bash
# Custom port
streamlit run app.py --server.port 8080

# Enable debug mode
streamlit run app.py -- --debug

# Headless mode
streamlit run app.py --server.headless true
```

### Training Custom Audio Model

```bash
# Using synthetic data
python audio_train.py --synthetic --samples-per-class 100

# Using custom dataset
python audio_train.py --data-dir /path/to/audio/dataset

# Dataset structure:
# data/audio/
#   ├── normal/
#   │   ├── sample1.wav
#   │   └── ...
#   ├── scream/
#   ├── glass_break/
#   └── ...
```

## ⚙️ Configuration

Configuration is managed via `config.yaml`:

```yaml
# Main settings
app_name: "AI Vision Monitor"
debug: false
demo_mode: true

# Performance
target_fps: 30
max_latency_ms: 150
max_memory_gb: 4.0

# Camera settings
cameras:
  - id: 0
    name: "Main Camera"
    source: "0"  # Device ID or RTSP URL
    width: 1280
    height: 720
    fps: 30

# Audio settings
audio:
  sample_rate: 16000
  vad_mode: 2
  noise_suppression: true

# Anomaly detection thresholds
anomaly:
  motion_threshold: 0.3
  anomaly_threshold: 0.75
  scream_threshold: 0.85
  glass_break_threshold: 0.80

# AI settings
ai:
  openai_model: "gpt-4o-mini"
  scene_summary_interval: 30
  sam_model_type: "vit_b"
  use_gpu: true

# Alerts
alerts:
  email_enabled: false
  sms_enabled: false
  desktop_enabled: true
```

### Environment Variables

Sensitive configuration via environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export TWILIO_SID="your-twilio-sid"
export TWILIO_TOKEN="your-twilio-token"
export SMTP_PASSWORD="your-email-password"
```

## 🏗️ Architecture

```
monitoring_app/
├── app.py                    # Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── video_processor.py    # SAM 3 video segmentation
│   ├── audio_processor.py    # Real-time audio analysis
│   └── anomaly_detector.py   # Motion + segmentation anomaly scoring
├── models/
│   ├── __init__.py
│   ├── audio_classifier.pkl  # Pre-trained audio model
│   └── scene_descriptions.py # AI-powered scene summaries
├── utils/
│   ├── __init__.py
│   ├── alerts.py             # Email/SMS/Desktop notifications
│   ├── logger.py             # Structured logging
│   └── config.py             # YAML configuration
├── static/                   # CSS/JS for UI
├── data/                     # Runtime data
│   ├── events/
│   ├── clips/
│   └── logs/
├── tests/                    # Unit tests
├── requirements.txt
├── audio_train.py            # Audio model training
├── setup.sh                  # Setup script
├── Dockerfile
├── docker-compose.yml
└── config.yaml
```

## 📚 API Reference

### VideoProcessor

```python
from core import VideoProcessor

processor = VideoProcessor(
    target_fps=30,
    enable_sam=True,
    enable_anomaly_detection=True
)

# Start capture
processor.start()

# Get current frame
frame = processor.get_frame()

# Get processing result
result = processor.get_result()
# result.frame, result.detections, result.anomaly_score

# Stop capture
processor.stop()
```

### AudioProcessor

```python
from core import AudioProcessor

processor = AudioProcessor(
    sample_rate=16000,
    noise_suppression=True
)

processor.start()

# Get classification
classification = processor.get_classification()
# classification.label, classification.confidence

# Get waveform for visualization
waveform = processor.get_waveform(duration=2.0)

processor.stop()
```

### AlertManager

```python
from utils import AlertManager

manager = AlertManager(
    email_enabled=True,
    desktop_enabled=True
)

manager.start()

# Trigger alert
manager.trigger_alert(
    alert_type="anomaly",
    message="Motion detected",
    severity="high",
    data={'score': 0.85}
)

# Get history
alerts = manager.get_history(limit=10)

manager.stop()
```

## 🐳 Docker

### Using Docker Compose

```bash
# Build and run
docker-compose up -d

# With environment variables
OPENAI_API_KEY=your-key docker-compose up -d

# Development mode
docker-compose --profile dev up
```

### Using Dockerfile

```bash
# Build
docker build -t ai-vision-monitor .

# Run
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  ai-vision-monitor
```

## 🔧 Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 .
black --check .
mypy .
```

### Code Style

- Follow PEP 8
- Use type hints
- Document with docstrings
- Maximum line length: 100 characters

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_anomaly_detector.py -v

# Run with verbose output
pytest -v --tb=short
```

### Test Coverage Target: 80%+

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| End-to-End Latency | <150ms |
| SAM 3 FPS | >15 FPS (RTX 3060) |
| Memory Usage | <4GB |
| CPU Usage | <30% (with GPU) |

## 🔒 Edge Cases Handled

✅ **Low light** → Auto-exposure + CLAHE enhancement  
✅ **Background noise** → VAD + noise gating  
✅ **Camera disconnect** → Auto-reconnect with retry  
✅ **High CPU** → Frame skipping + priority queue  
✅ **Network issues** → Offline mode + queued alerts  

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/irfan0807/visonSystem/issues)
- 📖 Docs: [Wiki](https://github.com/irfan0807/visonSystem/wiki)