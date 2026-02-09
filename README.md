# visonSystem

## IRIS — Intelligent Real-time Interactive System

A real-time AI monitoring and conversation system with neural network integration. Talk to IRIS like a real person — it sees through your camera, hears your voice, and responds naturally.

### Projects

- **`web-app/`** — React web frontend + Express backend with GPT-4 Vision, Whisper STT, and Neural TTS  
- **`monitoring_app/`** — Original React Native mobile monitoring app with SAM3 segmentation and anomaly detection

### Quick Start

```bash
# Backend
cd web-app/server && npm install && cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Frontend  
cd web-app/client && npm install

# Run both
cd web-app/server && npm run dev   # Terminal 1
cd web-app/client && npm run dev   # Terminal 2
```

Open **http://localhost:5173** and allow camera/mic access.

See [web-app/README.md](web-app/README.md) for full documentation.