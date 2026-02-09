# IRIS — Intelligent Real-time Interactive System

> A real-time neural network-powered AI system you can **talk to like a real person**. It sees through your camera, hears your voice, and responds naturally using GPT-4 Vision, Whisper, and Neural TTS.

![Architecture](https://img.shields.io/badge/Architecture-React%20%2B%20Express%20%2B%20WebSocket-blue)
![AI](https://img.shields.io/badge/AI-GPT--4%20Vision%20%7C%20Whisper%20%7C%20TTS-purple)

---

## 🧠 How It Works

```
┌─────────────────┐     WebSocket      ┌──────────────────┐     API     ┌──────────────┐
│   React Client   │ ◄──────────────► │  Express Server   │ ◄────────► │   OpenAI API  │
│                  │                   │                   │            │              │
│  📷 Webcam       │  video frames     │  🧠 Neural Net    │  GPT-4V    │  Vision      │
│  🎤 Microphone   │  audio chunks     │     Service       │  Whisper   │  Speech      │
│  💬 Chat UI      │  text messages    │  📡 WebSocket     │  TTS       │  Language    │
│  👁 Vision Panel │  ◄── responses    │     Handler       │            │              │
│  🔊 TTS Playback │                   │  🌐 REST API      │            │              │
└─────────────────┘                   └──────────────────┘            └──────────────┘
```

### Real-Time Multimodal Pipeline
1. **You speak** → Mic captures audio → Whisper transcribes → GPT-4 understands
2. **Camera sees** → Frames sent to server → GPT-4 Vision analyzes → Objects, emotions, scene detected
3. **AI responds** → GPT-4 generates natural response with visual awareness → TTS speaks it back
4. **All in real-time** → WebSocket keeps everything instant and bidirectional

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ 
- **OpenAI API Key** with access to GPT-4 Vision, Whisper, and TTS
- Modern browser (Chrome/Edge recommended for WebRTC)

### 1. Clone & Install

```bash
# Backend
cd web-app/server
npm install
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Frontend
cd ../client
npm install
```

### 2. Configure

Edit `web-app/server/.env`:
```env
OPENAI_API_KEY=sk-your-key-here
PORT=3001
CORS_ORIGIN=http://localhost:5173
GPT_MODEL=gpt-4o
TTS_VOICE=nova
```

### 3. Run

```bash
# Terminal 1 — Backend
cd web-app/server
npm run dev

# Terminal 2 — Frontend
cd web-app/client
npm run dev
```

Open **http://localhost:5173** — allow camera and microphone access.

---

## 🎯 Features

### 🗣 Voice Conversation
- **Push-to-talk** microphone — click to record, click to send
- **Whisper STT** — accurate speech-to-text transcription
- **Neural TTS** — natural-sounding voice responses (choice of voices)
- Feels like talking to a real person

### 👁 Visual Awareness
- **Live webcam feed** with real-time analysis
- **GPT-4 Vision** analyzes scenes, detects objects, reads emotions
- **Vision overlay** — see what IRIS detects directly on the video
- **Anomaly detection** — spots unusual activity
- Proactive comments about what it sees

### 💬 Text Chat
- **Streaming responses** — see text appear word by word
- Full conversation history
- Vision-aware responses — IRIS references what it sees

### 🧠 Neural Network Integration
- **GPT-4o** — multimodal reasoning (text + vision)
- **Whisper** — speech recognition
- **TTS-1** — speech synthesis
- **Real-time WebSocket** — low-latency bidirectional communication

---

## 📁 Project Structure

```
web-app/
├── server/                     # Express + WebSocket backend
│   ├── src/
│   │   ├── index.ts            # Server entry point
│   │   ├── types.ts            # TypeScript type definitions
│   │   ├── services/
│   │   │   └── NeuralNetworkService.ts  # GPT-4V + Whisper + TTS
│   │   ├── websocket/
│   │   │   └── socketHandler.ts         # Real-time WebSocket events
│   │   └── routes/
│   │       └── api.ts                   # REST API endpoints
│   ├── package.json
│   └── .env.example
│
├── client/                     # React + Vite frontend
│   ├── src/
│   │   ├── App.tsx             # Main application
│   │   ├── components/
│   │   │   ├── IrisAvatar.tsx      # Animated AI presence
│   │   │   ├── VideoFeed.tsx       # Webcam + vision overlay
│   │   │   ├── ChatPanel.tsx       # Conversation UI
│   │   │   ├── VoiceControl.tsx    # Push-to-talk controls
│   │   │   ├── VisionPanel.tsx     # Vision analysis display
│   │   │   └── StatusIndicator.tsx # Connection status
│   │   └── hooks/
│   │       ├── useSocket.ts        # WebSocket communication
│   │       ├── useWebcam.ts        # Camera capture
│   │       ├── useMicrophone.ts    # Audio recording
│   │       └── useAudioPlayer.ts   # TTS playback
│   ├── package.json
│   └── tailwind.config.js
│
└── monitoring_app/             # Original React Native monitoring app
```

---

## 🔌 API Reference

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `audio:stream` | Client → Server | Send recorded audio for processing |
| `audio:response` | Server → Client | TTS audio + text response |
| `video:frame` | Client → Server | Send webcam frame |
| `vision:analyze` | Client → Server | Request vision analysis |
| `vision:analysis` | Server → Client | Vision analysis results |
| `chat:message` | Client → Server | Send text message |
| `chat:stream` | Server → Client | Streaming text response |
| `system:status` | Server → Client | System health status |

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server health check |
| POST | `/api/chat` | Send chat message |
| POST | `/api/interact` | Full multimodal interaction |
| POST | `/api/audio/transcribe` | Transcribe audio file |
| POST | `/api/audio/synthesize` | Text to speech |
| POST | `/api/vision/analyze` | Analyze image frame |
| GET | `/api/vision/describe` | Describe current scene |
| GET | `/api/conversation` | Get conversation history |

---

## ⚙️ Configuration

### TTS Voices
Available voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

Set in `.env`:
```env
TTS_VOICE=nova
```

### GPT Models
- `gpt-4o` — Best quality, multimodal (recommended)
- `gpt-4o-mini` — Faster, cheaper, still multimodal
- `gpt-4-turbo` — Alternative

---

## 📄 License

MIT
