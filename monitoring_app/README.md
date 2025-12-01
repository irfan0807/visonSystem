# Real-Time Monitoring App

A React Native/Expo application for real-time monitoring with SAM3 segmentation, anomaly detection, audio classification, and AI-powered summaries.

## Features

- 📹 **Camera Capture**: Real-time video capture at 30 FPS with <150ms latency
- 🎯 **SAM3 Segmentation**: Integration with irfan0807/sam3Vision for real-time object segmentation
- 🔊 **Audio Analysis**: Audio classification and anomaly detection
- 🚨 **Alert System**: Push notifications and email alerts
- 🤖 **AI Summaries**: OpenAI-powered event summaries
- 📱 **Multi-Camera**: Support for multiple camera views
- 📊 **Performance Monitoring**: Real-time FPS and latency tracking

## Project Structure

```
monitoring_app/
├── App.tsx                     # Main application entry point
├── app.json                    # Expo configuration
├── package.json                # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
├── babel.config.js            # Babel configuration
├── src/
│   ├── components/            # UI Components
│   │   ├── CameraView.tsx     # Camera preview with controls
│   │   ├── MaskOverlay.tsx    # Segmentation mask overlay
│   │   ├── AlertList.tsx      # Alert list with acknowledgment
│   │   ├── StatusBar.tsx      # Status and metrics display
│   │   ├── SummaryCard.tsx    # AI summary display
│   │   └── index.ts
│   ├── services/              # Core Services
│   │   ├── VideoProcessor.ts  # Video capture and frame processing
│   │   ├── AudioAnalyzer.ts   # Audio capture and classification
│   │   ├── AlertManager.ts    # Push/email notification management
│   │   ├── SAM3Integration.ts # SAM3 API integration
│   │   ├── AnomalyDetector.ts # Visual/motion anomaly detection
│   │   ├── OpenAISummary.ts   # OpenAI summary generation
│   │   └── index.ts
│   ├── hooks/                 # Custom React Hooks
│   │   ├── useMonitoring.ts   # Main monitoring state management
│   │   ├── useCamera.ts       # Camera permission and control
│   │   ├── useAudio.ts        # Audio recording management
│   │   └── index.ts
│   ├── screens/               # App Screens
│   │   ├── MonitoringScreen.tsx
│   │   └── index.ts
│   ├── types/                 # TypeScript Definitions
│   │   └── index.ts
│   ├── config/                # App Configuration
│   │   └── index.ts
│   └── utils/                 # Utility Functions
│       └── index.ts
```

## Installation

```bash
# Navigate to monitoring app directory
cd monitoring_app

# Install dependencies
npm install

# Start Expo development server
npm start
```

## Running the App

```bash
# iOS Simulator
npm run ios

# Android Emulator
npm run android

# Web Browser
npm run web
```

## Configuration

### Environment Variables

Create a `.env` file in the monitoring_app directory:

```env
EXPO_PUBLIC_API_BASE_URL=https://api.sam3vision.com
EXPO_PUBLIC_SAM3_API_KEY=your_sam3_api_key
EXPO_PUBLIC_OPENAI_API_KEY=your_openai_api_key
```

### Video Configuration

Edit `src/config/index.ts` to adjust video settings:

```typescript
export const VIDEO_CONFIG = {
  targetFPS: 30,           // Target frame rate
  maxLatencyMs: 150,       // Maximum acceptable latency
  multiCameraEnabled: true, // Enable multi-camera support
  maxCameras: 4,           // Maximum number of cameras
};
```

### Alert Configuration

```typescript
export const ALERT_CONFIG = {
  pushEnabled: true,
  emailEnabled: true,
  emailRecipients: ['alerts@example.com'],
  severityThreshold: 'medium',
  cooldownMs: 60000,
};
```

## Services

### VideoProcessor

Handles camera capture and frame processing with 30 FPS target:

```typescript
import { getVideoProcessor } from './src/services';

const videoProcessor = getVideoProcessor();
await videoProcessor.initialize(cameras);
videoProcessor.startProcessing();

videoProcessor.onFrame((frame) => {
  console.log('Frame captured:', frame.id);
});
```

### AudioAnalyzer

Processes audio for classification and anomaly detection:

```typescript
import { getAudioAnalyzer } from './src/services';

const audioAnalyzer = getAudioAnalyzer();
await audioAnalyzer.initialize();
audioAnalyzer.startAnalysis();

audioAnalyzer.onAnomaly((anomaly) => {
  console.log('Audio anomaly:', anomaly.description);
});
```

### AlertManager

Manages push notifications and email alerts:

```typescript
import { getAlertManager } from './src/services';

const alertManager = getAlertManager();
await alertManager.initialize();
await alertManager.registerForPushNotifications();

await alertManager.createAlert(
  'Motion Detected',
  'Movement detected in Zone A',
  'high',
  'visual'
);
```

### SAM3Integration

Integrates with SAM3 Vision API for segmentation:

```typescript
import { getSAM3Integration } from './src/services';

const sam3 = getSAM3Integration();
await sam3.initialize();

const segmentation = await sam3.segmentFrame(videoFrame);
console.log('Masks detected:', segmentation.masks.length);
```

### AnomalyDetector

Detects visual and motion anomalies:

```typescript
import { getAnomalyDetector } from './src/services';

const detector = getAnomalyDetector();
await detector.initialize();
detector.startDetection();

detector.onAnomaly((anomaly) => {
  console.log('Anomaly detected:', anomaly.type, anomaly.severity);
});
```

### OpenAISummary

Generates AI-powered event summaries:

```typescript
import { getOpenAISummary } from './src/services';

const summary = getOpenAISummary();
await summary.initialize();
summary.startAutoSummary();

summary.onSummary((eventSummary) => {
  console.log('Summary:', eventSummary.summary);
});
```

## Custom Hooks

### useMonitoring

Main hook for coordinating all monitoring services:

```typescript
import { useMonitoring } from './src/hooks';

function MonitoringComponent() {
  const {
    state,
    startMonitoring,
    stopMonitoring,
    acknowledgeAlert,
    generateSummary,
  } = useMonitoring();

  return (
    <View>
      <Text>FPS: {state.performanceMetrics.fps}</Text>
      <Text>Alerts: {state.alerts.length}</Text>
    </View>
  );
}
```

### useCamera

Camera permission and control hook:

```typescript
import { useCamera } from './src/hooks';

function CameraComponent() {
  const {
    hasPermission,
    currentCamera,
    requestPermission,
    switchCamera,
    startCapture,
  } = useCamera();

  // ...
}
```

### useAudio

Audio recording management hook:

```typescript
import { useAudio } from './src/hooks';

function AudioComponent() {
  const {
    hasPermission,
    isRecording,
    audioLevel,
    startRecording,
    stopRecording,
  } = useAudio();

  // ...
}
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Frame Rate | 30 FPS |
| Latency | <150ms |
| Frame Drops | <5% |
| Memory Usage | <512MB |

## Anomaly Detection Types

- `motion_detected` - Movement detected in frame
- `unusual_activity` - Unexpected behavior patterns
- `object_detected` - Specific objects (e.g., weapons, packages)
- `audio_anomaly` - Unusual audio patterns
- `crowd_formation` - Multiple people detected
- `restricted_area_breach` - Zone violation
- `equipment_malfunction` - System issues

## Alert Severity Levels

- `low` - Informational alerts
- `medium` - Attention required
- `high` - Immediate action needed
- `critical` - Emergency response required

## Dependencies

- `expo` - React Native framework
- `expo-camera` - Camera access
- `expo-av` - Audio recording
- `expo-notifications` - Push notifications
- `react-native-vision-camera` - Advanced camera features
- `react-native-svg` - Mask overlay rendering
- `zustand` - State management (optional)

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
