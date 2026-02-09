/**
 * Type definitions for the Vision System Server
 */

// ─── Conversation Types ───────────────────────────────────────────────
export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  type: 'text' | 'audio' | 'vision' | 'multimodal';
  metadata?: MessageMetadata;
}

export interface MessageMetadata {
  audioUrl?: string;
  imageUrl?: string;
  visionAnalysis?: VisionAnalysis;
  audioTranscription?: string;
  emotionDetected?: string;
  processingTimeMs?: number;
}

export interface Conversation {
  id: string;
  messages: ConversationMessage[];
  createdAt: number;
  updatedAt: number;
  systemPrompt: string;
}

// ─── Vision Types ─────────────────────────────────────────────────────
export interface VisionAnalysis {
  description: string;
  objects: DetectedObject[];
  emotions?: EmotionResult[];
  scene: string;
  anomalies: string[];
  timestamp: number;
}

export interface DetectedObject {
  label: string;
  confidence: number;
  boundingBox?: { x: number; y: number; w: number; h: number };
}

export interface EmotionResult {
  emotion: string;
  confidence: number;
}

export interface FrameData {
  imageBase64: string;
  timestamp: number;
  width: number;
  height: number;
}

// ─── Audio Types ──────────────────────────────────────────────────────
export interface AudioData {
  buffer: Buffer;
  format: 'webm' | 'wav' | 'mp3';
  duration: number;
  sampleRate: number;
}

export interface TranscriptionResult {
  text: string;
  language: string;
  duration: number;
  segments?: TranscriptionSegment[];
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
}

export interface TTSResult {
  audioBuffer: Buffer;
  format: string;
  duration: number;
}

// ─── Neural Network Types ─────────────────────────────────────────────
export interface NeuralNetworkConfig {
  gptModel: string;
  whisperModel: string;
  ttsModel: string;
  ttsVoice: string;
  visionAnalysisInterval: number;
  maxFrameHistory: number;
}

export interface ProcessingPipeline {
  audio: {
    transcribe: (audio: AudioData) => Promise<TranscriptionResult>;
    synthesize: (text: string) => Promise<TTSResult>;
  };
  vision: {
    analyze: (frame: FrameData) => Promise<VisionAnalysis>;
    describeScene: (frames: FrameData[]) => Promise<string>;
  };
  language: {
    chat: (messages: ConversationMessage[], context?: VisionAnalysis) => Promise<string>;
    summarize: (text: string) => Promise<string>;
  };
}

// ─── WebSocket Events ─────────────────────────────────────────────────
export interface ClientToServerEvents {
  'audio:stream': (data: { audio: string; format: string }) => void;
  'audio:stop': () => void;
  'video:frame': (data: { frame: string; timestamp: number }) => void;
  'chat:message': (data: { text: string; includeVision: boolean }) => void;
  'vision:analyze': (data: { frame: string }) => void;
  'system:configure': (data: Partial<NeuralNetworkConfig>) => void;
}

export interface ServerToClientEvents {
  'audio:response': (data: { audio: string; text: string; format: string }) => void;
  'chat:response': (data: { text: string; id: string }) => void;
  'chat:stream': (data: { chunk: string; id: string; done: boolean }) => void;
  'vision:analysis': (data: VisionAnalysis) => void;
  'vision:description': (data: { text: string }) => void;
  'system:status': (data: SystemStatus) => void;
  'error': (data: { message: string; code: string }) => void;
  'emotion:detected': (data: EmotionResult) => void;
}

export interface SystemStatus {
  connected: boolean;
  modelsLoaded: boolean;
  activeStreams: number;
  lastActivity: number;
  visionActive: boolean;
  audioActive: boolean;
  fps: number;
  latencyMs: number;
}
