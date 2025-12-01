/**
 * Application Configuration
 * Real-Time Monitoring App
 */

import type {
  SAM3Config,
  AudioConfig,
  AnomalyConfig,
  AlertConfig,
  SummaryConfig,
  Resolution,
} from '../types';

// Environment-based API endpoints
const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || 'https://api.sam3vision.com';
const SAM3_API_KEY = process.env.EXPO_PUBLIC_SAM3_API_KEY || '';
const OPENAI_API_KEY = process.env.EXPO_PUBLIC_OPENAI_API_KEY || '';

/**
 * Video/Camera Configuration
 */
export const VIDEO_CONFIG = {
  // Target 30 FPS for real-time monitoring
  targetFPS: 30,
  // Maximum latency threshold in milliseconds
  maxLatencyMs: 150,
  // Default resolution settings
  defaultResolution: {
    width: 1920,
    height: 1080,
  } as Resolution,
  // Low resolution for processing
  processingResolution: {
    width: 640,
    height: 480,
  } as Resolution,
  // Frame buffer size
  frameBufferSize: 10,
  // JPEG quality for frame encoding (0-1)
  jpegQuality: 0.8,
  // Enable multi-camera support
  multiCameraEnabled: true,
  // Maximum number of cameras
  maxCameras: 4,
};

/**
 * SAM3 Segmentation API Configuration
 */
export const SAM3_CONFIG: SAM3Config = {
  apiEndpoint: `${API_BASE_URL}/v1/segment`,
  apiKey: SAM3_API_KEY,
  modelVersion: 'sam3-large',
  confidenceThreshold: 0.5,
  maxMasks: 20,
  timeout: 5000, // 5 second timeout
};

/**
 * Audio Analysis Configuration
 */
export const AUDIO_CONFIG: AudioConfig = {
  sampleRate: 44100,
  channels: 1, // Mono for analysis
  encoding: 'pcm_16bit',
  bufferSize: 4096,
  enabled: true,
};

/**
 * Anomaly Detection Configuration
 */
export const ANOMALY_CONFIG: AnomalyConfig = {
  visualThreshold: 0.7,
  audioThreshold: 0.6,
  motionThreshold: 0.5,
  alertCooldownMs: 30000, // 30 seconds between alerts
  enableVisualDetection: true,
  enableAudioDetection: true,
  enableMotionDetection: true,
};

/**
 * Alert Configuration
 */
export const ALERT_CONFIG: AlertConfig = {
  pushEnabled: true,
  emailEnabled: true,
  emailRecipients: [],
  smsEnabled: false,
  smsRecipients: [],
  severityThreshold: 'medium',
  cooldownMs: 60000, // 1 minute cooldown
};

/**
 * OpenAI Summary Configuration
 */
export const SUMMARY_CONFIG: SummaryConfig = {
  apiKey: OPENAI_API_KEY,
  model: 'gpt-4-turbo-preview',
  maxTokens: 500,
  temperature: 0.7,
  summaryInterval: 300000, // 5 minutes
  enabled: true,
};

/**
 * Performance Thresholds
 */
export const PERFORMANCE_THRESHOLDS = {
  minFPS: 24,
  maxLatency: 150,
  maxFrameDropPercent: 5,
  maxMemoryMB: 512,
  maxCPUPercent: 80,
};

/**
 * UI Configuration
 */
export const UI_CONFIG = {
  // Mask overlay colors by confidence level
  maskColors: {
    high: 'rgba(0, 255, 0, 0.4)',    // Green for high confidence
    medium: 'rgba(255, 255, 0, 0.4)', // Yellow for medium
    low: 'rgba(255, 0, 0, 0.4)',      // Red for low
  },
  // Alert colors by severity
  alertColors: {
    low: '#4CAF50',
    medium: '#FF9800',
    high: '#F44336',
    critical: '#9C27B0',
  },
  // Animation durations
  animationDuration: 300,
  // Toast duration
  toastDuration: 3000,
};

/**
 * Storage Keys
 */
export const STORAGE_KEYS = {
  CAMERAS: '@monitoring/cameras',
  ALERTS: '@monitoring/alerts',
  SETTINGS: '@monitoring/settings',
  PUSH_TOKEN: '@monitoring/push_token',
  LAST_SUMMARY: '@monitoring/last_summary',
};

/**
 * API Endpoints
 */
export const API_ENDPOINTS = {
  SAM3_SEGMENT: `${API_BASE_URL}/v1/segment`,
  SAM3_BATCH_SEGMENT: `${API_BASE_URL}/v1/segment/batch`,
  ANOMALY_DETECT: `${API_BASE_URL}/v1/anomaly/detect`,
  AUDIO_CLASSIFY: `${API_BASE_URL}/v1/audio/classify`,
  ALERTS_SEND: `${API_BASE_URL}/v1/alerts/send`,
  ALERTS_EMAIL: `${API_BASE_URL}/v1/alerts/email`,
};

export default {
  VIDEO_CONFIG,
  SAM3_CONFIG,
  AUDIO_CONFIG,
  ANOMALY_CONFIG,
  ALERT_CONFIG,
  SUMMARY_CONFIG,
  PERFORMANCE_THRESHOLDS,
  UI_CONFIG,
  STORAGE_KEYS,
  API_ENDPOINTS,
};
