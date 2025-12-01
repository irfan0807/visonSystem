/**
 * Type definitions for the Real-Time Monitoring App
 */

// Camera and Video Types
export interface CameraConfig {
  id: string;
  name: string;
  position: 'front' | 'back';
  resolution: Resolution;
  frameRate: number;
  enabled: boolean;
}

export interface Resolution {
  width: number;
  height: number;
}

export interface VideoFrame {
  id: string;
  timestamp: number;
  data: string; // Base64 encoded
  width: number;
  height: number;
  cameraId: string;
}

export interface ProcessedFrame extends VideoFrame {
  processingTime: number;
  segmentationMask?: SegmentationMask;
  anomalies?: AnomalyResult[];
}

// SAM3 Segmentation Types
export interface SegmentationMask {
  id: string;
  frameId: string;
  masks: MaskData[];
  timestamp: number;
  processingTimeMs: number;
}

export interface MaskData {
  id: string;
  label: string;
  confidence: number;
  boundingBox: BoundingBox;
  polygon: Point[];
  area: number;
  color: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface SAM3Config {
  apiEndpoint: string;
  apiKey: string;
  modelVersion: string;
  confidenceThreshold: number;
  maxMasks: number;
  timeout: number;
}

export interface SAM3Response {
  success: boolean;
  masks: MaskData[];
  processingTimeMs: number;
  error?: string;
}

// Audio Analysis Types
export interface AudioConfig {
  sampleRate: number;
  channels: number;
  encoding: 'pcm_16bit' | 'pcm_float';
  bufferSize: number;
  enabled: boolean;
}

export interface AudioFrame {
  id: string;
  timestamp: number;
  data: Float32Array | Int16Array;
  sampleRate: number;
  duration: number;
}

export interface AudioClassification {
  id: string;
  frameId: string;
  timestamp: number;
  classifications: ClassificationResult[];
  anomalyScore: number;
  isAnomaly: boolean;
}

export interface ClassificationResult {
  label: string;
  confidence: number;
  category: AudioCategory;
}

export type AudioCategory = 
  | 'speech'
  | 'music'
  | 'environmental'
  | 'mechanical'
  | 'alarm'
  | 'animal'
  | 'unknown';

// Anomaly Detection Types
export interface AnomalyConfig {
  visualThreshold: number;
  audioThreshold: number;
  motionThreshold: number;
  alertCooldownMs: number;
  enableVisualDetection: boolean;
  enableAudioDetection: boolean;
  enableMotionDetection: boolean;
}

export interface AnomalyResult {
  id: string;
  timestamp: number;
  type: AnomalyType;
  severity: AnomalySeverity;
  confidence: number;
  description: string;
  source: 'visual' | 'audio' | 'motion';
  metadata: Record<string, unknown>;
}

export type AnomalyType = 
  | 'motion_detected'
  | 'unusual_activity'
  | 'object_detected'
  | 'audio_anomaly'
  | 'crowd_formation'
  | 'restricted_area_breach'
  | 'equipment_malfunction';

export type AnomalySeverity = 'low' | 'medium' | 'high' | 'critical';

// Alert Types
export interface Alert {
  id: string;
  timestamp: number;
  type: AlertType;
  severity: AnomalySeverity;
  title: string;
  message: string;
  source: string;
  acknowledged: boolean;
  metadata: AlertMetadata;
}

export type AlertType = 
  | 'push_notification'
  | 'email'
  | 'sms'
  | 'in_app';

export interface AlertMetadata {
  anomalyId?: string;
  frameId?: string;
  cameraId?: string;
  imageUrl?: string;
  audioClipUrl?: string;
  location?: string;
}

export interface AlertConfig {
  pushEnabled: boolean;
  emailEnabled: boolean;
  emailRecipients: string[];
  smsEnabled: boolean;
  smsRecipients: string[];
  severityThreshold: AnomalySeverity;
  cooldownMs: number;
}

export interface PushNotificationConfig {
  expoPushToken?: string;
  sound: boolean;
  badge: boolean;
  vibrate: boolean;
}

// OpenAI Summary Types
export interface SummaryConfig {
  apiKey: string;
  model: string;
  maxTokens: number;
  temperature: number;
  summaryInterval: number;
  enabled: boolean;
}

export interface EventSummary {
  id: string;
  timestamp: number;
  timeRange: TimeRange;
  summary: string;
  keyEvents: KeyEvent[];
  alertCount: number;
  anomalyCount: number;
  recommendations: string[];
}

export interface TimeRange {
  start: number;
  end: number;
}

export interface KeyEvent {
  timestamp: number;
  type: string;
  description: string;
  severity: AnomalySeverity;
}

// App State Types
export interface MonitoringState {
  isMonitoring: boolean;
  cameras: CameraConfig[];
  activeCameraId: string | null;
  alerts: Alert[];
  recentAnomalies: AnomalyResult[];
  currentSummary: EventSummary | null;
  connectionStatus: ConnectionStatus;
  performanceMetrics: PerformanceMetrics;
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error';

export interface PerformanceMetrics {
  fps: number;
  latencyMs: number;
  frameDrops: number;
  processingTimeMs: number;
  memoryUsage: number;
  cpuUsage: number;
}

// UI Component Props Types
export interface CameraViewProps {
  cameraConfig: CameraConfig;
  showMasks: boolean;
  masks: MaskData[];
  onFrameCapture: (frame: VideoFrame) => void;
  onError: (error: Error) => void;
}

export interface AlertListProps {
  alerts: Alert[];
  onAcknowledge: (alertId: string) => void;
  onViewDetails: (alert: Alert) => void;
}

export interface MaskOverlayProps {
  masks: MaskData[];
  frameWidth: number;
  frameHeight: number;
  viewWidth: number;
  viewHeight: number;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  timestamp: number;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

// Event Types
export interface MonitoringEvent {
  id: string;
  type: MonitoringEventType;
  timestamp: number;
  data: unknown;
}

export type MonitoringEventType = 
  | 'frame_captured'
  | 'frame_processed'
  | 'segmentation_complete'
  | 'anomaly_detected'
  | 'alert_triggered'
  | 'audio_classified'
  | 'summary_generated'
  | 'camera_error'
  | 'connection_changed';
