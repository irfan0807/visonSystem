/**
 * VideoProcessor Service
 * Handles camera capture, frame processing, and video streaming
 * Target: 30 FPS with <150ms latency
 */

import { VIDEO_CONFIG } from '../config';
import type {
  VideoFrame,
  ProcessedFrame,
  CameraConfig,
  Resolution,
  PerformanceMetrics,
} from '../types';

// Frame processing callback type
type FrameCallback = (frame: ProcessedFrame) => void;
type ErrorCallback = (error: Error) => void;

// Frame buffer for managing captured frames
interface FrameBuffer {
  frames: VideoFrame[];
  maxSize: number;
}

/**
 * VideoProcessor class for real-time video capture and processing
 */
export class VideoProcessor {
  private isProcessing: boolean = false;
  private frameBuffer: FrameBuffer;
  private frameCallbacks: Set<FrameCallback> = new Set();
  private errorCallbacks: Set<ErrorCallback> = new Set();
  private frameCount: number = 0;
  private startTime: number = 0;
  private lastFrameTime: number = 0;
  private processingTimes: number[] = [];
  private frameDrops: number = 0;
  private cameras: Map<string, CameraConfig> = new Map();
  private activeCameraId: string | null = null;

  constructor() {
    this.frameBuffer = {
      frames: [],
      maxSize: VIDEO_CONFIG.frameBufferSize,
    };
  }

  /**
   * Initialize the video processor with camera configurations
   */
  async initialize(cameras: CameraConfig[]): Promise<void> {
    try {
      cameras.forEach(camera => {
        this.cameras.set(camera.id, camera);
      });

      if (cameras.length > 0) {
        this.activeCameraId = cameras[0].id;
      }

      console.log('[VideoProcessor] Initialized with', cameras.length, 'cameras');
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Start video capture and processing
   */
  startProcessing(): void {
    if (this.isProcessing) {
      console.warn('[VideoProcessor] Already processing');
      return;
    }

    this.isProcessing = true;
    this.startTime = Date.now();
    this.frameCount = 0;
    this.frameDrops = 0;
    this.processingTimes = [];

    console.log('[VideoProcessor] Started processing');
  }

  /**
   * Stop video capture and processing
   */
  stopProcessing(): void {
    this.isProcessing = false;
    this.clearBuffer();
    console.log('[VideoProcessor] Stopped processing');
  }

  /**
   * Process a captured frame from the camera
   */
  async processFrame(frameData: string, cameraId: string): Promise<ProcessedFrame | null> {
    if (!this.isProcessing) {
      return null;
    }

    const startProcessTime = performance.now();
    const timestamp = Date.now();

    // Check if we should drop this frame to maintain target FPS
    const timeSinceLastFrame = timestamp - this.lastFrameTime;
    const targetFrameInterval = 1000 / VIDEO_CONFIG.targetFPS;

    if (timeSinceLastFrame < targetFrameInterval * 0.8) {
      this.frameDrops++;
      return null;
    }

    try {
      // Create video frame
      const frame: VideoFrame = {
        id: this.generateFrameId(),
        timestamp,
        data: frameData,
        width: VIDEO_CONFIG.processingResolution.width,
        height: VIDEO_CONFIG.processingResolution.height,
        cameraId,
      };

      // Add to buffer
      this.addToBuffer(frame);

      // Calculate processing time
      const processingTime = performance.now() - startProcessTime;
      this.recordProcessingTime(processingTime);

      // Create processed frame
      const processedFrame: ProcessedFrame = {
        ...frame,
        processingTime,
      };

      this.frameCount++;
      this.lastFrameTime = timestamp;

      // Notify callbacks
      this.notifyFrameCallbacks(processedFrame);

      // Check latency threshold
      if (processingTime > VIDEO_CONFIG.maxLatencyMs) {
        console.warn('[VideoProcessor] Latency exceeded:', processingTime.toFixed(2), 'ms');
      }

      return processedFrame;
    } catch (error) {
      this.handleError(error as Error);
      return null;
    }
  }

  /**
   * Resize frame for processing
   */
  async resizeFrame(
    base64Data: string,
    targetResolution: Resolution
  ): Promise<string> {
    // In a real implementation, this would use expo-image-manipulator
    // For now, return the original data
    // The actual resize happens on the native side through camera config
    return base64Data;
  }

  /**
   * Convert frame to base64 for API transmission
   */
  frameToBase64(frame: VideoFrame): string {
    return frame.data;
  }

  /**
   * Subscribe to processed frames
   */
  onFrame(callback: FrameCallback): () => void {
    this.frameCallbacks.add(callback);
    return () => this.frameCallbacks.delete(callback);
  }

  /**
   * Subscribe to errors
   */
  onError(callback: ErrorCallback): () => void {
    this.errorCallbacks.add(callback);
    return () => this.errorCallbacks.delete(callback);
  }

  /**
   * Get current performance metrics
   */
  getPerformanceMetrics(): PerformanceMetrics {
    const elapsedTime = (Date.now() - this.startTime) / 1000;
    const averageProcessingTime = this.calculateAverageProcessingTime();

    return {
      fps: elapsedTime > 0 ? this.frameCount / elapsedTime : 0,
      latencyMs: averageProcessingTime,
      frameDrops: this.frameDrops,
      processingTimeMs: averageProcessingTime,
      memoryUsage: 0, // Would be populated by native memory monitoring
      cpuUsage: 0, // Would be populated by native CPU monitoring
    };
  }

  /**
   * Switch active camera
   */
  switchCamera(cameraId: string): boolean {
    if (!this.cameras.has(cameraId)) {
      console.error('[VideoProcessor] Camera not found:', cameraId);
      return false;
    }

    this.activeCameraId = cameraId;
    this.clearBuffer();
    console.log('[VideoProcessor] Switched to camera:', cameraId);
    return true;
  }

  /**
   * Get active camera configuration
   */
  getActiveCamera(): CameraConfig | null {
    if (!this.activeCameraId) return null;
    return this.cameras.get(this.activeCameraId) || null;
  }

  /**
   * Get all camera configurations
   */
  getAllCameras(): CameraConfig[] {
    return Array.from(this.cameras.values());
  }

  /**
   * Add a new camera configuration
   */
  addCamera(camera: CameraConfig): void {
    this.cameras.set(camera.id, camera);
    console.log('[VideoProcessor] Added camera:', camera.id);
  }

  /**
   * Remove a camera configuration
   */
  removeCamera(cameraId: string): void {
    this.cameras.delete(cameraId);
    if (this.activeCameraId === cameraId) {
      const remaining = Array.from(this.cameras.keys());
      this.activeCameraId = remaining.length > 0 ? remaining[0] : null;
    }
    console.log('[VideoProcessor] Removed camera:', cameraId);
  }

  /**
   * Check if processing is active
   */
  isActive(): boolean {
    return this.isProcessing;
  }

  /**
   * Get frame from buffer by index
   */
  getBufferedFrame(index: number): VideoFrame | null {
    if (index < 0 || index >= this.frameBuffer.frames.length) {
      return null;
    }
    return this.frameBuffer.frames[index];
  }

  /**
   * Get recent frames from buffer
   */
  getRecentFrames(count: number): VideoFrame[] {
    return this.frameBuffer.frames.slice(-count);
  }

  // Private methods

  private generateFrameId(): string {
    return `frame_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private addToBuffer(frame: VideoFrame): void {
    this.frameBuffer.frames.push(frame);
    if (this.frameBuffer.frames.length > this.frameBuffer.maxSize) {
      this.frameBuffer.frames.shift();
    }
  }

  private clearBuffer(): void {
    this.frameBuffer.frames = [];
  }

  private recordProcessingTime(time: number): void {
    this.processingTimes.push(time);
    // Keep only last 100 measurements
    if (this.processingTimes.length > 100) {
      this.processingTimes.shift();
    }
  }

  private calculateAverageProcessingTime(): number {
    if (this.processingTimes.length === 0) return 0;
    const sum = this.processingTimes.reduce((a, b) => a + b, 0);
    return sum / this.processingTimes.length;
  }

  private notifyFrameCallbacks(frame: ProcessedFrame): void {
    this.frameCallbacks.forEach(callback => {
      try {
        callback(frame);
      } catch (error) {
        console.error('[VideoProcessor] Callback error:', error);
      }
    });
  }

  private handleError(error: Error): void {
    console.error('[VideoProcessor] Error:', error.message);
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (callbackError) {
        console.error('[VideoProcessor] Error callback failed:', callbackError);
      }
    });
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stopProcessing();
    this.frameCallbacks.clear();
    this.errorCallbacks.clear();
    this.cameras.clear();
    console.log('[VideoProcessor] Disposed');
  }
}

// Singleton instance
let videoProcessorInstance: VideoProcessor | null = null;

/**
 * Get the VideoProcessor singleton instance
 */
export function getVideoProcessor(): VideoProcessor {
  if (!videoProcessorInstance) {
    videoProcessorInstance = new VideoProcessor();
  }
  return videoProcessorInstance;
}

/**
 * Reset the VideoProcessor instance
 */
export function resetVideoProcessor(): void {
  if (videoProcessorInstance) {
    videoProcessorInstance.dispose();
    videoProcessorInstance = null;
  }
}

export default VideoProcessor;
