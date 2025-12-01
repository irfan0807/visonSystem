/**
 * AudioAnalyzer Service
 * Handles audio capture, processing, and classification
 * for anomaly detection in real-time monitoring
 */

import { AUDIO_CONFIG, API_ENDPOINTS } from '../config';
import type {
  AudioConfig,
  AudioFrame,
  AudioClassification,
  ClassificationResult,
  AudioCategory,
  AnomalyResult,
} from '../types';

// Callback types
type ClassificationCallback = (classification: AudioClassification) => void;
type AnomalyCallback = (anomaly: AnomalyResult) => void;
type ErrorCallback = (error: Error) => void;

// Audio buffer for analysis
interface AudioBuffer {
  samples: Float32Array;
  sampleRate: number;
  duration: number;
}

// Audio analysis result from local processing
interface AudioFeatures {
  rms: number;
  zeroCrossingRate: number;
  spectralCentroid: number;
  mfcc: number[];
  energy: number;
}

/**
 * AudioAnalyzer class for real-time audio analysis and classification
 */
export class AudioAnalyzer {
  private isAnalyzing: boolean = false;
  private config: AudioConfig;
  private audioBuffer: AudioBuffer;
  private classificationCallbacks: Set<ClassificationCallback> = new Set();
  private anomalyCallbacks: Set<AnomalyCallback> = new Set();
  private errorCallbacks: Set<ErrorCallback> = new Set();
  private frameCount: number = 0;
  private startTime: number = 0;
  private anomalyThreshold: number = 0.6;
  private baselineFeatures: AudioFeatures | null = null;
  private analysisInterval: ReturnType<typeof setInterval> | null = null;
  private recentClassifications: AudioClassification[] = [];

  // Audio category labels for classification
  private readonly AUDIO_LABELS: Record<string, AudioCategory> = {
    'speech': 'speech',
    'talking': 'speech',
    'conversation': 'speech',
    'music': 'music',
    'singing': 'music',
    'instrument': 'music',
    'traffic': 'environmental',
    'wind': 'environmental',
    'rain': 'environmental',
    'machine': 'mechanical',
    'motor': 'mechanical',
    'engine': 'mechanical',
    'alarm': 'alarm',
    'siren': 'alarm',
    'beep': 'alarm',
    'dog': 'animal',
    'cat': 'animal',
    'bird': 'animal',
  };

  constructor(config: Partial<AudioConfig> = {}) {
    this.config = { ...AUDIO_CONFIG, ...config };
    this.audioBuffer = {
      samples: new Float32Array(this.config.bufferSize),
      sampleRate: this.config.sampleRate,
      duration: this.config.bufferSize / this.config.sampleRate,
    };
  }

  /**
   * Initialize the audio analyzer
   */
  async initialize(): Promise<void> {
    try {
      // Reset state
      this.frameCount = 0;
      this.recentClassifications = [];
      this.baselineFeatures = null;

      console.log('[AudioAnalyzer] Initialized with config:', {
        sampleRate: this.config.sampleRate,
        bufferSize: this.config.bufferSize,
      });
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Start audio analysis
   */
  startAnalysis(): void {
    if (this.isAnalyzing) {
      console.warn('[AudioAnalyzer] Already analyzing');
      return;
    }

    if (!this.config.enabled) {
      console.warn('[AudioAnalyzer] Audio analysis is disabled');
      return;
    }

    this.isAnalyzing = true;
    this.startTime = Date.now();
    this.frameCount = 0;

    console.log('[AudioAnalyzer] Started analysis');
  }

  /**
   * Stop audio analysis
   */
  stopAnalysis(): void {
    this.isAnalyzing = false;

    if (this.analysisInterval) {
      clearInterval(this.analysisInterval);
      this.analysisInterval = null;
    }

    console.log('[AudioAnalyzer] Stopped analysis');
  }

  /**
   * Process an audio frame
   */
  async processAudioFrame(audioData: Float32Array | Int16Array): Promise<AudioClassification | null> {
    if (!this.isAnalyzing) {
      return null;
    }

    try {
      const timestamp = Date.now();

      // Convert to Float32Array if needed
      const floatData = audioData instanceof Float32Array 
        ? audioData 
        : this.int16ToFloat32(audioData);

      // Create audio frame
      const audioFrame: AudioFrame = {
        id: this.generateFrameId(),
        timestamp,
        data: floatData,
        sampleRate: this.config.sampleRate,
        duration: floatData.length / this.config.sampleRate,
      };

      // Extract audio features locally
      const features = this.extractFeatures(floatData);

      // Set baseline if not set
      if (!this.baselineFeatures) {
        this.baselineFeatures = features;
      }

      // Perform local anomaly detection
      const anomalyScore = this.calculateAnomalyScore(features);
      const isAnomaly = anomalyScore > this.anomalyThreshold;

      // Classify audio
      const classifications = this.classifyLocally(features);

      // Create classification result
      const classification: AudioClassification = {
        id: this.generateClassificationId(),
        frameId: audioFrame.id,
        timestamp,
        classifications,
        anomalyScore,
        isAnomaly,
      };

      // Store recent classification
      this.recentClassifications.push(classification);
      if (this.recentClassifications.length > 100) {
        this.recentClassifications.shift();
      }

      this.frameCount++;

      // Notify callbacks
      this.notifyClassificationCallbacks(classification);

      // If anomaly detected, create anomaly result
      if (isAnomaly) {
        const anomaly = this.createAnomalyFromClassification(classification, features);
        this.notifyAnomalyCallbacks(anomaly);
      }

      return classification;
    } catch (error) {
      this.handleError(error as Error);
      return null;
    }
  }

  /**
   * Classify audio using remote API
   */
  async classifyRemote(audioData: Float32Array): Promise<ClassificationResult[]> {
    try {
      const base64Audio = this.float32ToBase64(audioData);

      const response = await fetch(API_ENDPOINTS.AUDIO_CLASSIFY, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio: base64Audio,
          sampleRate: this.config.sampleRate,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();
      return result.classifications || [];
    } catch (error) {
      console.error('[AudioAnalyzer] Remote classification failed:', error);
      // Fall back to local classification
      return this.classifyLocally(this.extractFeatures(audioData));
    }
  }

  /**
   * Subscribe to classification results
   */
  onClassification(callback: ClassificationCallback): () => void {
    this.classificationCallbacks.add(callback);
    return () => this.classificationCallbacks.delete(callback);
  }

  /**
   * Subscribe to anomaly detections
   */
  onAnomaly(callback: AnomalyCallback): () => void {
    this.anomalyCallbacks.add(callback);
    return () => this.anomalyCallbacks.delete(callback);
  }

  /**
   * Subscribe to errors
   */
  onError(callback: ErrorCallback): () => void {
    this.errorCallbacks.add(callback);
    return () => this.errorCallbacks.delete(callback);
  }

  /**
   * Get recent classifications
   */
  getRecentClassifications(count: number = 10): AudioClassification[] {
    return this.recentClassifications.slice(-count);
  }

  /**
   * Set anomaly detection threshold
   */
  setAnomalyThreshold(threshold: number): void {
    this.anomalyThreshold = Math.max(0, Math.min(1, threshold));
  }

  /**
   * Get current anomaly threshold
   */
  getAnomalyThreshold(): number {
    return this.anomalyThreshold;
  }

  /**
   * Reset baseline features (for recalibration)
   */
  resetBaseline(): void {
    this.baselineFeatures = null;
    console.log('[AudioAnalyzer] Baseline reset');
  }

  /**
   * Check if analyzer is active
   */
  isActive(): boolean {
    return this.isAnalyzing;
  }

  /**
   * Get analysis statistics
   */
  getStatistics(): { frameCount: number; duration: number; anomalyRate: number } {
    const duration = (Date.now() - this.startTime) / 1000;
    const anomalyCount = this.recentClassifications.filter(c => c.isAnomaly).length;
    const anomalyRate = this.recentClassifications.length > 0 
      ? anomalyCount / this.recentClassifications.length 
      : 0;

    return {
      frameCount: this.frameCount,
      duration,
      anomalyRate,
    };
  }

  // Private methods

  private generateFrameId(): string {
    return `audio_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private generateClassificationId(): string {
    return `class_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Convert Int16Array to Float32Array
   */
  private int16ToFloat32(int16Array: Int16Array): Float32Array {
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768.0;
    }
    return float32Array;
  }

  /**
   * Convert Float32Array to base64 for API transmission
   * Uses chunked processing for better performance with large buffers
   */
  private float32ToBase64(float32Array: Float32Array): string {
    const buffer = float32Array.buffer;
    const bytes = new Uint8Array(buffer, float32Array.byteOffset, float32Array.byteLength);
    
    // Process in chunks for better performance
    const chunkSize = 8192;
    const chunks: string[] = [];
    
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
      chunks.push(String.fromCharCode.apply(null, Array.from(chunk)));
    }
    
    return btoa(chunks.join(''));
  }

  /**
   * Extract audio features for analysis
   */
  private extractFeatures(samples: Float32Array): AudioFeatures {
    // Calculate RMS (Root Mean Square) - audio energy level
    let sumSquares = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSquares += samples[i] * samples[i];
    }
    const rms = Math.sqrt(sumSquares / samples.length);

    // Calculate Zero Crossing Rate
    let zeroCrossings = 0;
    for (let i = 1; i < samples.length; i++) {
      if ((samples[i] >= 0 && samples[i - 1] < 0) || 
          (samples[i] < 0 && samples[i - 1] >= 0)) {
        zeroCrossings++;
      }
    }
    const zeroCrossingRate = zeroCrossings / samples.length;

    // Calculate spectral centroid (simplified)
    // In a real implementation, this would use FFT
    let weightedSum = 0;
    let totalMagnitude = 0;
    for (let i = 0; i < samples.length; i++) {
      const magnitude = Math.abs(samples[i]);
      weightedSum += i * magnitude;
      totalMagnitude += magnitude;
    }
    const spectralCentroid = totalMagnitude > 0 ? weightedSum / totalMagnitude : 0;

    // Calculate energy
    const energy = sumSquares;

    // Simplified MFCC placeholder (would need full FFT implementation)
    const mfcc = Array(13).fill(0).map((_, i) => {
      return rms * Math.cos(i * Math.PI / 13) + zeroCrossingRate * Math.sin(i * Math.PI / 13);
    });

    return {
      rms,
      zeroCrossingRate,
      spectralCentroid,
      mfcc,
      energy,
    };
  }

  /**
   * Calculate anomaly score based on deviation from baseline
   */
  private calculateAnomalyScore(features: AudioFeatures): number {
    if (!this.baselineFeatures) {
      return 0;
    }

    // Calculate normalized differences
    const rmsDeviation = Math.abs(features.rms - this.baselineFeatures.rms) / 
      Math.max(this.baselineFeatures.rms, 0.001);
    const zcrDeviation = Math.abs(features.zeroCrossingRate - this.baselineFeatures.zeroCrossingRate) /
      Math.max(this.baselineFeatures.zeroCrossingRate, 0.001);
    const centroidDeviation = Math.abs(features.spectralCentroid - this.baselineFeatures.spectralCentroid) /
      Math.max(this.baselineFeatures.spectralCentroid, 0.001);

    // Weight and combine
    const score = (
      0.4 * Math.min(rmsDeviation, 1) +
      0.3 * Math.min(zcrDeviation, 1) +
      0.3 * Math.min(centroidDeviation, 1)
    );

    return Math.min(score, 1);
  }

  /**
   * Perform local audio classification based on features
   */
  private classifyLocally(features: AudioFeatures): ClassificationResult[] {
    const classifications: ClassificationResult[] = [];

    // Speech detection (high zero crossing rate, moderate energy)
    if (features.zeroCrossingRate > 0.1 && features.rms > 0.01 && features.rms < 0.3) {
      classifications.push({
        label: 'speech',
        confidence: 0.5 + features.zeroCrossingRate,
        category: 'speech',
      });
    }

    // Music detection (high energy, varied spectral content)
    if (features.rms > 0.05 && features.spectralCentroid > features.rms * 100) {
      classifications.push({
        label: 'music',
        confidence: Math.min(0.7, features.rms * 2),
        category: 'music',
      });
    }

    // Alarm detection (high energy, high frequency content)
    if (features.rms > 0.3 && features.zeroCrossingRate > 0.3) {
      classifications.push({
        label: 'alarm',
        confidence: Math.min(0.9, features.rms + features.zeroCrossingRate),
        category: 'alarm',
      });
    }

    // Environmental sounds (low zero crossing, variable energy)
    if (features.zeroCrossingRate < 0.1 && features.rms > 0.001) {
      classifications.push({
        label: 'environmental',
        confidence: 0.4,
        category: 'environmental',
      });
    }

    // Mechanical sounds (consistent energy, low variation)
    if (features.rms > 0.02 && features.zeroCrossingRate < 0.05) {
      classifications.push({
        label: 'mechanical',
        confidence: 0.5,
        category: 'mechanical',
      });
    }

    // If no classifications, mark as unknown
    if (classifications.length === 0) {
      classifications.push({
        label: 'unknown',
        confidence: 0.3,
        category: 'unknown',
      });
    }

    // Sort by confidence
    classifications.sort((a, b) => b.confidence - a.confidence);

    return classifications.slice(0, 3); // Return top 3
  }

  /**
   * Create anomaly result from classification
   */
  private createAnomalyFromClassification(
    classification: AudioClassification,
    features: AudioFeatures
  ): AnomalyResult {
    const topClassification = classification.classifications[0];

    return {
      id: `anomaly_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      timestamp: classification.timestamp,
      type: 'audio_anomaly',
      severity: this.getSeverityFromScore(classification.anomalyScore),
      confidence: classification.anomalyScore,
      description: `Audio anomaly detected: ${topClassification?.label || 'unknown'} (score: ${classification.anomalyScore.toFixed(2)})`,
      source: 'audio',
      metadata: {
        classification: topClassification,
        features: {
          rms: features.rms,
          zeroCrossingRate: features.zeroCrossingRate,
          energy: features.energy,
        },
      },
    };
  }

  /**
   * Map anomaly score to severity level
   */
  private getSeverityFromScore(score: number): 'low' | 'medium' | 'high' | 'critical' {
    if (score >= 0.9) return 'critical';
    if (score >= 0.75) return 'high';
    if (score >= 0.6) return 'medium';
    return 'low';
  }

  /**
   * Notify classification callbacks
   */
  private notifyClassificationCallbacks(classification: AudioClassification): void {
    this.classificationCallbacks.forEach(callback => {
      try {
        callback(classification);
      } catch (error) {
        console.error('[AudioAnalyzer] Classification callback error:', error);
      }
    });
  }

  /**
   * Notify anomaly callbacks
   */
  private notifyAnomalyCallbacks(anomaly: AnomalyResult): void {
    this.anomalyCallbacks.forEach(callback => {
      try {
        callback(anomaly);
      } catch (error) {
        console.error('[AudioAnalyzer] Anomaly callback error:', error);
      }
    });
  }

  /**
   * Handle errors
   */
  private handleError(error: Error): void {
    console.error('[AudioAnalyzer] Error:', error.message);
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (callbackError) {
        console.error('[AudioAnalyzer] Error callback failed:', callbackError);
      }
    });
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stopAnalysis();
    this.classificationCallbacks.clear();
    this.anomalyCallbacks.clear();
    this.errorCallbacks.clear();
    this.recentClassifications = [];
    console.log('[AudioAnalyzer] Disposed');
  }
}

// Singleton instance
let audioAnalyzerInstance: AudioAnalyzer | null = null;

/**
 * Get the AudioAnalyzer singleton instance
 */
export function getAudioAnalyzer(): AudioAnalyzer {
  if (!audioAnalyzerInstance) {
    audioAnalyzerInstance = new AudioAnalyzer();
  }
  return audioAnalyzerInstance;
}

/**
 * Reset the AudioAnalyzer instance
 */
export function resetAudioAnalyzer(): void {
  if (audioAnalyzerInstance) {
    audioAnalyzerInstance.dispose();
    audioAnalyzerInstance = null;
  }
}

export default AudioAnalyzer;
