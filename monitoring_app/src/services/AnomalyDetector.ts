/**
 * AnomalyDetector Service
 * Handles visual and motion-based anomaly detection
 * for real-time monitoring
 */

import { ANOMALY_CONFIG } from '../config';
import type {
  AnomalyConfig,
  AnomalyResult,
  AnomalyType,
  AnomalySeverity,
  ProcessedFrame,
  SegmentationMask,
  MaskData,
} from '../types';

// Callback types
type AnomalyCallback = (anomaly: AnomalyResult) => void;
type ErrorCallback = (error: Error) => void;

// Motion detection state
interface MotionState {
  previousFrame: ProcessedFrame | null;
  motionHistory: number[];
  baselineMotion: number;
}

// Object tracking state
interface TrackedObject {
  id: string;
  label: string;
  lastSeen: number;
  positions: { x: number; y: number; timestamp: number }[];
  velocity: { x: number; y: number };
}

// Zone monitoring
interface MonitoringZone {
  id: string;
  name: string;
  polygon: { x: number; y: number }[];
  type: 'restricted' | 'monitored' | 'safe';
  enabled: boolean;
}

/**
 * AnomalyDetector class for visual anomaly detection
 */
export class AnomalyDetector {
  private config: AnomalyConfig;
  private isDetecting: boolean = false;
  private anomalyCallbacks: Set<AnomalyCallback> = new Set();
  private errorCallbacks: Set<ErrorCallback> = new Set();
  private detectedAnomalies: AnomalyResult[] = [];
  private lastAnomalyTime: Map<string, number> = new Map();
  private motionState: MotionState;
  private trackedObjects: Map<string, TrackedObject> = new Map();
  private monitoringZones: MonitoringZone[] = [];
  private frameCount: number = 0;
  private startTime: number = 0;

  // Anomaly type thresholds
  private readonly MOTION_THRESHOLD = 0.3;
  private readonly CROWD_THRESHOLD = 5;
  private readonly LOITERING_THRESHOLD = 30000; // 30 seconds

  constructor(config: Partial<AnomalyConfig> = {}) {
    this.config = { ...ANOMALY_CONFIG, ...config };
    this.motionState = {
      previousFrame: null,
      motionHistory: [],
      baselineMotion: 0,
    };
  }

  /**
   * Initialize the anomaly detector
   */
  async initialize(): Promise<void> {
    try {
      this.frameCount = 0;
      this.detectedAnomalies = [];
      this.trackedObjects.clear();

      console.log('[AnomalyDetector] Initialized');
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Start anomaly detection
   */
  startDetection(): void {
    if (this.isDetecting) {
      console.warn('[AnomalyDetector] Already detecting');
      return;
    }

    this.isDetecting = true;
    this.startTime = Date.now();
    this.frameCount = 0;

    console.log('[AnomalyDetector] Started detection');
  }

  /**
   * Stop anomaly detection
   */
  stopDetection(): void {
    this.isDetecting = false;
    console.log('[AnomalyDetector] Stopped detection');
  }

  /**
   * Analyze a frame for anomalies
   */
  async analyzeFrame(
    frame: ProcessedFrame,
    segmentationMask?: SegmentationMask
  ): Promise<AnomalyResult[]> {
    if (!this.isDetecting) {
      return [];
    }

    const anomalies: AnomalyResult[] = [];

    try {
      // Motion detection
      if (this.config.enableMotionDetection) {
        const motionAnomaly = this.detectMotion(frame);
        if (motionAnomaly) {
          anomalies.push(motionAnomaly);
        }
      }

      // Visual anomaly detection with segmentation
      if (this.config.enableVisualDetection && segmentationMask) {
        const visualAnomalies = this.analyzeSegmentation(frame, segmentationMask);
        anomalies.push(...visualAnomalies);
      }

      // Update tracking
      if (segmentationMask) {
        this.updateObjectTracking(segmentationMask);
      }

      // Check for zone violations
      if (segmentationMask) {
        const zoneAnomalies = this.checkZoneViolations(segmentationMask);
        anomalies.push(...zoneAnomalies);
      }

      // Check for loitering
      const loiteringAnomalies = this.detectLoitering();
      anomalies.push(...loiteringAnomalies);

      // Check for crowd formation
      if (segmentationMask) {
        const crowdAnomaly = this.detectCrowdFormation(segmentationMask);
        if (crowdAnomaly) {
          anomalies.push(crowdAnomaly);
        }
      }

      // Store frame for next iteration
      this.motionState.previousFrame = frame;
      this.frameCount++;

      // Apply cooldown filter
      const filteredAnomalies = this.applyAlertCooldown(anomalies);

      // Store and notify
      this.detectedAnomalies.push(...filteredAnomalies);
      filteredAnomalies.forEach(anomaly => {
        this.notifyAnomalyCallbacks(anomaly);
      });

      return filteredAnomalies;
    } catch (error) {
      this.handleError(error as Error);
      return [];
    }
  }

  /**
   * Detect motion between frames
   */
  private detectMotion(frame: ProcessedFrame): AnomalyResult | null {
    if (!this.motionState.previousFrame) {
      return null;
    }

    // In a real implementation, this would compare frame pixels
    // Here we simulate motion detection
    const motionScore = this.calculateMotionScore(frame);

    // Update motion history
    this.motionState.motionHistory.push(motionScore);
    if (this.motionState.motionHistory.length > 30) {
      this.motionState.motionHistory.shift();
    }

    // Calculate baseline if we have enough history
    if (this.motionState.motionHistory.length >= 30) {
      this.motionState.baselineMotion = this.motionState.motionHistory
        .slice(0, 20)
        .reduce((a, b) => a + b, 0) / 20;
    }

    // Check for significant motion
    const deviation = motionScore - this.motionState.baselineMotion;
    if (deviation > this.config.motionThreshold) {
      return {
        id: this.generateAnomalyId(),
        timestamp: Date.now(),
        type: 'motion_detected',
        severity: this.getSeverityFromScore(deviation),
        confidence: Math.min(deviation / this.config.motionThreshold, 1),
        description: `Motion detected: ${(deviation * 100).toFixed(1)}% above baseline`,
        source: 'visual',
        metadata: {
          motionScore,
          baseline: this.motionState.baselineMotion,
          frameId: frame.id,
        },
      };
    }

    return null;
  }

  /**
   * Calculate motion score (simplified)
   */
  private calculateMotionScore(frame: ProcessedFrame): number {
    // In a real implementation, this would compare pixel differences
    // Simulated motion score based on timestamp variation
    const timeDelta = frame.timestamp - (this.motionState.previousFrame?.timestamp || 0);
    const normalizedDelta = Math.min(timeDelta / 100, 1);
    return Math.random() * 0.3 * normalizedDelta; // Simulated
  }

  /**
   * Analyze segmentation masks for anomalies
   */
  private analyzeSegmentation(
    frame: ProcessedFrame,
    mask: SegmentationMask
  ): AnomalyResult[] {
    const anomalies: AnomalyResult[] = [];

    // Check for unusual objects
    for (const maskData of mask.masks) {
      // Check for low confidence detections
      if (maskData.confidence < this.config.visualThreshold) {
        continue;
      }

      // Check for unusual size
      const sizeRatio = maskData.area / (frame.width * frame.height);
      if (sizeRatio > 0.5) {
        anomalies.push({
          id: this.generateAnomalyId(),
          timestamp: Date.now(),
          type: 'unusual_activity',
          severity: 'medium',
          confidence: maskData.confidence,
          description: `Large object detected: ${maskData.label} (${(sizeRatio * 100).toFixed(1)}% of frame)`,
          source: 'visual',
          metadata: {
            maskId: maskData.id,
            label: maskData.label,
            sizeRatio,
          },
        });
      }

      // Check for specific object types
      if (this.isAlertableObject(maskData.label)) {
        anomalies.push({
          id: this.generateAnomalyId(),
          timestamp: Date.now(),
          type: 'object_detected',
          severity: this.getObjectSeverity(maskData.label),
          confidence: maskData.confidence,
          description: `Alert object detected: ${maskData.label}`,
          source: 'visual',
          metadata: {
            maskId: maskData.id,
            label: maskData.label,
            boundingBox: maskData.boundingBox,
          },
        });
      }
    }

    return anomalies;
  }

  /**
   * Update object tracking state
   */
  private updateObjectTracking(mask: SegmentationMask): void {
    const currentTime = Date.now();

    for (const maskData of mask.masks) {
      const centerX = maskData.boundingBox.x + maskData.boundingBox.width / 2;
      const centerY = maskData.boundingBox.y + maskData.boundingBox.height / 2;

      // Try to match with existing tracked object
      let matched = false;
      this.trackedObjects.forEach((tracked, id) => {
        if (tracked.label === maskData.label && !matched) {
          const lastPos = tracked.positions[tracked.positions.length - 1];
          const distance = Math.sqrt(
            Math.pow(centerX - lastPos.x, 2) + Math.pow(centerY - lastPos.y, 2)
          );

          if (distance < 100) { // Matching threshold
            tracked.positions.push({ x: centerX, y: centerY, timestamp: currentTime });
            tracked.lastSeen = currentTime;

            // Update velocity
            if (tracked.positions.length > 1) {
              const prev = tracked.positions[tracked.positions.length - 2];
              const dt = (currentTime - prev.timestamp) / 1000;
              if (dt > 0) {
                tracked.velocity = {
                  x: (centerX - prev.x) / dt,
                  y: (centerY - prev.y) / dt,
                };
              }
            }

            // Limit position history
            if (tracked.positions.length > 100) {
              tracked.positions.shift();
            }

            matched = true;
          }
        }
      });

      // Create new tracked object if no match
      if (!matched) {
        const newId = `obj_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
        this.trackedObjects.set(newId, {
          id: newId,
          label: maskData.label,
          lastSeen: currentTime,
          positions: [{ x: centerX, y: centerY, timestamp: currentTime }],
          velocity: { x: 0, y: 0 },
        });
      }
    }

    // Clean up old tracked objects
    this.trackedObjects.forEach((tracked, id) => {
      if (currentTime - tracked.lastSeen > 5000) {
        this.trackedObjects.delete(id);
      }
    });
  }

  /**
   * Check for zone violations
   */
  private checkZoneViolations(mask: SegmentationMask): AnomalyResult[] {
    const anomalies: AnomalyResult[] = [];

    for (const zone of this.monitoringZones) {
      if (!zone.enabled || zone.type !== 'restricted') {
        continue;
      }

      for (const maskData of mask.masks) {
        const centerX = maskData.boundingBox.x + maskData.boundingBox.width / 2;
        const centerY = maskData.boundingBox.y + maskData.boundingBox.height / 2;

        if (this.isPointInPolygon({ x: centerX, y: centerY }, zone.polygon)) {
          anomalies.push({
            id: this.generateAnomalyId(),
            timestamp: Date.now(),
            type: 'restricted_area_breach',
            severity: 'high',
            confidence: maskData.confidence,
            description: `${maskData.label} detected in restricted zone: ${zone.name}`,
            source: 'visual',
            metadata: {
              zoneId: zone.id,
              zoneName: zone.name,
              objectLabel: maskData.label,
            },
          });
        }
      }
    }

    return anomalies;
  }

  /**
   * Detect loitering behavior
   */
  private detectLoitering(): AnomalyResult[] {
    const anomalies: AnomalyResult[] = [];
    const currentTime = Date.now();

    this.trackedObjects.forEach(tracked => {
      if (tracked.positions.length < 10) {
        return;
      }

      // Check if object has been in similar position for too long
      const recentPositions = tracked.positions.slice(-10);
      const avgX = recentPositions.reduce((sum, p) => sum + p.x, 0) / recentPositions.length;
      const avgY = recentPositions.reduce((sum, p) => sum + p.y, 0) / recentPositions.length;

      const variance = recentPositions.reduce((sum, p) => {
        return sum + Math.pow(p.x - avgX, 2) + Math.pow(p.y - avgY, 2);
      }, 0) / recentPositions.length;

      const duration = currentTime - recentPositions[0].timestamp;

      if (variance < 500 && duration > this.LOITERING_THRESHOLD) {
        anomalies.push({
          id: this.generateAnomalyId(),
          timestamp: currentTime,
          type: 'unusual_activity',
          severity: 'medium',
          confidence: 0.7,
          description: `Loitering detected: ${tracked.label} stationary for ${Math.round(duration / 1000)}s`,
          source: 'visual',
          metadata: {
            objectId: tracked.id,
            duration,
            position: { x: avgX, y: avgY },
          },
        });
      }
    });

    return anomalies;
  }

  /**
   * Detect crowd formation
   */
  private detectCrowdFormation(mask: SegmentationMask): AnomalyResult | null {
    // Count people in the frame
    const personMasks = mask.masks.filter(m => 
      m.label.toLowerCase().includes('person') || 
      m.label.toLowerCase().includes('human')
    );

    if (personMasks.length >= this.CROWD_THRESHOLD) {
      return {
        id: this.generateAnomalyId(),
        timestamp: Date.now(),
        type: 'crowd_formation',
        severity: personMasks.length >= this.CROWD_THRESHOLD * 2 ? 'high' : 'medium',
        confidence: 0.8,
        description: `Crowd detected: ${personMasks.length} people in frame`,
        source: 'visual',
        metadata: {
          personCount: personMasks.length,
          masks: personMasks.map(m => m.id),
        },
      };
    }

    return null;
  }

  /**
   * Add a monitoring zone
   */
  addMonitoringZone(zone: Omit<MonitoringZone, 'id'>): string {
    const id = `zone_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    const newZone: MonitoringZone = { ...zone, id };
    this.monitoringZones.push(newZone);
    console.log('[AnomalyDetector] Added zone:', id);
    return id;
  }

  /**
   * Remove a monitoring zone
   */
  removeMonitoringZone(zoneId: string): boolean {
    const index = this.monitoringZones.findIndex(z => z.id === zoneId);
    if (index >= 0) {
      this.monitoringZones.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * Get all monitoring zones
   */
  getMonitoringZones(): MonitoringZone[] {
    return [...this.monitoringZones];
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
   * Get recent anomalies
   */
  getRecentAnomalies(count: number = 10): AnomalyResult[] {
    return this.detectedAnomalies
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, count);
  }

  /**
   * Get anomalies by type
   */
  getAnomaliesByType(type: AnomalyType): AnomalyResult[] {
    return this.detectedAnomalies.filter(a => a.type === type);
  }

  /**
   * Clear anomaly history
   */
  clearAnomalyHistory(): void {
    this.detectedAnomalies = [];
  }

  /**
   * Get detection statistics
   */
  getStatistics(): {
    frameCount: number;
    anomalyCount: number;
    byType: Record<AnomalyType, number>;
    trackedObjects: number;
  } {
    const byType: Partial<Record<AnomalyType, number>> = {};
    
    this.detectedAnomalies.forEach(anomaly => {
      byType[anomaly.type] = (byType[anomaly.type] || 0) + 1;
    });

    return {
      frameCount: this.frameCount,
      anomalyCount: this.detectedAnomalies.length,
      byType: byType as Record<AnomalyType, number>,
      trackedObjects: this.trackedObjects.size,
    };
  }

  /**
   * Check if detector is active
   */
  isActive(): boolean {
    return this.isDetecting;
  }

  // Private helper methods

  private generateAnomalyId(): string {
    return `anomaly_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private getSeverityFromScore(score: number): AnomalySeverity {
    if (score >= 0.9) return 'critical';
    if (score >= 0.7) return 'high';
    if (score >= 0.4) return 'medium';
    return 'low';
  }

  private isAlertableObject(label: string): boolean {
    const alertableObjects = [
      'weapon', 'gun', 'knife', 'fire', 'smoke',
      'package', 'bag', 'suspicious',
    ];
    return alertableObjects.some(obj => 
      label.toLowerCase().includes(obj)
    );
  }

  private getObjectSeverity(label: string): AnomalySeverity {
    const criticalObjects = ['weapon', 'gun', 'knife', 'fire'];
    const highObjects = ['smoke', 'suspicious'];
    
    if (criticalObjects.some(obj => label.toLowerCase().includes(obj))) {
      return 'critical';
    }
    if (highObjects.some(obj => label.toLowerCase().includes(obj))) {
      return 'high';
    }
    return 'medium';
  }

  private isPointInPolygon(point: { x: number; y: number }, polygon: { x: number; y: number }[]): boolean {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x, yi = polygon[i].y;
      const xj = polygon[j].x, yj = polygon[j].y;

      if (((yi > point.y) !== (yj > point.y)) &&
          (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  private applyAlertCooldown(anomalies: AnomalyResult[]): AnomalyResult[] {
    const currentTime = Date.now();
    
    return anomalies.filter(anomaly => {
      const key = `${anomaly.type}_${anomaly.source}`;
      const lastTime = this.lastAnomalyTime.get(key);
      
      if (!lastTime || currentTime - lastTime >= this.config.alertCooldownMs) {
        this.lastAnomalyTime.set(key, currentTime);
        return true;
      }
      
      return false;
    });
  }

  private notifyAnomalyCallbacks(anomaly: AnomalyResult): void {
    this.anomalyCallbacks.forEach(callback => {
      try {
        callback(anomaly);
      } catch (error) {
        console.error('[AnomalyDetector] Callback error:', error);
      }
    });
  }

  private handleError(error: Error): void {
    console.error('[AnomalyDetector] Error:', error.message);
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (callbackError) {
        console.error('[AnomalyDetector] Error callback failed:', callbackError);
      }
    });
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stopDetection();
    this.anomalyCallbacks.clear();
    this.errorCallbacks.clear();
    this.trackedObjects.clear();
    this.detectedAnomalies = [];
    console.log('[AnomalyDetector] Disposed');
  }
}

// Singleton instance
let anomalyDetectorInstance: AnomalyDetector | null = null;

/**
 * Get the AnomalyDetector singleton instance
 */
export function getAnomalyDetector(): AnomalyDetector {
  if (!anomalyDetectorInstance) {
    anomalyDetectorInstance = new AnomalyDetector();
  }
  return anomalyDetectorInstance;
}

/**
 * Reset the AnomalyDetector instance
 */
export function resetAnomalyDetector(): void {
  if (anomalyDetectorInstance) {
    anomalyDetectorInstance.dispose();
    anomalyDetectorInstance = null;
  }
}

export default AnomalyDetector;
