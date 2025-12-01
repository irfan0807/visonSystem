/**
 * useMonitoring Hook
 * Manages the monitoring state and coordinates all services
 */

import { useCallback, useEffect, useState, useRef } from 'react';
import {
  getVideoProcessor,
  getAudioAnalyzer,
  getAlertManager,
  getSAM3Integration,
  getAnomalyDetector,
  getOpenAISummary,
} from '../services';
import type {
  MonitoringState,
  CameraConfig,
  Alert,
  AnomalyResult,
  EventSummary,
  PerformanceMetrics,
  ProcessedFrame,
  SegmentationMask,
  AudioClassification,
} from '../types';

interface UseMonitoringReturn {
  state: MonitoringState;
  startMonitoring: () => Promise<void>;
  stopMonitoring: () => void;
  switchCamera: (cameraId: string) => void;
  addCamera: (camera: CameraConfig) => void;
  removeCamera: (cameraId: string) => void;
  acknowledgeAlert: (alertId: string) => void;
  acknowledgeAllAlerts: () => void;
  generateSummary: () => Promise<EventSummary | null>;
  processFrame: (frameData: string, cameraId: string) => Promise<void>;
  processAudio: (audioData: Float32Array) => Promise<void>;
}

/**
 * Custom hook for managing the monitoring system
 */
export function useMonitoring(): UseMonitoringReturn {
  const [state, setState] = useState<MonitoringState>({
    isMonitoring: false,
    cameras: [],
    activeCameraId: null,
    alerts: [],
    recentAnomalies: [],
    currentSummary: null,
    connectionStatus: 'disconnected',
    performanceMetrics: {
      fps: 0,
      latencyMs: 0,
      frameDrops: 0,
      processingTimeMs: 0,
      memoryUsage: 0,
      cpuUsage: 0,
    },
  });

  // Service references
  const videoProcessor = useRef(getVideoProcessor());
  const audioAnalyzer = useRef(getAudioAnalyzer());
  const alertManager = useRef(getAlertManager());
  const sam3Integration = useRef(getSAM3Integration());
  const anomalyDetector = useRef(getAnomalyDetector());
  const openAISummary = useRef(getOpenAISummary());

  // Performance tracking
  const metricsInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  /**
   * Initialize all services
   */
  const initializeServices = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, connectionStatus: 'connecting' }));

      await Promise.all([
        sam3Integration.current.initialize(),
        audioAnalyzer.current.initialize(),
        alertManager.current.initialize(),
        anomalyDetector.current.initialize(),
        openAISummary.current.initialize(),
      ]);

      // Register for push notifications
      await alertManager.current.registerForPushNotifications();

      setState(prev => ({ ...prev, connectionStatus: 'connected' }));
    } catch (error) {
      console.error('[useMonitoring] Failed to initialize services:', error);
      setState(prev => ({ ...prev, connectionStatus: 'error' }));
      throw error;
    }
  }, []);

  /**
   * Start monitoring
   */
  const startMonitoring = useCallback(async () => {
    try {
      await initializeServices();

      // Start all processors
      videoProcessor.current.startProcessing();
      audioAnalyzer.current.startAnalysis();
      anomalyDetector.current.startDetection();
      openAISummary.current.startAutoSummary();

      // Start metrics tracking
      metricsInterval.current = setInterval(() => {
        const metrics = videoProcessor.current.getPerformanceMetrics();
        setState(prev => ({ ...prev, performanceMetrics: metrics }));
      }, 1000);

      setState(prev => ({ ...prev, isMonitoring: true }));
    } catch (error) {
      console.error('[useMonitoring] Failed to start monitoring:', error);
      throw error;
    }
  }, [initializeServices]);

  /**
   * Stop monitoring
   */
  const stopMonitoring = useCallback(() => {
    videoProcessor.current.stopProcessing();
    audioAnalyzer.current.stopAnalysis();
    anomalyDetector.current.stopDetection();
    openAISummary.current.stopAutoSummary();

    if (metricsInterval.current) {
      clearInterval(metricsInterval.current);
      metricsInterval.current = null;
    }

    setState(prev => ({
      ...prev,
      isMonitoring: false,
      connectionStatus: 'disconnected',
    }));
  }, []);

  /**
   * Process a video frame
   */
  const processFrame = useCallback(async (frameData: string, cameraId: string) => {
    if (!state.isMonitoring) return;

    // Process frame through video processor
    const processedFrame = await videoProcessor.current.processFrame(frameData, cameraId);
    if (!processedFrame) return;

    // Get segmentation from SAM3
    const segmentation = await sam3Integration.current.segmentFrame(processedFrame);

    // Analyze for anomalies
    if (segmentation) {
      const anomalies = await anomalyDetector.current.analyzeFrame(processedFrame, segmentation);
      
      // Create alerts for significant anomalies
      for (const anomaly of anomalies) {
        await alertManager.current.createAlertFromAnomaly(anomaly);
        openAISummary.current.addAnomaly(anomaly);
      }

      if (anomalies.length > 0) {
        setState(prev => ({
          ...prev,
          recentAnomalies: [...anomalies, ...prev.recentAnomalies].slice(0, 50),
        }));
      }
    }
  }, [state.isMonitoring]);

  /**
   * Process audio data
   */
  const processAudio = useCallback(async (audioData: Float32Array) => {
    if (!state.isMonitoring) return;

    const classification = await audioAnalyzer.current.processAudioFrame(audioData);
    
    if (classification?.isAnomaly) {
      // Audio anomaly callbacks will handle alert creation
    }
  }, [state.isMonitoring]);

  /**
   * Switch active camera
   */
  const switchCamera = useCallback((cameraId: string) => {
    videoProcessor.current.switchCamera(cameraId);
    setState(prev => ({ ...prev, activeCameraId: cameraId }));
  }, []);

  /**
   * Add a camera
   */
  const addCamera = useCallback((camera: CameraConfig) => {
    videoProcessor.current.addCamera(camera);
    setState(prev => ({
      ...prev,
      cameras: [...prev.cameras, camera],
      activeCameraId: prev.activeCameraId || camera.id,
    }));
  }, []);

  /**
   * Remove a camera
   */
  const removeCamera = useCallback((cameraId: string) => {
    videoProcessor.current.removeCamera(cameraId);
    setState(prev => ({
      ...prev,
      cameras: prev.cameras.filter(c => c.id !== cameraId),
      activeCameraId: prev.activeCameraId === cameraId 
        ? (prev.cameras.length > 1 ? prev.cameras[0].id : null)
        : prev.activeCameraId,
    }));
  }, []);

  /**
   * Acknowledge an alert
   */
  const acknowledgeAlert = useCallback((alertId: string) => {
    alertManager.current.acknowledgeAlert(alertId);
    setState(prev => ({
      ...prev,
      alerts: prev.alerts.map(a =>
        a.id === alertId ? { ...a, acknowledged: true } : a
      ),
    }));
  }, []);

  /**
   * Acknowledge all alerts
   */
  const acknowledgeAllAlerts = useCallback(() => {
    alertManager.current.acknowledgeAllAlerts();
    setState(prev => ({
      ...prev,
      alerts: prev.alerts.map(a => ({ ...a, acknowledged: true })),
    }));
  }, []);

  /**
   * Generate a summary on demand
   */
  const generateSummary = useCallback(async (): Promise<EventSummary | null> => {
    const summary = await openAISummary.current.generateSummary();
    if (summary) {
      setState(prev => ({ ...prev, currentSummary: summary }));
    }
    return summary;
  }, []);

  /**
   * Set up service callbacks
   */
  useEffect(() => {
    // Alert callbacks
    const unsubAlert = alertManager.current.onAlert((alert) => {
      setState(prev => ({
        ...prev,
        alerts: [alert, ...prev.alerts].slice(0, 100),
      }));
      openAISummary.current.addAlert(alert);
    });

    // Anomaly callbacks
    const unsubAudioAnomaly = audioAnalyzer.current.onAnomaly(async (anomaly) => {
      await alertManager.current.createAlertFromAnomaly(anomaly);
      openAISummary.current.addAnomaly(anomaly);
      setState(prev => ({
        ...prev,
        recentAnomalies: [anomaly, ...prev.recentAnomalies].slice(0, 50),
      }));
    });

    // Summary callbacks
    const unsubSummary = openAISummary.current.onSummary((summary) => {
      setState(prev => ({ ...prev, currentSummary: summary }));
    });

    return () => {
      unsubAlert();
      unsubAudioAnomaly();
      unsubSummary();
    };
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      stopMonitoring();
    };
  }, [stopMonitoring]);

  return {
    state,
    startMonitoring,
    stopMonitoring,
    switchCamera,
    addCamera,
    removeCamera,
    acknowledgeAlert,
    acknowledgeAllAlerts,
    generateSummary,
    processFrame,
    processAudio,
  };
}

export default useMonitoring;
