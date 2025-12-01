/**
 * useCamera Hook
 * Handles camera permissions, configuration, and frame capture
 */

import { useCallback, useEffect, useState, useRef } from 'react';
import { VIDEO_CONFIG } from '../config';
import type { CameraConfig, Resolution } from '../types';

// Camera permission status
type PermissionStatus = 'undetermined' | 'granted' | 'denied';

interface UseCameraReturn {
  hasPermission: boolean;
  permissionStatus: PermissionStatus;
  isReady: boolean;
  currentCamera: CameraConfig | null;
  availableCameras: CameraConfig[];
  requestPermission: () => Promise<boolean>;
  switchCamera: () => void;
  setResolution: (resolution: Resolution) => void;
  captureFrame: () => Promise<string | null>;
  startCapture: (onFrame: (frameData: string) => void) => void;
  stopCapture: () => void;
}

/**
 * Custom hook for camera management
 */
export function useCamera(): UseCameraReturn {
  const [hasPermission, setHasPermission] = useState(false);
  const [permissionStatus, setPermissionStatus] = useState<PermissionStatus>('undetermined');
  const [isReady, setIsReady] = useState(false);
  const [currentCamera, setCurrentCamera] = useState<CameraConfig | null>(null);
  const [availableCameras, setAvailableCameras] = useState<CameraConfig[]>([]);
  
  const captureInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameCallback = useRef<((frameData: string) => void) | null>(null);

  /**
   * Initialize cameras
   */
  const initializeCameras = useCallback(() => {
    // Create default camera configurations
    const cameras: CameraConfig[] = [
      {
        id: 'back',
        name: 'Back Camera',
        position: 'back',
        resolution: VIDEO_CONFIG.defaultResolution,
        frameRate: VIDEO_CONFIG.targetFPS,
        enabled: true,
      },
      {
        id: 'front',
        name: 'Front Camera',
        position: 'front',
        resolution: VIDEO_CONFIG.defaultResolution,
        frameRate: VIDEO_CONFIG.targetFPS,
        enabled: true,
      },
    ];

    setAvailableCameras(cameras);
    setCurrentCamera(cameras[0]);
    setIsReady(true);
  }, []);

  /**
   * Request camera permission
   */
  const requestPermission = useCallback(async (): Promise<boolean> => {
    try {
      // In a real implementation, this would use expo-camera
      // const { status } = await Camera.requestCameraPermissionsAsync();
      // const granted = status === 'granted';
      
      // Simulate permission request
      const granted = true; // Would be actual permission result
      
      setHasPermission(granted);
      setPermissionStatus(granted ? 'granted' : 'denied');
      
      if (granted) {
        initializeCameras();
      }
      
      return granted;
    } catch (error) {
      console.error('[useCamera] Permission request failed:', error);
      setPermissionStatus('denied');
      return false;
    }
  }, [initializeCameras]);

  /**
   * Switch between front and back camera
   */
  const switchCamera = useCallback(() => {
    setCurrentCamera(current => {
      if (!current) return null;
      
      const nextPosition = current.position === 'back' ? 'front' : 'back';
      const nextCamera = availableCameras.find(c => c.position === nextPosition);
      return nextCamera || current;
    });
  }, [availableCameras]);

  /**
   * Set camera resolution
   */
  const setResolution = useCallback((resolution: Resolution) => {
    setCurrentCamera(current => {
      if (!current) return null;
      return { ...current, resolution };
    });
  }, []);

  /**
   * Capture a single frame
   */
  const captureFrame = useCallback(async (): Promise<string | null> => {
    if (!isReady || !currentCamera) {
      return null;
    }

    try {
      // In a real implementation, this would capture from expo-camera
      // const photo = await cameraRef.current?.takePictureAsync({
      //   base64: true,
      //   quality: VIDEO_CONFIG.jpegQuality,
      // });
      // return photo?.base64 || null;
      
      // Return placeholder for demo
      return 'placeholder_frame_data';
    } catch (error) {
      console.error('[useCamera] Frame capture failed:', error);
      return null;
    }
  }, [isReady, currentCamera]);

  /**
   * Start continuous frame capture
   */
  const startCapture = useCallback((onFrame: (frameData: string) => void) => {
    if (captureInterval.current) {
      return;
    }

    frameCallback.current = onFrame;
    const interval = 1000 / VIDEO_CONFIG.targetFPS;

    captureInterval.current = setInterval(async () => {
      const frame = await captureFrame();
      if (frame && frameCallback.current) {
        frameCallback.current(frame);
      }
    }, interval);
  }, [captureFrame]);

  /**
   * Stop continuous frame capture
   */
  const stopCapture = useCallback(() => {
    if (captureInterval.current) {
      clearInterval(captureInterval.current);
      captureInterval.current = null;
    }
    frameCallback.current = null;
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      stopCapture();
    };
  }, [stopCapture]);

  return {
    hasPermission,
    permissionStatus,
    isReady,
    currentCamera,
    availableCameras,
    requestPermission,
    switchCamera,
    setResolution,
    captureFrame,
    startCapture,
    stopCapture,
  };
}

export default useCamera;
