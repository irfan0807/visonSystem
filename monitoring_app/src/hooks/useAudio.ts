/**
 * useAudio Hook
 * Handles microphone permissions and audio capture
 */

import { useCallback, useEffect, useState, useRef } from 'react';
import { AUDIO_CONFIG } from '../config';
import type { AudioConfig } from '../types';

// Audio permission status
type PermissionStatus = 'undetermined' | 'granted' | 'denied';

interface UseAudioReturn {
  hasPermission: boolean;
  permissionStatus: PermissionStatus;
  isRecording: boolean;
  audioLevel: number;
  requestPermission: () => Promise<boolean>;
  startRecording: (onAudioData: (data: Float32Array) => void) => Promise<void>;
  stopRecording: () => void;
}

/**
 * Custom hook for audio capture management
 */
export function useAudio(): UseAudioReturn {
  const [hasPermission, setHasPermission] = useState(false);
  const [permissionStatus, setPermissionStatus] = useState<PermissionStatus>('undetermined');
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  
  const audioCallback = useRef<((data: Float32Array) => void) | null>(null);
  const recordingInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  /**
   * Request microphone permission
   */
  const requestPermission = useCallback(async (): Promise<boolean> => {
    try {
      // In a real implementation, this would use expo-av
      // const { status } = await Audio.requestPermissionsAsync();
      // const granted = status === 'granted';
      
      // Simulate permission request
      const granted = true; // Would be actual permission result
      
      setHasPermission(granted);
      setPermissionStatus(granted ? 'granted' : 'denied');
      
      return granted;
    } catch (error) {
      console.error('[useAudio] Permission request failed:', error);
      setPermissionStatus('denied');
      return false;
    }
  }, []);

  /**
   * Start audio recording
   */
  const startRecording = useCallback(async (onAudioData: (data: Float32Array) => void) => {
    if (!hasPermission) {
      const granted = await requestPermission();
      if (!granted) {
        throw new Error('Microphone permission denied');
      }
    }

    try {
      // In a real implementation, this would use expo-av
      // await Audio.setAudioModeAsync({
      //   allowsRecordingIOS: true,
      //   playsInSilentModeIOS: true,
      // });
      // const recording = new Audio.Recording();
      // await recording.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      // await recording.startAsync();

      audioCallback.current = onAudioData;
      setIsRecording(true);

      // Simulate audio data capture
      recordingInterval.current = setInterval(() => {
        // Generate simulated audio data
        const bufferSize = AUDIO_CONFIG.bufferSize;
        const simulatedData = new Float32Array(bufferSize);
        
        // Create random audio-like data
        for (let i = 0; i < bufferSize; i++) {
          simulatedData[i] = (Math.random() - 0.5) * 0.1;
        }

        // Calculate audio level
        let sum = 0;
        for (let i = 0; i < bufferSize; i++) {
          sum += simulatedData[i] * simulatedData[i];
        }
        const rms = Math.sqrt(sum / bufferSize);
        setAudioLevel(rms);

        // Send data to callback
        if (audioCallback.current) {
          audioCallback.current(simulatedData);
        }
      }, 100); // ~10 times per second

    } catch (error) {
      console.error('[useAudio] Failed to start recording:', error);
      throw error;
    }
  }, [hasPermission, requestPermission]);

  /**
   * Stop audio recording
   */
  const stopRecording = useCallback(() => {
    if (recordingInterval.current) {
      clearInterval(recordingInterval.current);
      recordingInterval.current = null;
    }

    audioCallback.current = null;
    setIsRecording(false);
    setAudioLevel(0);
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return {
    hasPermission,
    permissionStatus,
    isRecording,
    audioLevel,
    requestPermission,
    startRecording,
    stopRecording,
  };
}

export default useAudio;
