/**
 * CameraView Component
 * Renders the camera feed with mask overlays
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
  Text,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { MaskOverlay } from './MaskOverlay';
import type { CameraConfig, MaskData } from '../types';

interface CameraViewProps {
  camera: CameraConfig | null;
  masks: MaskData[];
  showMasks: boolean;
  isRecording: boolean;
  fps: number;
  latencyMs: number;
  onSwitchCamera: () => void;
  onToggleRecording: () => void;
  onCaptureFrame?: () => void;
}

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const CAMERA_ASPECT_RATIO = 16 / 9;

export const CameraView: React.FC<CameraViewProps> = ({
  camera,
  masks,
  showMasks,
  isRecording,
  fps,
  latencyMs,
  onSwitchCamera,
  onToggleRecording,
  onCaptureFrame,
}) => {
  const cameraWidth = SCREEN_WIDTH;
  const cameraHeight = cameraWidth / CAMERA_ASPECT_RATIO;

  const getLatencyColor = () => {
    if (latencyMs < 100) return '#4CAF50';
    if (latencyMs < 150) return '#FF9800';
    return '#F44336';
  };

  return (
    <View style={styles.container}>
      {/* Camera Preview */}
      <View style={[styles.cameraContainer, { height: cameraHeight }]}>
        {/* Placeholder for actual camera - would use expo-camera */}
        <View style={styles.cameraPlaceholder}>
          <Ionicons name="videocam" size={64} color="#666" />
          <Text style={styles.placeholderText}>
            {camera ? `${camera.name} - ${camera.position}` : 'No Camera'}
          </Text>
        </View>

        {/* Mask Overlay */}
        {showMasks && masks.length > 0 && (
          <MaskOverlay
            masks={masks}
            frameWidth={camera?.resolution.width || 1920}
            frameHeight={camera?.resolution.height || 1080}
            viewWidth={cameraWidth}
            viewHeight={cameraHeight}
          />
        )}

        {/* Recording Indicator */}
        {isRecording && (
          <View style={styles.recordingIndicator}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>REC</Text>
          </View>
        )}

        {/* Performance Metrics */}
        <View style={styles.metricsContainer}>
          <Text style={styles.metricsText}>{fps.toFixed(1)} FPS</Text>
          <Text style={[styles.metricsText, { color: getLatencyColor() }]}>
            {latencyMs.toFixed(0)}ms
          </Text>
        </View>
      </View>

      {/* Camera Controls */}
      <View style={styles.controlsContainer}>
        <TouchableOpacity
          style={styles.controlButton}
          onPress={onSwitchCamera}
        >
          <Ionicons name="camera-reverse" size={28} color="#fff" />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.recordButton, isRecording && styles.recordButtonActive]}
          onPress={onToggleRecording}
        >
          <View style={[styles.recordButtonInner, isRecording && styles.recordButtonInnerActive]} />
        </TouchableOpacity>

        {onCaptureFrame && (
          <TouchableOpacity
            style={styles.controlButton}
            onPress={onCaptureFrame}
          >
            <Ionicons name="camera" size={28} color="#fff" />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  cameraContainer: {
    width: '100%',
    backgroundColor: '#1a1a2e',
    position: 'relative',
  },
  cameraPlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1a1a2e',
  },
  placeholderText: {
    color: '#666',
    marginTop: 16,
    fontSize: 16,
  },
  recordingIndicator: {
    position: 'absolute',
    top: 20,
    left: 20,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#F44336',
    marginRight: 8,
  },
  recordingText: {
    color: '#F44336',
    fontWeight: 'bold',
    fontSize: 14,
  },
  metricsContainer: {
    position: 'absolute',
    top: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  metricsText: {
    color: '#4CAF50',
    fontSize: 12,
    fontWeight: 'bold',
  },
  controlsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 20,
    backgroundColor: '#1a1a2e',
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 30,
  },
  recordButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#fff',
  },
  recordButtonActive: {
    backgroundColor: 'rgba(244, 67, 54, 0.3)',
  },
  recordButtonInner: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#F44336',
  },
  recordButtonInnerActive: {
    width: 30,
    height: 30,
    borderRadius: 4,
  },
});

export default CameraView;
