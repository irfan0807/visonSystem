/**
 * StatusBar Component
 * Displays connection status, performance metrics, and system state
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { ConnectionStatus, PerformanceMetrics } from '../types';
import { PERFORMANCE_THRESHOLDS } from '../config';

interface StatusBarProps {
  connectionStatus: ConnectionStatus;
  isMonitoring: boolean;
  metrics: PerformanceMetrics;
  cameraCount: number;
  alertCount: number;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  connectionStatus,
  isMonitoring,
  metrics,
  cameraCount,
  alertCount,
}) => {
  /**
   * Get connection status color and icon
   */
  const getConnectionInfo = () => {
    switch (connectionStatus) {
      case 'connected':
        return { color: '#4CAF50', icon: 'cloud-done' as const, text: 'Connected' };
      case 'connecting':
        return { color: '#FF9800', icon: 'cloud-upload' as const, text: 'Connecting...' };
      case 'disconnected':
        return { color: '#9E9E9E', icon: 'cloud-offline' as const, text: 'Disconnected' };
      case 'error':
        return { color: '#F44336', icon: 'cloud-offline' as const, text: 'Error' };
      default:
        return { color: '#9E9E9E', icon: 'cloud-offline' as const, text: 'Unknown' };
    }
  };

  /**
   * Get FPS status color
   */
  const getFPSColor = () => {
    if (metrics.fps >= PERFORMANCE_THRESHOLDS.minFPS) return '#4CAF50';
    if (metrics.fps >= PERFORMANCE_THRESHOLDS.minFPS * 0.8) return '#FF9800';
    return '#F44336';
  };

  /**
   * Get latency status color
   */
  const getLatencyColor = () => {
    if (metrics.latencyMs <= PERFORMANCE_THRESHOLDS.maxLatency * 0.7) return '#4CAF50';
    if (metrics.latencyMs <= PERFORMANCE_THRESHOLDS.maxLatency) return '#FF9800';
    return '#F44336';
  };

  const connectionInfo = getConnectionInfo();

  return (
    <View style={styles.container}>
      {/* Connection Status */}
      <View style={styles.statusItem}>
        <Ionicons
          name={connectionInfo.icon}
          size={16}
          color={connectionInfo.color}
        />
        <Text style={[styles.statusText, { color: connectionInfo.color }]}>
          {connectionInfo.text}
        </Text>
      </View>

      {/* Monitoring Status */}
      <View style={styles.statusItem}>
        <View style={[
          styles.recordingDot,
          { backgroundColor: isMonitoring ? '#F44336' : '#666' }
        ]} />
        <Text style={styles.statusText}>
          {isMonitoring ? 'Monitoring' : 'Stopped'}
        </Text>
      </View>

      {/* FPS */}
      <View style={styles.statusItem}>
        <Ionicons name="speedometer" size={14} color={getFPSColor()} />
        <Text style={[styles.metricText, { color: getFPSColor() }]}>
          {metrics.fps.toFixed(0)} FPS
        </Text>
      </View>

      {/* Latency */}
      <View style={styles.statusItem}>
        <Ionicons name="time" size={14} color={getLatencyColor()} />
        <Text style={[styles.metricText, { color: getLatencyColor() }]}>
          {metrics.latencyMs.toFixed(0)}ms
        </Text>
      </View>

      {/* Camera Count */}
      <View style={styles.statusItem}>
        <Ionicons name="videocam" size={14} color="#fff" />
        <Text style={styles.metricText}>{cameraCount}</Text>
      </View>

      {/* Alert Count */}
      {alertCount > 0 && (
        <View style={styles.alertBadge}>
          <Ionicons name="alert-circle" size={14} color="#fff" />
          <Text style={styles.alertText}>{alertCount}</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a2e',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a4e',
  },
  statusItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 16,
  },
  statusText: {
    fontSize: 12,
    color: '#fff',
    marginLeft: 4,
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 4,
  },
  metricText: {
    fontSize: 11,
    color: '#fff',
    marginLeft: 4,
    fontWeight: '600',
  },
  alertBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F44336',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    marginLeft: 'auto',
  },
  alertText: {
    fontSize: 12,
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 4,
  },
});

export default StatusBar;
