/**
 * MonitoringScreen
 * Main monitoring screen with camera view, alerts, and controls
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  StyleSheet,
  SafeAreaView,
  StatusBar as RNStatusBar,
  TouchableOpacity,
  Text,
  Modal,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { CameraView, AlertList, StatusBar, SummaryCard } from '../components';
import { useMonitoring, useCamera, useAudio } from '../hooks';
import type { Alert as AlertType, MaskData } from '../types';
import { VIDEO_CONFIG } from '../config';

type Tab = 'camera' | 'alerts' | 'summary';

export const MonitoringScreen: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('camera');
  const [showMasks, setShowMasks] = useState(true);
  const [masks, setMasks] = useState<MaskData[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<AlertType | null>(null);
  const [isSummaryLoading, setIsSummaryLoading] = useState(false);

  // Hooks
  const {
    state,
    startMonitoring,
    stopMonitoring,
    acknowledgeAlert,
    acknowledgeAllAlerts,
    generateSummary,
    processFrame,
    processAudio,
    switchCamera: monitoringSwitchCamera,
    addCamera,
  } = useMonitoring();

  const {
    hasPermission,
    currentCamera,
    requestPermission,
    switchCamera,
    startCapture,
    stopCapture,
  } = useCamera();

  const {
    hasPermission: hasAudioPermission,
    requestPermission: requestAudioPermission,
    startRecording,
    stopRecording,
  } = useAudio();

  /**
   * Request permissions and start monitoring
   */
  const handleStartMonitoring = useCallback(async () => {
    try {
      // Request camera permission
      if (!hasPermission) {
        const granted = await requestPermission();
        if (!granted) {
          Alert.alert('Permission Required', 'Camera permission is required for monitoring.');
          return;
        }
      }

      // Request audio permission
      if (!hasAudioPermission) {
        await requestAudioPermission();
      }

      // Start monitoring
      await startMonitoring();

      // Add default camera
      if (currentCamera) {
        addCamera(currentCamera);
      }

      // Start frame capture
      startCapture(async (frameData) => {
        if (currentCamera) {
          await processFrame(frameData, currentCamera.id);
        }
      });

      // Start audio recording
      try {
        await startRecording(processAudio);
      } catch (error) {
        console.warn('Audio recording not available:', error);
      }
    } catch (error) {
      console.error('Failed to start monitoring:', error);
      Alert.alert('Error', 'Failed to start monitoring. Please try again.');
    }
  }, [
    hasPermission,
    hasAudioPermission,
    requestPermission,
    requestAudioPermission,
    startMonitoring,
    currentCamera,
    addCamera,
    startCapture,
    processFrame,
    startRecording,
    processAudio,
  ]);

  /**
   * Stop monitoring
   */
  const handleStopMonitoring = useCallback(() => {
    stopMonitoring();
    stopCapture();
    stopRecording();
  }, [stopMonitoring, stopCapture, stopRecording]);

  /**
   * Toggle monitoring state
   */
  const handleToggleMonitoring = useCallback(() => {
    if (state.isMonitoring) {
      handleStopMonitoring();
    } else {
      handleStartMonitoring();
    }
  }, [state.isMonitoring, handleStartMonitoring, handleStopMonitoring]);

  /**
   * Handle camera switch
   */
  const handleSwitchCamera = useCallback(() => {
    switchCamera();
  }, [switchCamera]);

  /**
   * Handle alert acknowledgment
   */
  const handleAcknowledgeAlert = useCallback((alertId: string) => {
    acknowledgeAlert(alertId);
  }, [acknowledgeAlert]);

  /**
   * Handle view alert details
   */
  const handleViewAlertDetails = useCallback((alert: AlertType) => {
    setSelectedAlert(alert);
  }, []);

  /**
   * Handle generate summary
   */
  const handleGenerateSummary = useCallback(async () => {
    setIsSummaryLoading(true);
    try {
      await generateSummary();
    } finally {
      setIsSummaryLoading(false);
    }
  }, [generateSummary]);

  /**
   * Render tab content
   */
  const renderTabContent = () => {
    switch (activeTab) {
      case 'camera':
        return (
          <CameraView
            camera={currentCamera}
            masks={masks}
            showMasks={showMasks}
            isRecording={state.isMonitoring}
            fps={state.performanceMetrics.fps}
            latencyMs={state.performanceMetrics.latencyMs}
            onSwitchCamera={handleSwitchCamera}
            onToggleRecording={handleToggleMonitoring}
          />
        );
      
      case 'alerts':
        return (
          <AlertList
            alerts={state.alerts}
            onAcknowledge={handleAcknowledgeAlert}
            onViewDetails={handleViewAlertDetails}
            onAcknowledgeAll={acknowledgeAllAlerts}
          />
        );
      
      case 'summary':
        return (
          <SummaryCard
            summary={state.currentSummary}
            onRefresh={handleGenerateSummary}
            isLoading={isSummaryLoading}
          />
        );
    }
  };

  /**
   * Render alert detail modal
   */
  const renderAlertModal = () => (
    <Modal
      visible={selectedAlert !== null}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={() => setSelectedAlert(null)}
    >
      <SafeAreaView style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>Alert Details</Text>
          <TouchableOpacity onPress={() => setSelectedAlert(null)}>
            <Ionicons name="close" size={28} color="#fff" />
          </TouchableOpacity>
        </View>
        
        {selectedAlert && (
          <View style={styles.modalContent}>
            <Text style={styles.alertDetailTitle}>{selectedAlert.title}</Text>
            <Text style={styles.alertDetailMessage}>{selectedAlert.message}</Text>
            
            <View style={styles.alertDetailRow}>
              <Text style={styles.alertDetailLabel}>Severity:</Text>
              <Text style={styles.alertDetailValue}>{selectedAlert.severity}</Text>
            </View>
            
            <View style={styles.alertDetailRow}>
              <Text style={styles.alertDetailLabel}>Source:</Text>
              <Text style={styles.alertDetailValue}>{selectedAlert.source}</Text>
            </View>
            
            <View style={styles.alertDetailRow}>
              <Text style={styles.alertDetailLabel}>Time:</Text>
              <Text style={styles.alertDetailValue}>
                {new Date(selectedAlert.timestamp).toLocaleString()}
              </Text>
            </View>
            
            {!selectedAlert.acknowledged && (
              <TouchableOpacity
                style={styles.acknowledgeButton}
                onPress={() => {
                  handleAcknowledgeAlert(selectedAlert.id);
                  setSelectedAlert(null);
                }}
              >
                <Text style={styles.acknowledgeButtonText}>Acknowledge Alert</Text>
              </TouchableOpacity>
            )}
          </View>
        )}
      </SafeAreaView>
    </Modal>
  );

  return (
    <SafeAreaView style={styles.container}>
      <RNStatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
      
      {/* Status Bar */}
      <StatusBar
        connectionStatus={state.connectionStatus}
        isMonitoring={state.isMonitoring}
        metrics={state.performanceMetrics}
        cameraCount={state.cameras.length || 1}
        alertCount={state.alerts.filter(a => !a.acknowledged).length}
      />

      {/* Tab Content */}
      <View style={styles.content}>
        {renderTabContent()}
      </View>

      {/* Tab Bar */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tabItem, activeTab === 'camera' && styles.tabItemActive]}
          onPress={() => setActiveTab('camera')}
        >
          <Ionicons
            name="videocam"
            size={24}
            color={activeTab === 'camera' ? '#4ECDC4' : '#666'}
          />
          <Text style={[styles.tabLabel, activeTab === 'camera' && styles.tabLabelActive]}>
            Camera
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tabItem, activeTab === 'alerts' && styles.tabItemActive]}
          onPress={() => setActiveTab('alerts')}
        >
          <View>
            <Ionicons
              name="notifications"
              size={24}
              color={activeTab === 'alerts' ? '#4ECDC4' : '#666'}
            />
            {state.alerts.filter(a => !a.acknowledged).length > 0 && (
              <View style={styles.tabBadge}>
                <Text style={styles.tabBadgeText}>
                  {state.alerts.filter(a => !a.acknowledged).length}
                </Text>
              </View>
            )}
          </View>
          <Text style={[styles.tabLabel, activeTab === 'alerts' && styles.tabLabelActive]}>
            Alerts
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tabItem, activeTab === 'summary' && styles.tabItemActive]}
          onPress={() => setActiveTab('summary')}
        >
          <Ionicons
            name="analytics"
            size={24}
            color={activeTab === 'summary' ? '#4ECDC4' : '#666'}
          />
          <Text style={[styles.tabLabel, activeTab === 'summary' && styles.tabLabelActive]}>
            Summary
          </Text>
        </TouchableOpacity>
      </View>

      {/* Settings/Masks Toggle */}
      <TouchableOpacity
        style={styles.settingsButton}
        onPress={() => setShowMasks(!showMasks)}
      >
        <Ionicons
          name={showMasks ? 'eye' : 'eye-off'}
          size={24}
          color="#fff"
        />
      </TouchableOpacity>

      {/* Alert Detail Modal */}
      {renderAlertModal()}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  content: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#2a2a4e',
    borderTopWidth: 1,
    borderTopColor: '#3a3a5e',
    paddingBottom: 20,
    paddingTop: 8,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 8,
  },
  tabItemActive: {
    // Active state styles if needed
  },
  tabLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  tabLabelActive: {
    color: '#4ECDC4',
  },
  tabBadge: {
    position: 'absolute',
    top: -4,
    right: -8,
    backgroundColor: '#F44336',
    borderRadius: 10,
    minWidth: 18,
    height: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  tabBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  settingsButton: {
    position: 'absolute',
    top: 80,
    right: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a4e',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  modalContent: {
    padding: 16,
  },
  alertDetailTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  alertDetailMessage: {
    fontSize: 16,
    color: '#ccc',
    marginBottom: 24,
    lineHeight: 24,
  },
  alertDetailRow: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  alertDetailLabel: {
    fontSize: 14,
    color: '#666',
    width: 80,
  },
  alertDetailValue: {
    fontSize: 14,
    color: '#fff',
    flex: 1,
    textTransform: 'capitalize',
  },
  acknowledgeButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 24,
  },
  acknowledgeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default MonitoringScreen;
