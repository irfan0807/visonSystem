/**
 * Real-Time Monitoring App
 * Main Application Entry Point
 * 
 * Features:
 * - Camera/audio capture with 30 FPS target
 * - SAM3 segmentation via API integration
 * - Anomaly detection (visual and audio)
 * - Real-time mask overlays
 * - Push/email alerts
 * - OpenAI-powered summaries
 * - Multi-camera support
 * - iOS/Android ready
 */

import React, { useEffect, useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { MonitoringScreen } from './src/screens';

// App loading state type
type AppState = 'loading' | 'ready' | 'error';

/**
 * Loading Screen Component
 */
const LoadingScreen: React.FC = () => (
  <View style={styles.loadingContainer}>
    <ActivityIndicator size="large" color="#4ECDC4" />
    <Text style={styles.loadingText}>Initializing Monitoring System...</Text>
    <Text style={styles.loadingSubtext}>Setting up camera and audio capture</Text>
  </View>
);

/**
 * Error Screen Component
 */
const ErrorScreen: React.FC<{ error: string; onRetry: () => void }> = ({ error, onRetry }) => (
  <View style={styles.errorContainer}>
    <Text style={styles.errorIcon}>⚠️</Text>
    <Text style={styles.errorTitle}>Initialization Failed</Text>
    <Text style={styles.errorMessage}>{error}</Text>
    <Text style={styles.retryButton} onPress={onRetry}>
      Tap to Retry
    </Text>
  </View>
);

/**
 * Main App Component
 */
export default function App() {
  const [appState, setAppState] = useState<AppState>('loading');
  const [error, setError] = useState<string>('');

  /**
   * Initialize the application
   */
  const initializeApp = async () => {
    try {
      setAppState('loading');
      setError('');

      // Simulate initialization delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Initialization checks would go here
      // - Check device capabilities
      // - Verify API connectivity
      // - Load cached settings

      setAppState('ready');
    } catch (err) {
      console.error('[App] Initialization failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      setAppState('error');
    }
  };

  useEffect(() => {
    initializeApp();
  }, []);

  /**
   * Render based on app state
   */
  const renderContent = () => {
    switch (appState) {
      case 'loading':
        return <LoadingScreen />;
      case 'error':
        return <ErrorScreen error={error} onRetry={initializeApp} />;
      case 'ready':
        return <MonitoringScreen />;
    }
  };

  return (
    <GestureHandlerRootView style={styles.container}>
      <SafeAreaProvider>
        <StatusBar style="light" />
        {renderContent()}
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1a1a2e',
    padding: 20,
  },
  loadingText: {
    marginTop: 20,
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    textAlign: 'center',
  },
  loadingSubtext: {
    marginTop: 8,
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1a1a2e',
    padding: 20,
  },
  errorIcon: {
    fontSize: 64,
    marginBottom: 20,
  },
  errorTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#F44336',
    marginBottom: 12,
    textAlign: 'center',
  },
  errorMessage: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 20,
  },
  retryButton: {
    fontSize: 16,
    color: '#4ECDC4',
    fontWeight: '600',
    padding: 12,
  },
});
