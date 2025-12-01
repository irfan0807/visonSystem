/**
 * AlertManager Service
 * Handles push notifications, email alerts, and in-app notifications
 * for the real-time monitoring system
 */

import { ALERT_CONFIG, API_ENDPOINTS, STORAGE_KEYS, UI_CONFIG } from '../config';
import type {
  Alert,
  AlertConfig,
  AlertMetadata,
  AlertType,
  AnomalyResult,
  AnomalySeverity,
  PushNotificationConfig,
} from '../types';

// Callback types
type AlertCallback = (alert: Alert) => void;
type ErrorCallback = (error: Error) => void;

// Alert queue item
interface QueuedAlert {
  alert: Alert;
  retryCount: number;
  maxRetries: number;
}

/**
 * AlertManager class for managing notifications and alerts
 */
export class AlertManager {
  private config: AlertConfig;
  private pushConfig: PushNotificationConfig;
  private alertCallbacks: Set<AlertCallback> = new Set();
  private errorCallbacks: Set<ErrorCallback> = new Set();
  private alerts: Alert[] = [];
  private alertQueue: QueuedAlert[] = [];
  private lastAlertTime: Map<string, number> = new Map();
  private isProcessingQueue: boolean = false;
  private queueProcessorInterval: ReturnType<typeof setInterval> | null = null;
  private expoPushToken: string | null = null;

  // Severity priority for filtering
  private readonly SEVERITY_PRIORITY: Record<AnomalySeverity, number> = {
    low: 1,
    medium: 2,
    high: 3,
    critical: 4,
  };

  constructor(config: Partial<AlertConfig> = {}, pushConfig: Partial<PushNotificationConfig> = {}) {
    this.config = { ...ALERT_CONFIG, ...config };
    this.pushConfig = {
      sound: true,
      badge: true,
      vibrate: true,
      ...pushConfig,
    };
  }

  /**
   * Initialize the alert manager
   */
  async initialize(): Promise<void> {
    try {
      // Start queue processor
      this.startQueueProcessor();

      console.log('[AlertManager] Initialized');
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Register for push notifications and get Expo push token
   */
  async registerForPushNotifications(): Promise<string | null> {
    try {
      // In a real implementation, this would use expo-notifications
      // const { status: existingStatus } = await Notifications.getPermissionsAsync();
      // let finalStatus = existingStatus;
      // if (existingStatus !== 'granted') {
      //   const { status } = await Notifications.requestPermissionsAsync();
      //   finalStatus = status;
      // }
      // if (finalStatus !== 'granted') {
      //   console.warn('[AlertManager] Push notification permissions not granted');
      //   return null;
      // }
      // const token = await Notifications.getExpoPushTokenAsync();
      // this.expoPushToken = token.data;

      // Placeholder for demo
      this.expoPushToken = 'ExponentPushToken[PLACEHOLDER]';
      this.pushConfig.expoPushToken = this.expoPushToken;

      console.log('[AlertManager] Registered for push notifications');
      return this.expoPushToken;
    } catch (error) {
      console.error('[AlertManager] Failed to register for push notifications:', error);
      return null;
    }
  }

  /**
   * Create and send an alert from an anomaly
   */
  async createAlertFromAnomaly(anomaly: AnomalyResult, imageUrl?: string): Promise<Alert | null> {
    // Check severity threshold
    if (!this.meetsSeverityThreshold(anomaly.severity)) {
      console.log('[AlertManager] Anomaly below severity threshold:', anomaly.severity);
      return null;
    }

    // Check cooldown
    const cooldownKey = `${anomaly.type}_${anomaly.source}`;
    if (!this.checkCooldown(cooldownKey)) {
      console.log('[AlertManager] Alert in cooldown period');
      return null;
    }

    const alert: Alert = {
      id: this.generateAlertId(),
      timestamp: Date.now(),
      type: 'in_app',
      severity: anomaly.severity,
      title: this.getAlertTitle(anomaly),
      message: anomaly.description,
      source: anomaly.source,
      acknowledged: false,
      metadata: {
        anomalyId: anomaly.id,
        imageUrl,
        ...anomaly.metadata as AlertMetadata,
      },
    };

    // Add to alerts list
    this.alerts.push(alert);
    this.lastAlertTime.set(cooldownKey, Date.now());

    // Send notifications based on config
    await this.sendAlert(alert);

    // Notify callbacks
    this.notifyAlertCallbacks(alert);

    return alert;
  }

  /**
   * Create a custom alert
   */
  async createAlert(
    title: string,
    message: string,
    severity: AnomalySeverity,
    source: string,
    metadata: AlertMetadata = {}
  ): Promise<Alert> {
    const alert: Alert = {
      id: this.generateAlertId(),
      timestamp: Date.now(),
      type: 'in_app',
      severity,
      title,
      message,
      source,
      acknowledged: false,
      metadata,
    };

    this.alerts.push(alert);

    // Send notifications based on config
    await this.sendAlert(alert);

    // Notify callbacks
    this.notifyAlertCallbacks(alert);

    return alert;
  }

  /**
   * Send alert through configured channels
   */
  async sendAlert(alert: Alert): Promise<void> {
    const promises: Promise<void>[] = [];

    // Push notification
    if (this.config.pushEnabled && this.expoPushToken) {
      promises.push(this.sendPushNotification(alert));
    }

    // Email notification
    if (this.config.emailEnabled && this.config.emailRecipients.length > 0) {
      promises.push(this.sendEmailNotification(alert));
    }

    // SMS notification (if enabled)
    if (this.config.smsEnabled && this.config.smsRecipients.length > 0) {
      promises.push(this.sendSmsNotification(alert));
    }

    try {
      await Promise.allSettled(promises);
    } catch (error) {
      console.error('[AlertManager] Error sending alerts:', error);
      // Queue for retry
      this.queueAlert(alert);
    }
  }

  /**
   * Send push notification
   */
  private async sendPushNotification(alert: Alert): Promise<void> {
    if (!this.expoPushToken) {
      console.warn('[AlertManager] No push token available');
      return;
    }

    try {
      // In production, this would send via Expo push notification service
      // await Notifications.scheduleNotificationAsync({
      //   content: {
      //     title: alert.title,
      //     body: alert.message,
      //     data: { alertId: alert.id },
      //     sound: this.pushConfig.sound,
      //     badge: this.pushConfig.badge ? 1 : 0,
      //   },
      //   trigger: null, // Immediate
      // });

      console.log('[AlertManager] Push notification sent:', alert.title);
    } catch (error) {
      console.error('[AlertManager] Failed to send push notification:', error);
      throw error;
    }
  }

  /**
   * Send email notification
   */
  private async sendEmailNotification(alert: Alert): Promise<void> {
    try {
      const response = await fetch(API_ENDPOINTS.ALERTS_EMAIL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recipients: this.config.emailRecipients,
          subject: `[${alert.severity.toUpperCase()}] ${alert.title}`,
          body: this.formatEmailBody(alert),
          alertId: alert.id,
        }),
      });

      if (!response.ok) {
        throw new Error(`Email API error: ${response.status}`);
      }

      console.log('[AlertManager] Email notification sent:', alert.title);
    } catch (error) {
      console.error('[AlertManager] Failed to send email notification:', error);
      throw error;
    }
  }

  /**
   * Send SMS notification
   */
  private async sendSmsNotification(alert: Alert): Promise<void> {
    try {
      // SMS sending would be implemented with a service like Twilio
      console.log('[AlertManager] SMS notification queued:', alert.title);
    } catch (error) {
      console.error('[AlertManager] Failed to send SMS notification:', error);
      throw error;
    }
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      console.log('[AlertManager] Alert acknowledged:', alertId);
      return true;
    }
    return false;
  }

  /**
   * Acknowledge all alerts
   */
  acknowledgeAllAlerts(): void {
    this.alerts.forEach(alert => {
      alert.acknowledged = true;
    });
    console.log('[AlertManager] All alerts acknowledged');
  }

  /**
   * Get all alerts
   */
  getAlerts(): Alert[] {
    return [...this.alerts];
  }

  /**
   * Get unacknowledged alerts
   */
  getUnacknowledgedAlerts(): Alert[] {
    return this.alerts.filter(a => !a.acknowledged);
  }

  /**
   * Get alerts by severity
   */
  getAlertsBySeverity(severity: AnomalySeverity): Alert[] {
    return this.alerts.filter(a => a.severity === severity);
  }

  /**
   * Get recent alerts
   */
  getRecentAlerts(count: number = 10): Alert[] {
    return this.alerts
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, count);
  }

  /**
   * Clear old alerts
   */
  clearOldAlerts(maxAgeMs: number = 24 * 60 * 60 * 1000): number {
    const cutoff = Date.now() - maxAgeMs;
    const initialCount = this.alerts.length;
    this.alerts = this.alerts.filter(a => a.timestamp >= cutoff);
    const removed = initialCount - this.alerts.length;
    console.log('[AlertManager] Cleared', removed, 'old alerts');
    return removed;
  }

  /**
   * Clear all alerts
   */
  clearAllAlerts(): void {
    this.alerts = [];
    console.log('[AlertManager] All alerts cleared');
  }

  /**
   * Subscribe to new alerts
   */
  onAlert(callback: AlertCallback): () => void {
    this.alertCallbacks.add(callback);
    return () => this.alertCallbacks.delete(callback);
  }

  /**
   * Subscribe to errors
   */
  onError(callback: ErrorCallback): () => void {
    this.errorCallbacks.add(callback);
    return () => this.errorCallbacks.delete(callback);
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<AlertConfig>): void {
    this.config = { ...this.config, ...config };
    console.log('[AlertManager] Configuration updated');
  }

  /**
   * Add email recipient
   */
  addEmailRecipient(email: string): void {
    if (!this.config.emailRecipients.includes(email)) {
      this.config.emailRecipients.push(email);
    }
  }

  /**
   * Remove email recipient
   */
  removeEmailRecipient(email: string): void {
    this.config.emailRecipients = this.config.emailRecipients.filter(e => e !== email);
  }

  /**
   * Get alert statistics
   */
  getStatistics(): {
    total: number;
    unacknowledged: number;
    bySeverity: Record<AnomalySeverity, number>;
  } {
    const bySeverity: Record<AnomalySeverity, number> = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };

    this.alerts.forEach(alert => {
      bySeverity[alert.severity]++;
    });

    return {
      total: this.alerts.length,
      unacknowledged: this.getUnacknowledgedAlerts().length,
      bySeverity,
    };
  }

  /**
   * Get alert color by severity
   */
  getAlertColor(severity: AnomalySeverity): string {
    return UI_CONFIG.alertColors[severity];
  }

  // Private methods

  private generateAlertId(): string {
    return `alert_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private meetsSeverityThreshold(severity: AnomalySeverity): boolean {
    return this.SEVERITY_PRIORITY[severity] >= this.SEVERITY_PRIORITY[this.config.severityThreshold];
  }

  private checkCooldown(key: string): boolean {
    const lastTime = this.lastAlertTime.get(key);
    if (!lastTime) return true;
    return Date.now() - lastTime >= this.config.cooldownMs;
  }

  private getAlertTitle(anomaly: AnomalyResult): string {
    const typeLabels: Record<string, string> = {
      motion_detected: 'Motion Detected',
      unusual_activity: 'Unusual Activity',
      object_detected: 'Object Detected',
      audio_anomaly: 'Audio Anomaly',
      crowd_formation: 'Crowd Formation',
      restricted_area_breach: 'Restricted Area Breach',
      equipment_malfunction: 'Equipment Malfunction',
    };

    return typeLabels[anomaly.type] || 'Alert';
  }

  private formatEmailBody(alert: Alert): string {
    return `
Monitoring System Alert
========================

Title: ${alert.title}
Severity: ${alert.severity.toUpperCase()}
Time: ${new Date(alert.timestamp).toLocaleString()}
Source: ${alert.source}

Message:
${alert.message}

Alert ID: ${alert.id}
    `.trim();
  }

  private queueAlert(alert: Alert): void {
    this.alertQueue.push({
      alert,
      retryCount: 0,
      maxRetries: 3,
    });
  }

  private startQueueProcessor(): void {
    if (this.queueProcessorInterval) return;

    this.queueProcessorInterval = setInterval(async () => {
      if (this.isProcessingQueue || this.alertQueue.length === 0) return;

      this.isProcessingQueue = true;

      const item = this.alertQueue[0];
      try {
        await this.sendAlert(item.alert);
        this.alertQueue.shift();
      } catch (error) {
        item.retryCount++;
        if (item.retryCount >= item.maxRetries) {
          console.error('[AlertManager] Max retries reached for alert:', item.alert.id);
          this.alertQueue.shift();
        }
      }

      this.isProcessingQueue = false;
    }, 5000); // Process queue every 5 seconds
  }

  private stopQueueProcessor(): void {
    if (this.queueProcessorInterval) {
      clearInterval(this.queueProcessorInterval);
      this.queueProcessorInterval = null;
    }
  }

  private notifyAlertCallbacks(alert: Alert): void {
    this.alertCallbacks.forEach(callback => {
      try {
        callback(alert);
      } catch (error) {
        console.error('[AlertManager] Alert callback error:', error);
      }
    });
  }

  private handleError(error: Error): void {
    console.error('[AlertManager] Error:', error.message);
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (callbackError) {
        console.error('[AlertManager] Error callback failed:', callbackError);
      }
    });
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stopQueueProcessor();
    this.alertCallbacks.clear();
    this.errorCallbacks.clear();
    this.alertQueue = [];
    console.log('[AlertManager] Disposed');
  }
}

// Singleton instance
let alertManagerInstance: AlertManager | null = null;

/**
 * Get the AlertManager singleton instance
 */
export function getAlertManager(): AlertManager {
  if (!alertManagerInstance) {
    alertManagerInstance = new AlertManager();
  }
  return alertManagerInstance;
}

/**
 * Reset the AlertManager instance
 */
export function resetAlertManager(): void {
  if (alertManagerInstance) {
    alertManagerInstance.dispose();
    alertManagerInstance = null;
  }
}

export default AlertManager;
