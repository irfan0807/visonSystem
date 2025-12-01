/**
 * OpenAISummary Service
 * Generates AI-powered summaries of monitoring events
 * using OpenAI's GPT models
 */

import { SUMMARY_CONFIG } from '../config';
import type {
  SummaryConfig,
  EventSummary,
  KeyEvent,
  Alert,
  AnomalyResult,
  TimeRange,
  AnomalySeverity,
} from '../types';

// Callback types
type SummaryCallback = (summary: EventSummary) => void;
type ErrorCallback = (error: Error) => void;

// Event data for summary generation
interface EventData {
  alerts: Alert[];
  anomalies: AnomalyResult[];
  timeRange: TimeRange;
}

/**
 * OpenAISummary class for generating event summaries
 */
export class OpenAISummary {
  private config: SummaryConfig;
  private summaryCallbacks: Set<SummaryCallback> = new Set();
  private errorCallbacks: Set<ErrorCallback> = new Set();
  private summaries: EventSummary[] = [];
  private summaryInterval: ReturnType<typeof setInterval> | null = null;
  private pendingEvents: EventData;

  constructor(config: Partial<SummaryConfig> = {}) {
    this.config = { ...SUMMARY_CONFIG, ...config };
    this.pendingEvents = {
      alerts: [],
      anomalies: [],
      timeRange: { start: Date.now(), end: Date.now() },
    };
  }

  /**
   * Initialize the summary service
   */
  async initialize(): Promise<void> {
    try {
      if (!this.config.apiKey) {
        console.warn('[OpenAISummary] No API key configured');
      }

      console.log('[OpenAISummary] Initialized');
    } catch (error) {
      this.handleError(error as Error);
      throw error;
    }
  }

  /**
   * Start automatic summary generation
   */
  startAutoSummary(): void {
    if (!this.config.enabled) {
      console.warn('[OpenAISummary] Summary generation is disabled');
      return;
    }

    if (this.summaryInterval) {
      return;
    }

    this.pendingEvents = {
      alerts: [],
      anomalies: [],
      timeRange: { start: Date.now(), end: Date.now() },
    };

    this.summaryInterval = setInterval(async () => {
      if (this.pendingEvents.alerts.length > 0 || this.pendingEvents.anomalies.length > 0) {
        await this.generateSummary();
      }
    }, this.config.summaryInterval);

    console.log('[OpenAISummary] Auto summary started');
  }

  /**
   * Stop automatic summary generation
   */
  stopAutoSummary(): void {
    if (this.summaryInterval) {
      clearInterval(this.summaryInterval);
      this.summaryInterval = null;
    }
    console.log('[OpenAISummary] Auto summary stopped');
  }

  /**
   * Add an alert to pending events
   */
  addAlert(alert: Alert): void {
    this.pendingEvents.alerts.push(alert);
    this.pendingEvents.timeRange.end = Date.now();
  }

  /**
   * Add an anomaly to pending events
   */
  addAnomaly(anomaly: AnomalyResult): void {
    this.pendingEvents.anomalies.push(anomaly);
    this.pendingEvents.timeRange.end = Date.now();
  }

  /**
   * Generate a summary of pending events
   */
  async generateSummary(): Promise<EventSummary | null> {
    const events = { ...this.pendingEvents };
    
    // Reset pending events
    this.pendingEvents = {
      alerts: [],
      anomalies: [],
      timeRange: { start: Date.now(), end: Date.now() },
    };

    if (events.alerts.length === 0 && events.anomalies.length === 0) {
      return null;
    }

    try {
      // Build prompt
      const prompt = this.buildSummaryPrompt(events);

      // Call OpenAI API
      const summaryText = await this.callOpenAI(prompt);

      // Parse response and create summary
      const summary = this.createSummary(events, summaryText);

      // Store and notify
      this.summaries.push(summary);
      this.notifySummaryCallbacks(summary);

      return summary;
    } catch (error) {
      this.handleError(error as Error);
      
      // Create fallback summary
      return this.createFallbackSummary(events);
    }
  }

  /**
   * Generate an on-demand summary for specific events
   */
  async generateCustomSummary(
    alerts: Alert[],
    anomalies: AnomalyResult[],
    timeRange?: TimeRange
  ): Promise<EventSummary | null> {
    const range = timeRange || {
      start: Math.min(...alerts.map(a => a.timestamp), ...anomalies.map(a => a.timestamp)),
      end: Math.max(...alerts.map(a => a.timestamp), ...anomalies.map(a => a.timestamp)),
    };

    const events: EventData = { alerts, anomalies, timeRange: range };

    try {
      const prompt = this.buildSummaryPrompt(events);
      const summaryText = await this.callOpenAI(prompt);
      return this.createSummary(events, summaryText);
    } catch (error) {
      this.handleError(error as Error);
      return this.createFallbackSummary(events);
    }
  }

  /**
   * Call OpenAI API
   */
  private async callOpenAI(prompt: string): Promise<string> {
    if (!this.config.apiKey) {
      throw new Error('OpenAI API key not configured');
    }

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify({
          model: this.config.model,
          messages: [
            {
              role: 'system',
              content: 'You are a security monitoring assistant. Analyze the provided events and generate a concise, actionable summary. Focus on patterns, severity, and recommended actions.',
            },
            {
              role: 'user',
              content: prompt,
            },
          ],
          max_tokens: this.config.maxTokens,
          temperature: this.config.temperature,
        }),
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status}`);
      }

      const result = await response.json();
      return result.choices[0]?.message?.content || 'Unable to generate summary';
    } catch (error) {
      console.error('[OpenAISummary] API call failed:', error);
      throw error;
    }
  }

  /**
   * Build summary prompt from events
   */
  private buildSummaryPrompt(events: EventData): string {
    const alertSummary = events.alerts.map(alert => 
      `- [${alert.severity.toUpperCase()}] ${alert.title}: ${alert.message} (${new Date(alert.timestamp).toLocaleTimeString()})`
    ).join('\n');

    const anomalySummary = events.anomalies.map(anomaly =>
      `- [${anomaly.severity.toUpperCase()}] ${anomaly.type}: ${anomaly.description} (${new Date(anomaly.timestamp).toLocaleTimeString()})`
    ).join('\n');

    const duration = (events.timeRange.end - events.timeRange.start) / 1000 / 60; // minutes

    return `
Monitoring Summary Request
==========================

Time Period: ${new Date(events.timeRange.start).toLocaleString()} to ${new Date(events.timeRange.end).toLocaleString()} (${duration.toFixed(1)} minutes)

Alerts (${events.alerts.length}):
${alertSummary || 'No alerts'}

Anomalies Detected (${events.anomalies.length}):
${anomalySummary || 'No anomalies'}

Please provide:
1. A brief summary of the monitoring period
2. Key events that require attention
3. Any patterns observed
4. Recommended actions

Format the response in clear, concise paragraphs.
    `.trim();
  }

  /**
   * Create summary from API response
   */
  private createSummary(events: EventData, summaryText: string): EventSummary {
    // Extract key events
    const keyEvents = this.extractKeyEvents(events);

    // Extract recommendations (simple extraction)
    const recommendations = this.extractRecommendations(summaryText);

    return {
      id: this.generateSummaryId(),
      timestamp: Date.now(),
      timeRange: events.timeRange,
      summary: summaryText,
      keyEvents,
      alertCount: events.alerts.length,
      anomalyCount: events.anomalies.length,
      recommendations,
    };
  }

  /**
   * Create fallback summary when API fails
   */
  private createFallbackSummary(events: EventData): EventSummary {
    const keyEvents = this.extractKeyEvents(events);
    
    // Group by severity
    const criticalCount = [...events.alerts, ...events.anomalies].filter(
      e => ('severity' in e) && e.severity === 'critical'
    ).length;
    
    const highCount = [...events.alerts, ...events.anomalies].filter(
      e => ('severity' in e) && e.severity === 'high'
    ).length;

    let summaryText = `Monitoring Summary: ${events.alerts.length} alerts and ${events.anomalies.length} anomalies detected. `;
    
    if (criticalCount > 0) {
      summaryText += `${criticalCount} critical event(s) require immediate attention. `;
    }
    
    if (highCount > 0) {
      summaryText += `${highCount} high severity event(s) detected. `;
    }

    if (keyEvents.length > 0) {
      summaryText += `Key events include: ${keyEvents.map(e => e.type).join(', ')}.`;
    }

    const recommendations = [];
    if (criticalCount > 0) {
      recommendations.push('Review critical alerts immediately');
    }
    if (highCount > 0) {
      recommendations.push('Investigate high severity events');
    }
    if (events.anomalies.length > 5) {
      recommendations.push('Consider adjusting anomaly detection thresholds');
    }

    return {
      id: this.generateSummaryId(),
      timestamp: Date.now(),
      timeRange: events.timeRange,
      summary: summaryText,
      keyEvents,
      alertCount: events.alerts.length,
      anomalyCount: events.anomalies.length,
      recommendations,
    };
  }

  /**
   * Extract key events from event data
   */
  private extractKeyEvents(events: EventData): KeyEvent[] {
    const keyEvents: KeyEvent[] = [];

    // Add high/critical severity alerts
    events.alerts
      .filter(a => a.severity === 'critical' || a.severity === 'high')
      .forEach(alert => {
        keyEvents.push({
          timestamp: alert.timestamp,
          type: alert.type,
          description: alert.title,
          severity: alert.severity,
        });
      });

    // Add significant anomalies
    events.anomalies
      .filter(a => a.severity === 'critical' || a.severity === 'high')
      .forEach(anomaly => {
        keyEvents.push({
          timestamp: anomaly.timestamp,
          type: anomaly.type,
          description: anomaly.description,
          severity: anomaly.severity,
        });
      });

    // Sort by timestamp and limit
    return keyEvents
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 10);
  }

  /**
   * Extract recommendations from summary text
   */
  private extractRecommendations(text: string): string[] {
    const recommendations: string[] = [];
    
    // Simple extraction based on keywords
    const lines = text.split('\n');
    let inRecommendations = false;

    for (const line of lines) {
      const trimmedLine = line.trim();
      
      if (trimmedLine.toLowerCase().includes('recommend') || 
          trimmedLine.toLowerCase().includes('action')) {
        inRecommendations = true;
      }

      if (inRecommendations && trimmedLine.startsWith('-')) {
        recommendations.push(trimmedLine.substring(1).trim());
      }

      if (trimmedLine.startsWith('•') || trimmedLine.match(/^\d+\./)) {
        const content = trimmedLine.replace(/^[•\d.]+\s*/, '').trim();
        if (content.length > 0) {
          recommendations.push(content);
        }
      }
    }

    return recommendations.slice(0, 5);
  }

  /**
   * Subscribe to summaries
   */
  onSummary(callback: SummaryCallback): () => void {
    this.summaryCallbacks.add(callback);
    return () => this.summaryCallbacks.delete(callback);
  }

  /**
   * Subscribe to errors
   */
  onError(callback: ErrorCallback): () => void {
    this.errorCallbacks.add(callback);
    return () => this.errorCallbacks.delete(callback);
  }

  /**
   * Get recent summaries
   */
  getRecentSummaries(count: number = 5): EventSummary[] {
    return this.summaries
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, count);
  }

  /**
   * Get latest summary
   */
  getLatestSummary(): EventSummary | null {
    if (this.summaries.length === 0) return null;
    return this.summaries.reduce((latest, current) => 
      current.timestamp > latest.timestamp ? current : latest
    );
  }

  /**
   * Clear summary history
   */
  clearSummaryHistory(): void {
    this.summaries = [];
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SummaryConfig>): void {
    this.config = { ...this.config, ...config };
    console.log('[OpenAISummary] Configuration updated');
  }

  // Private methods

  private generateSummaryId(): string {
    return `summary_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private notifySummaryCallbacks(summary: EventSummary): void {
    this.summaryCallbacks.forEach(callback => {
      try {
        callback(summary);
      } catch (error) {
        console.error('[OpenAISummary] Callback error:', error);
      }
    });
  }

  private handleError(error: Error): void {
    console.error('[OpenAISummary] Error:', error.message);
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (callbackError) {
        console.error('[OpenAISummary] Error callback failed:', callbackError);
      }
    });
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stopAutoSummary();
    this.summaryCallbacks.clear();
    this.errorCallbacks.clear();
    this.summaries = [];
    console.log('[OpenAISummary] Disposed');
  }
}

// Singleton instance
let openAISummaryInstance: OpenAISummary | null = null;

/**
 * Get the OpenAISummary singleton instance
 */
export function getOpenAISummary(): OpenAISummary {
  if (!openAISummaryInstance) {
    openAISummaryInstance = new OpenAISummary();
  }
  return openAISummaryInstance;
}

/**
 * Reset the OpenAISummary instance
 */
export function resetOpenAISummary(): void {
  if (openAISummaryInstance) {
    openAISummaryInstance.dispose();
    openAISummaryInstance = null;
  }
}

export default OpenAISummary;
