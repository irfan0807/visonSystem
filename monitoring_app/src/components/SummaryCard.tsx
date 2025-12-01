/**
 * SummaryCard Component
 * Displays AI-generated event summaries
 */

import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { EventSummary, KeyEvent, AnomalySeverity } from '../types';
import { UI_CONFIG } from '../config';

interface SummaryCardProps {
  summary: EventSummary | null;
  onRefresh: () => void;
  isLoading?: boolean;
}

interface KeyEventItemProps {
  event: KeyEvent;
}

/**
 * Get severity color
 */
const getSeverityColor = (severity: AnomalySeverity): string => {
  return UI_CONFIG.alertColors[severity];
};

/**
 * Format timestamp to readable string
 */
const formatTime = (timestamp: number): string => {
  return new Date(timestamp).toLocaleTimeString();
};

/**
 * Format time range
 */
const formatTimeRange = (start: number, end: number): string => {
  const startDate = new Date(start);
  const endDate = new Date(end);
  const durationMins = Math.round((end - start) / 60000);
  
  return `${startDate.toLocaleTimeString()} - ${endDate.toLocaleTimeString()} (${durationMins} min)`;
};

/**
 * Key Event Item Component
 */
const KeyEventItem: React.FC<KeyEventItemProps> = ({ event }) => {
  const severityColor = getSeverityColor(event.severity);
  
  return (
    <View style={styles.keyEventItem}>
      <View style={[styles.eventDot, { backgroundColor: severityColor }]} />
      <View style={styles.eventContent}>
        <Text style={styles.eventType}>{event.type}</Text>
        <Text style={styles.eventDescription} numberOfLines={2}>
          {event.description}
        </Text>
        <Text style={styles.eventTime}>{formatTime(event.timestamp)}</Text>
      </View>
    </View>
  );
};

/**
 * Summary Card Component
 */
export const SummaryCard: React.FC<SummaryCardProps> = ({
  summary,
  onRefresh,
  isLoading = false,
}) => {
  if (!summary) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <View style={styles.headerTitle}>
            <Ionicons name="analytics" size={24} color="#fff" />
            <Text style={styles.title}>AI Summary</Text>
          </View>
          <TouchableOpacity onPress={onRefresh} disabled={isLoading}>
            <Ionicons 
              name={isLoading ? 'hourglass' : 'refresh'} 
              size={24} 
              color="#4ECDC4" 
            />
          </TouchableOpacity>
        </View>
        
        <View style={styles.emptyContainer}>
          <Ionicons name="document-text-outline" size={48} color="#666" />
          <Text style={styles.emptyText}>No summary available</Text>
          <Text style={styles.emptySubtext}>
            Summaries are generated periodically or on-demand
          </Text>
          <TouchableOpacity style={styles.generateButton} onPress={onRefresh}>
            <Text style={styles.generateButtonText}>Generate Summary</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerTitle}>
          <Ionicons name="analytics" size={24} color="#fff" />
          <Text style={styles.title}>AI Summary</Text>
        </View>
        <TouchableOpacity onPress={onRefresh} disabled={isLoading}>
          <Ionicons 
            name={isLoading ? 'hourglass' : 'refresh'} 
            size={24} 
            color="#4ECDC4" 
          />
        </TouchableOpacity>
      </View>

      {/* Time Range */}
      <View style={styles.timeRangeContainer}>
        <Ionicons name="time-outline" size={16} color="#666" />
        <Text style={styles.timeRangeText}>
          {formatTimeRange(summary.timeRange.start, summary.timeRange.end)}
        </Text>
      </View>

      {/* Stats */}
      <View style={styles.statsContainer}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{summary.alertCount}</Text>
          <Text style={styles.statLabel}>Alerts</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{summary.anomalyCount}</Text>
          <Text style={styles.statLabel}>Anomalies</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{summary.keyEvents.length}</Text>
          <Text style={styles.statLabel}>Key Events</Text>
        </View>
      </View>

      {/* Summary Text */}
      <View style={styles.summarySection}>
        <Text style={styles.sectionTitle}>Summary</Text>
        <Text style={styles.summaryText}>{summary.summary}</Text>
      </View>

      {/* Key Events */}
      {summary.keyEvents.length > 0 && (
        <View style={styles.keyEventsSection}>
          <Text style={styles.sectionTitle}>Key Events</Text>
          {summary.keyEvents.map((event, index) => (
            <KeyEventItem key={index} event={event} />
          ))}
        </View>
      )}

      {/* Recommendations */}
      {summary.recommendations.length > 0 && (
        <View style={styles.recommendationsSection}>
          <Text style={styles.sectionTitle}>Recommendations</Text>
          {summary.recommendations.map((rec, index) => (
            <View key={index} style={styles.recommendationItem}>
              <Ionicons name="checkmark-circle" size={16} color="#4CAF50" />
              <Text style={styles.recommendationText}>{rec}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Generated Time */}
      <Text style={styles.generatedTime}>
        Generated: {new Date(summary.timestamp).toLocaleString()}
      </Text>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  headerTitle: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginLeft: 8,
  },
  timeRangeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  timeRangeText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 6,
  },
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: '#2a2a4e',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4ECDC4',
  },
  statLabel: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
  },
  statDivider: {
    width: 1,
    backgroundColor: '#3a3a5e',
    marginHorizontal: 16,
  },
  summarySection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 8,
  },
  summaryText: {
    fontSize: 14,
    color: '#ccc',
    lineHeight: 22,
  },
  keyEventsSection: {
    marginBottom: 20,
  },
  keyEventItem: {
    flexDirection: 'row',
    backgroundColor: '#2a2a4e',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  eventDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 4,
    marginRight: 12,
  },
  eventContent: {
    flex: 1,
  },
  eventType: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4ECDC4',
    textTransform: 'uppercase',
    marginBottom: 2,
  },
  eventDescription: {
    fontSize: 14,
    color: '#ccc',
  },
  eventTime: {
    fontSize: 11,
    color: '#666',
    marginTop: 4,
  },
  recommendationsSection: {
    marginBottom: 20,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  recommendationText: {
    fontSize: 14,
    color: '#ccc',
    marginLeft: 8,
    flex: 1,
  },
  generatedTime: {
    fontSize: 11,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
    textAlign: 'center',
  },
  generateButton: {
    marginTop: 20,
    backgroundColor: '#4ECDC4',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  generateButtonText: {
    color: '#1a1a2e',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default SummaryCard;
