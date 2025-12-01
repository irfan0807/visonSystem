/**
 * AlertList Component
 * Displays a list of alerts with acknowledgment functionality
 */

import React, { useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { Alert, AnomalySeverity } from '../types';
import { UI_CONFIG } from '../config';

interface AlertListProps {
  alerts: Alert[];
  onAcknowledge: (alertId: string) => void;
  onViewDetails: (alert: Alert) => void;
  onAcknowledgeAll?: () => void;
}

interface AlertItemProps {
  alert: Alert;
  onAcknowledge: (alertId: string) => void;
  onViewDetails: (alert: Alert) => void;
}

/**
 * Get icon name based on alert type and source
 */
const getAlertIcon = (alert: Alert): keyof typeof Ionicons.glyphMap => {
  switch (alert.source) {
    case 'visual':
      return 'eye';
    case 'audio':
      return 'volume-high';
    case 'motion':
      return 'move';
    default:
      return 'alert-circle';
  }
};

/**
 * Get severity color
 */
const getSeverityColor = (severity: AnomalySeverity): string => {
  return UI_CONFIG.alertColors[severity];
};

/**
 * Format timestamp to readable string
 */
const formatTimestamp = (timestamp: number): string => {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
  return date.toLocaleDateString();
};

/**
 * Individual Alert Item
 */
const AlertItem: React.FC<AlertItemProps> = ({
  alert,
  onAcknowledge,
  onViewDetails,
}) => {
  const severityColor = getSeverityColor(alert.severity);
  const iconName = getAlertIcon(alert);

  return (
    <TouchableOpacity
      style={[
        styles.alertItem,
        alert.acknowledged && styles.alertItemAcknowledged,
      ]}
      onPress={() => onViewDetails(alert)}
      activeOpacity={0.7}
    >
      <View style={[styles.severityIndicator, { backgroundColor: severityColor }]} />
      
      <View style={styles.iconContainer}>
        <Ionicons name={iconName} size={24} color={severityColor} />
      </View>
      
      <View style={styles.contentContainer}>
        <View style={styles.headerRow}>
          <Text style={styles.alertTitle} numberOfLines={1}>
            {alert.title}
          </Text>
          <View style={[styles.severityBadge, { backgroundColor: severityColor }]}>
            <Text style={styles.severityText}>{alert.severity.toUpperCase()}</Text>
          </View>
        </View>
        
        <Text style={styles.alertMessage} numberOfLines={2}>
          {alert.message}
        </Text>
        
        <View style={styles.footerRow}>
          <Text style={styles.timestamp}>{formatTimestamp(alert.timestamp)}</Text>
          <Text style={styles.source}>{alert.source}</Text>
        </View>
      </View>
      
      {!alert.acknowledged && (
        <TouchableOpacity
          style={styles.acknowledgeButton}
          onPress={() => onAcknowledge(alert.id)}
        >
          <Ionicons name="checkmark-circle" size={24} color="#4CAF50" />
        </TouchableOpacity>
      )}
    </TouchableOpacity>
  );
};

/**
 * Alert List Component
 */
export const AlertList: React.FC<AlertListProps> = ({
  alerts,
  onAcknowledge,
  onViewDetails,
  onAcknowledgeAll,
}) => {
  const unacknowledgedCount = alerts.filter(a => !a.acknowledged).length;

  const renderItem = useCallback(
    ({ item }: { item: Alert }) => (
      <AlertItem
        alert={item}
        onAcknowledge={onAcknowledge}
        onViewDetails={onViewDetails}
      />
    ),
    [onAcknowledge, onViewDetails]
  );

  const keyExtractor = useCallback((item: Alert) => item.id, []);

  const renderHeader = () => (
    <View style={styles.listHeader}>
      <View style={styles.headerTitleRow}>
        <Ionicons name="notifications" size={24} color="#fff" />
        <Text style={styles.headerTitle}>Alerts</Text>
        <View style={styles.badgeContainer}>
          <Text style={styles.badgeText}>{alerts.length}</Text>
        </View>
      </View>
      
      {unacknowledgedCount > 0 && onAcknowledgeAll && (
        <TouchableOpacity
          style={styles.acknowledgeAllButton}
          onPress={onAcknowledgeAll}
        >
          <Text style={styles.acknowledgeAllText}>
            Acknowledge All ({unacknowledgedCount})
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );

  const renderEmpty = () => (
    <View style={styles.emptyContainer}>
      <Ionicons name="checkmark-circle" size={48} color="#4CAF50" />
      <Text style={styles.emptyText}>No alerts</Text>
      <Text style={styles.emptySubtext}>All systems normal</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={alerts}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        ListHeaderComponent={renderHeader}
        ListEmptyComponent={renderEmpty}
        contentContainerStyle={alerts.length === 0 ? styles.emptyList : undefined}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  listHeader: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a4e',
  },
  headerTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginLeft: 8,
  },
  badgeContainer: {
    backgroundColor: '#F44336',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 2,
    marginLeft: 8,
  },
  badgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  acknowledgeAllButton: {
    marginTop: 8,
    paddingVertical: 8,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(76, 175, 80, 0.2)',
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  acknowledgeAllText: {
    color: '#4CAF50',
    fontSize: 14,
    fontWeight: '600',
  },
  alertItem: {
    flexDirection: 'row',
    backgroundColor: '#2a2a4e',
    marginHorizontal: 16,
    marginVertical: 4,
    borderRadius: 12,
    overflow: 'hidden',
  },
  alertItemAcknowledged: {
    opacity: 0.6,
  },
  severityIndicator: {
    width: 4,
  },
  iconContainer: {
    width: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  contentContainer: {
    flex: 1,
    paddingVertical: 12,
    paddingRight: 8,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    flex: 1,
    marginRight: 8,
  },
  severityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  severityText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  alertMessage: {
    fontSize: 14,
    color: '#aaa',
    lineHeight: 20,
  },
  footerRow: {
    flexDirection: 'row',
    marginTop: 8,
  },
  timestamp: {
    fontSize: 12,
    color: '#666',
    marginRight: 16,
  },
  source: {
    fontSize: 12,
    color: '#666',
    textTransform: 'capitalize',
  },
  acknowledgeButton: {
    width: 50,
    justifyContent: 'center',
    alignItems: 'center',
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
  },
  emptyList: {
    flexGrow: 1,
  },
});

export default AlertList;
