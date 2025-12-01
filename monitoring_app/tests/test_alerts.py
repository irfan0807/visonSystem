"""
Tests for alert system module.
"""

import pytest
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.alerts import AlertManager, Alert


class TestAlert:
    """Tests for Alert dataclass."""
    
    def test_alert_creation(self):
        """Test alert creation with required fields."""
        alert = Alert(
            id="test_001",
            timestamp=datetime.now(),
            alert_type="motion",
            severity="high",
            message="Motion detected"
        )
        
        assert alert.id == "test_001"
        assert alert.alert_type == "motion"
        assert alert.severity == "high"
        assert alert.acknowledged is False
    
    def test_alert_with_data(self):
        """Test alert with additional data."""
        alert = Alert(
            id="test_002",
            timestamp=datetime.now(),
            alert_type="audio",
            severity="critical",
            message="Gunshot detected",
            data={'confidence': 0.95, 'location': 'zone_a'}
        )
        
        assert alert.data['confidence'] == 0.95
        assert alert.data['location'] == 'zone_a'
    
    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            id="test_003",
            timestamp=datetime.now(),
            alert_type="anomaly",
            severity="medium",
            message="Unusual activity"
        )
        
        data = alert.to_dict()
        
        assert data['id'] == "test_003"
        assert data['alert_type'] == "anomaly"
        assert 'timestamp' in data


class TestAlertManager:
    """Tests for AlertManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create an alert manager for testing."""
        return AlertManager(
            desktop_enabled=False,  # Disable for testing
            sound_enabled=False,
            cooldown_seconds=0.1
        )
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.desktop_enabled is False
        assert manager.sound_enabled is False
        assert len(manager.history) == 0
    
    def test_trigger_alert(self, manager):
        """Test triggering an alert."""
        alert = manager.trigger_alert(
            alert_type="test",
            message="Test alert",
            severity="low"
        )
        
        assert alert is not None
        assert alert.alert_type == "test"
        assert alert.message == "Test alert"
        assert len(manager.history) == 1
    
    def test_alert_cooldown(self, manager):
        """Test alert cooldown mechanism."""
        # First alert
        alert1 = manager.trigger_alert(
            alert_type="motion",
            message="First motion"
        )
        
        # Immediate second alert should be rate-limited
        alert2 = manager.trigger_alert(
            alert_type="motion",
            message="Second motion"
        )
        
        assert alert1 is not None
        assert alert2 is None  # Rate limited
        
        # Wait for cooldown
        time.sleep(0.2)
        
        # Third alert should work
        alert3 = manager.trigger_alert(
            alert_type="motion",
            message="Third motion"
        )
        
        assert alert3 is not None
    
    def test_bypass_cooldown(self, manager):
        """Test bypassing cooldown."""
        alert1 = manager.trigger_alert(
            alert_type="critical",
            message="First"
        )
        
        alert2 = manager.trigger_alert(
            alert_type="critical",
            message="Second",
            bypass_cooldown=True
        )
        
        assert alert1 is not None
        assert alert2 is not None
    
    def test_alert_severity_levels(self, manager):
        """Test different severity levels."""
        severities = ['low', 'medium', 'high', 'critical']
        
        for sev in severities:
            alert = manager.trigger_alert(
                alert_type=f"test_{sev}",
                message=f"{sev} alert",
                severity=sev,
                bypass_cooldown=True
            )
            assert alert.severity == sev
    
    def test_get_history(self, manager):
        """Test alert history retrieval."""
        for i in range(5):
            manager.trigger_alert(
                alert_type=f"type_{i}",
                message=f"Alert {i}",
                bypass_cooldown=True
            )
        
        history = manager.get_history(limit=3)
        
        assert len(history) == 3
    
    def test_filter_history_by_type(self, manager):
        """Test filtering history by alert type."""
        manager.trigger_alert(alert_type="motion", message="M1", bypass_cooldown=True)
        manager.trigger_alert(alert_type="audio", message="A1", bypass_cooldown=True)
        manager.trigger_alert(alert_type="motion", message="M2", bypass_cooldown=True)
        
        motion_alerts = manager.get_history(alert_type="motion")
        audio_alerts = manager.get_history(alert_type="audio")
        
        assert len(motion_alerts) == 2
        assert len(audio_alerts) == 1
    
    def test_filter_history_by_severity(self, manager):
        """Test filtering history by severity."""
        manager.trigger_alert(alert_type="t1", message="Low", severity="low", bypass_cooldown=True)
        manager.trigger_alert(alert_type="t2", message="High", severity="high", bypass_cooldown=True)
        manager.trigger_alert(alert_type="t3", message="Critical", severity="critical", bypass_cooldown=True)
        
        high_alerts = manager.get_history(severity="high")
        
        assert len(high_alerts) == 1
        assert high_alerts[0].severity == "high"
    
    def test_acknowledge_alert(self, manager):
        """Test acknowledging an alert."""
        alert = manager.trigger_alert(
            alert_type="test",
            message="To acknowledge"
        )
        
        assert alert.acknowledged is False
        
        result = manager.acknowledge_alert(alert.id)
        
        assert result is True
        assert alert.acknowledged is True
    
    def test_acknowledge_nonexistent(self, manager):
        """Test acknowledging non-existent alert."""
        result = manager.acknowledge_alert("nonexistent_id")
        assert result is False
    
    def test_callback_mechanism(self, manager):
        """Test alert callbacks."""
        callback_received = []
        
        def test_callback(alert):
            callback_received.append(alert)
        
        manager.add_callback(test_callback)
        
        manager.trigger_alert(
            alert_type="callback_test",
            message="Testing callbacks"
        )
        
        assert len(callback_received) == 1
        assert callback_received[0].alert_type == "callback_test"
    
    def test_get_stats(self, manager):
        """Test statistics retrieval."""
        manager.trigger_alert(alert_type="t1", message="M1", severity="high", bypass_cooldown=True)
        manager.trigger_alert(alert_type="t2", message="M2", severity="critical", bypass_cooldown=True)
        
        stats = manager.get_stats()
        
        assert stats['total_alerts'] == 2
        assert stats['critical_count'] == 1
        assert stats['high_count'] == 1
    
    def test_start_stop(self, manager):
        """Test manager start/stop."""
        manager.start()
        assert manager._running is True
        
        manager.stop()
        assert manager._running is False


class TestAlertManagerWorker:
    """Tests for AlertManager worker thread."""
    
    @pytest.fixture
    def manager_with_worker(self):
        """Create manager with worker thread."""
        manager = AlertManager(
            desktop_enabled=False,
            sound_enabled=False
        )
        manager.start()
        yield manager
        manager.stop()
    
    def test_worker_processes_alerts(self, manager_with_worker):
        """Test that worker processes alerts from queue."""
        alert = manager_with_worker.trigger_alert(
            alert_type="worker_test",
            message="Testing worker"
        )
        
        # Give worker time to process
        time.sleep(0.1)
        
        assert alert is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
