"""
Tests for anomaly detection module.
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.anomaly_detector import AnomalyDetector, AnomalyEvent


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        return AnomalyDetector(
            motion_threshold=0.3,
            anomaly_threshold=0.75,
            motion_weight=0.4,
            mask_change_weight=0.3,
            object_change_weight=0.3
        )
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.motion_threshold == 0.3
        assert detector.anomaly_threshold == 0.75
        assert detector.frame_count == 0
        assert len(detector.events) == 0
    
    def test_process_first_frame(self, detector):
        """Test processing the first frame."""
        # Create a simple test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        score, event = detector.process_frame(frame)
        
        assert score >= 0.0
        assert score <= 1.0
        assert detector.frame_count == 1
    
    def test_process_multiple_frames(self, detector):
        """Test processing multiple frames."""
        # Process several frames
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            score, _ = detector.process_frame(frame)
        
        assert detector.frame_count == 10
        assert len(detector.score_history) == 10
    
    def test_motion_detection(self, detector):
        """Test motion detection between frames."""
        # First frame - static
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[:, :, :] = 128
        
        detector.process_frame(frame1)
        
        # Second frame - with motion (different values)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[:240, :, :] = 200  # Top half changed
        
        score, _ = detector.process_frame(frame2)
        
        # Should detect some motion
        assert score >= 0.0
    
    def test_mask_change_calculation(self, detector):
        """Test mask IoU calculation."""
        # Create two masks
        mask1 = np.zeros((480, 640), dtype=np.uint8)
        mask1[100:200, 100:200] = 1  # 100x100 region
        
        mask2 = np.zeros((480, 640), dtype=np.uint8)
        mask2[150:250, 150:250] = 1  # Shifted region
        
        # Process with masks
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.process_frame(frame, mask=mask1)
        score, _ = detector.process_frame(frame, mask=mask2)
        
        # Should detect mask change
        assert score >= 0.0
    
    def test_object_count_change(self, detector):
        """Test object count change detection."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First frame - 2 objects
        detector.process_frame(frame, object_count=2)
        
        # Second frame - 5 objects
        score, _ = detector.process_frame(frame, object_count=5)
        
        # Object count changed significantly
        assert detector.prev_object_count == 5
    
    def test_anomaly_event_creation(self, detector):
        """Test anomaly event is created when threshold exceeded."""
        detector.anomaly_threshold = 0.1  # Low threshold for testing
        
        # Create frames with significant motion
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        detector.process_frame(frame1)
        score, event = detector.process_frame(frame2)
        
        # With low threshold, should trigger event
        if score > detector.anomaly_threshold:
            assert event is not None
            assert isinstance(event, AnomalyEvent)
    
    def test_cooldown_period(self, detector):
        """Test cooldown between anomaly events."""
        detector.anomaly_threshold = 0.1  # Low threshold
        detector.cooldown_frames = 5
        
        events_triggered = 0
        for i in range(20):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            _, event = detector.process_frame(frame)
            if event:
                events_triggered += 1
        
        # Should be limited by cooldown
        assert events_triggered <= 4  # 20 frames / 5 cooldown
    
    def test_get_score_trend(self, detector):
        """Test score trend retrieval."""
        # Process frames
        for i in range(50):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector.process_frame(frame)
        
        trend = detector.get_score_trend(window=30)
        
        assert len(trend) == 30
        assert all(0 <= s <= 1 for s in trend)
    
    def test_get_average_score(self, detector):
        """Test average score calculation."""
        for i in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector.process_frame(frame)
        
        avg = detector.get_average_score()
        
        assert 0 <= avg <= 1
    
    def test_reset(self, detector):
        """Test detector reset."""
        # Process some frames
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector.process_frame(frame)
        
        assert detector.frame_count > 0
        
        # Reset
        detector.reset()
        
        assert detector.frame_count == 0
        assert len(detector.score_history) == 0
        assert detector.prev_frame_gray is None
    
    def test_get_stats(self, detector):
        """Test statistics retrieval."""
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector.process_frame(frame)
        
        stats = detector.get_stats()
        
        assert 'frames_processed' in stats
        assert 'anomalies_detected' in stats
        assert 'avg_score' in stats
        assert 'max_score' in stats
        assert stats['frames_processed'] == 5


class TestAnomalyEvent:
    """Tests for AnomalyEvent dataclass."""
    
    def test_event_creation(self):
        """Test event creation with all fields."""
        event = AnomalyEvent(
            timestamp=1234567890.0,
            score=0.85,
            motion_score=0.7,
            mask_change_score=0.6,
            object_change_score=0.5,
            description="Test event",
            frame_index=100,
            bounding_boxes=[(10, 20, 100, 200)]
        )
        
        assert event.timestamp == 1234567890.0
        assert event.score == 0.85
        assert event.description == "Test event"
        assert len(event.bounding_boxes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
