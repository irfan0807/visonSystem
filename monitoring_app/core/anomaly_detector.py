"""
Anomaly detection module using motion analysis and segmentation changes.
Calculates anomaly scores based on multiple factors.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ..utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly event."""
    timestamp: float
    score: float
    motion_score: float
    mask_change_score: float
    object_change_score: float
    description: str
    frame_index: int
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


class AnomalyDetector:
    """
    Anomaly detection based on motion, segmentation, and object tracking.
    
    Anomaly score formula:
    anomaly_score = motion_magnitude * (1 - iou_prev_mask) * object_count_change
    """
    
    def __init__(
        self,
        motion_threshold: float = 0.3,
        anomaly_threshold: float = 0.75,
        motion_weight: float = 0.4,
        mask_change_weight: float = 0.3,
        object_change_weight: float = 0.3,
        history_size: int = 30,
        cooldown_frames: int = 15
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            motion_threshold: Minimum motion to consider
            anomaly_threshold: Score threshold for anomaly trigger
            motion_weight: Weight for motion component
            mask_change_weight: Weight for mask change component
            object_change_weight: Weight for object count change
            history_size: Number of frames to keep for analysis
            cooldown_frames: Minimum frames between anomaly events
        """
        self.logger = get_logger("anomaly")
        
        # Thresholds and weights
        self.motion_threshold = motion_threshold
        self.anomaly_threshold = anomaly_threshold
        self.motion_weight = motion_weight
        self.mask_change_weight = mask_change_weight
        self.object_change_weight = object_change_weight
        
        # History
        self.history_size = history_size
        self.frame_history: deque = deque(maxlen=history_size)
        self.mask_history: deque = deque(maxlen=history_size)
        self.object_count_history: deque = deque(maxlen=history_size)
        self.score_history: deque = deque(maxlen=history_size * 2)
        
        # State
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None
        self.prev_object_count: int = 0
        self.frame_count: int = 0
        self.cooldown_frames = cooldown_frames
        self.frames_since_anomaly: int = cooldown_frames
        
        # Motion detection (background subtractor)
        if cv2:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        else:
            self.bg_subtractor = None
        
        # Events
        self.events: deque = deque(maxlen=100)
        
        # Stats
        self.stats = {
            'frames_processed': 0,
            'anomalies_detected': 0,
            'avg_score': 0.0,
            'max_score': 0.0
        }
    
    def process_frame(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        object_count: int = 0,
        detections: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, Optional[AnomalyEvent]]:
        """
        Process a frame and calculate anomaly score.
        
        Args:
            frame: Input frame (BGR)
            mask: Segmentation mask
            object_count: Number of detected objects
            detections: List of detection dictionaries
        
        Returns:
            Tuple of (anomaly_score, anomaly_event if triggered)
        """
        self.frame_count += 1
        self.frames_since_anomaly += 1
        self.stats['frames_processed'] += 1
        
        if cv2 is None:
            return 0.0, None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion score
        motion_score = self._calculate_motion_score(gray)
        
        # Calculate mask change score
        mask_change_score = self._calculate_mask_change(mask)
        
        # Calculate object count change
        object_change_score = self._calculate_object_change(object_count)
        
        # Combine scores
        anomaly_score = self._calculate_anomaly_score(
            motion_score, mask_change_score, object_change_score
        )
        
        # Update history
        self.score_history.append(anomaly_score)
        self.prev_frame_gray = gray
        self.prev_mask = mask
        self.prev_object_count = object_count
        
        # Update stats
        self.stats['avg_score'] = np.mean(list(self.score_history))
        self.stats['max_score'] = max(self.stats['max_score'], anomaly_score)
        
        # Check for anomaly
        event = None
        if anomaly_score >= self.anomaly_threshold and \
           self.frames_since_anomaly >= self.cooldown_frames:
            event = self._create_anomaly_event(
                anomaly_score, motion_score, mask_change_score, 
                object_change_score, detections
            )
            self.events.append(event)
            self.stats['anomalies_detected'] += 1
            self.frames_since_anomaly = 0
            self.logger.warning(f"Anomaly detected! Score: {anomaly_score:.2f}")
        
        return anomaly_score, event
    
    def _calculate_motion_score(self, gray: np.ndarray) -> float:
        """Calculate motion score using optical flow and background subtraction."""
        if self.prev_frame_gray is None:
            return 0.0
        
        # Frame difference
        diff = cv2.absdiff(gray, self.prev_frame_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_pixels = np.sum(thresh > 0) / thresh.size
        
        # Background subtraction
        if self.bg_subtractor:
            fg_mask = self.bg_subtractor.apply(gray)
            fg_ratio = np.sum(fg_mask > 0) / fg_mask.size
        else:
            fg_ratio = motion_pixels
        
        # Combine scores
        motion_score = min(1.0, (motion_pixels + fg_ratio) / 2 * 5)  # Scale up
        
        return motion_score
    
    def _calculate_mask_change(self, mask: Optional[np.ndarray]) -> float:
        """Calculate segmentation mask change (1 - IoU)."""
        if mask is None or self.prev_mask is None:
            return 0.0
        
        # Ensure same size
        if mask.shape != self.prev_mask.shape:
            return 0.5  # Moderate change if size changed
        
        # Calculate IoU
        intersection = np.logical_and(mask > 0, self.prev_mask > 0)
        union = np.logical_or(mask > 0, self.prev_mask > 0)
        
        if np.sum(union) == 0:
            return 0.0
        
        iou = np.sum(intersection) / np.sum(union)
        return 1.0 - iou  # Higher score = more change
    
    def _calculate_object_change(self, object_count: int) -> float:
        """Calculate object count change score."""
        if self.prev_object_count == 0 and object_count == 0:
            return 0.0
        
        # Calculate relative change
        max_count = max(self.prev_object_count, object_count, 1)
        change = abs(object_count - self.prev_object_count) / max_count
        
        return min(1.0, change)
    
    def _calculate_anomaly_score(
        self,
        motion_score: float,
        mask_change_score: float,
        object_change_score: float
    ) -> float:
        """
        Calculate combined anomaly score.
        
        Formula: motion * (1 - iou) * object_change with weights
        """
        # Weighted combination
        weighted_sum = (
            self.motion_weight * motion_score +
            self.mask_change_weight * mask_change_score +
            self.object_change_weight * object_change_score
        )
        
        # Apply non-linear scaling for better sensitivity
        if motion_score > self.motion_threshold:
            # Boost score when significant motion detected
            score = weighted_sum * (1 + motion_score * mask_change_score)
        else:
            score = weighted_sum
        
        return min(1.0, score)
    
    def _create_anomaly_event(
        self,
        score: float,
        motion_score: float,
        mask_change_score: float,
        object_change_score: float,
        detections: Optional[List[Dict[str, Any]]] = None
    ) -> AnomalyEvent:
        """Create an anomaly event with description."""
        # Generate description
        descriptions = []
        
        if motion_score > 0.7:
            descriptions.append("High motion detected")
        elif motion_score > 0.4:
            descriptions.append("Significant movement")
        
        if mask_change_score > 0.5:
            descriptions.append("Scene composition changed")
        
        if object_change_score > 0.3:
            descriptions.append("Object count changed")
        
        description = "; ".join(descriptions) if descriptions else "Anomaly detected"
        
        # Extract bounding boxes from detections
        bboxes = []
        if detections:
            for det in detections:
                if 'bbox' in det:
                    bboxes.append(tuple(det['bbox']))
        
        return AnomalyEvent(
            timestamp=time.time(),
            score=score,
            motion_score=motion_score,
            mask_change_score=mask_change_score,
            object_change_score=object_change_score,
            description=description,
            frame_index=self.frame_count,
            bounding_boxes=bboxes
        )
    
    def get_recent_events(self, limit: int = 10) -> List[AnomalyEvent]:
        """Get recent anomaly events."""
        events = list(self.events)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_score_trend(self, window: int = 30) -> List[float]:
        """Get recent score history for trend analysis."""
        scores = list(self.score_history)
        return scores[-window:] if len(scores) > window else scores
    
    def get_average_score(self, window: int = 30) -> float:
        """Get average anomaly score over recent frames."""
        scores = self.get_score_trend(window)
        return np.mean(scores) if scores else 0.0
    
    def is_anomalous_period(self, threshold_ratio: float = 0.5) -> bool:
        """Check if currently in an anomalous period."""
        recent_scores = self.get_score_trend(30)
        if not recent_scores:
            return False
        
        high_scores = sum(1 for s in recent_scores if s > self.anomaly_threshold * 0.5)
        return high_scores / len(recent_scores) > threshold_ratio
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.frame_history.clear()
        self.mask_history.clear()
        self.object_count_history.clear()
        self.score_history.clear()
        self.prev_frame_gray = None
        self.prev_mask = None
        self.prev_object_count = 0
        self.frame_count = 0
        self.frames_since_anomaly = self.cooldown_frames
        
        if cv2 and self.bg_subtractor:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        
        self.logger.info("Anomaly detector reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            **self.stats,
            'current_trend': self.get_average_score(),
            'is_anomalous': self.is_anomalous_period(),
            'events_count': len(self.events)
        }
