"""
Video processing module with SAM 3 integration for real-time segmentation.
Handles live webcam capture, object detection, and low-light enhancement.
"""

import os
import time
import threading
from pathlib import Path
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Generator
from collections import deque
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

try:
    from ..utils.logger import get_logger
    from ..utils.config import CameraConfig
    from .anomaly_detector import AnomalyDetector
except ImportError:
    from utils.logger import get_logger
    from utils.config import CameraConfig
    from core.anomaly_detector import AnomalyDetector


@dataclass
class Detection:
    """Represents a detected object."""
    id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    mask: Optional[np.ndarray] = None
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame: np.ndarray
    timestamp: float
    frame_index: int
    detections: List[Detection] = field(default_factory=list)
    combined_mask: Optional[np.ndarray] = None
    anomaly_score: float = 0.0
    processing_time_ms: float = 0.0
    fps: float = 0.0


class VideoProcessor:
    """
    Video processor with SAM 3 integration for real-time object segmentation.
    
    Features:
    - Live webcam capture at configurable FPS
    - SAM 3 real-time segmentation
    - Motion-based auto-prompting for SAM
    - Low-light enhancement (CLAHE)
    - Multi-camera support
    - Frame skipping under high CPU load
    """
    
    # Color palette for mask visualization
    MASK_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        target_fps: int = 30,
        enable_sam: bool = True,
        enable_enhancement: bool = True,
        enable_anomaly_detection: bool = True,
        auto_reconnect: bool = True,
        frame_buffer_size: int = 30,
        clip_duration: int = 10,
        clips_dir: Optional[Path] = None
    ):
        """
        Initialize the video processor.
        
        Args:
            camera_config: Camera configuration
            target_fps: Target frames per second
            enable_sam: Enable SAM 3 segmentation
            enable_enhancement: Enable low-light enhancement
            enable_anomaly_detection: Enable anomaly detection
            auto_reconnect: Auto-reconnect on camera disconnect
            frame_buffer_size: Size of frame buffer for clips
            clip_duration: Duration of event clips in seconds
            clips_dir: Directory to save video clips
        """
        self.logger = get_logger("video")
        
        # Configuration
        self.camera_config = camera_config or CameraConfig()
        self.target_fps = target_fps
        self.enable_sam = enable_sam
        self.enable_enhancement = enable_enhancement
        self.enable_anomaly_detection = enable_anomaly_detection
        self.auto_reconnect = auto_reconnect
        self.frame_buffer_size = frame_buffer_size
        self.clip_duration = clip_duration
        self.clips_dir = clips_dir or Path("data/clips")
        
        # Camera
        self.cap: Optional[Any] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        
        # Frame processing
        self.frame_queue: Queue = Queue(maxsize=frame_buffer_size)
        self.frame_buffer: deque = deque(maxlen=frame_buffer_size * target_fps)
        self.current_frame: Optional[np.ndarray] = None
        self.current_result: Optional[FrameResult] = None
        
        # Timing
        self.frame_count = 0
        self.start_time: Optional[float] = None
        self._frame_times: deque = deque(maxlen=30)
        
        # SAM model
        self.sam_model = None
        self.sam_predictor = None
        self._sam_initialized = False
        
        # Anomaly detection
        self.anomaly_detector: Optional[AnomalyDetector] = None
        if enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector()
        
        # CLAHE for low-light enhancement
        if cv2:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self.clahe = None
        
        # Callbacks
        self._frame_callbacks: List[Callable[[FrameResult], None]] = []
        self._anomaly_callbacks: List[Callable[[float, Dict], None]] = []
        
        # Stats
        self.stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'avg_fps': 0.0,
            'dropped_frames': 0,
            'reconnects': 0
        }
    
    def initialize_sam(self, model_type: str = "vit_b", checkpoint: Optional[str] = None) -> bool:
        """
        Initialize SAM 3 model for segmentation.
        
        Args:
            model_type: SAM model type (vit_b, vit_l, vit_h)
            checkpoint: Path to model checkpoint
        
        Returns:
            True if initialization successful
        """
        if not self.enable_sam:
            return False
        
        try:
            # Try to import SAM 3
            try:
                from sam3vision import SamAutomaticMaskGenerator, sam_model_registry
                self.logger.info("SAM 3 loaded from sam3vision")
            except ImportError:
                try:
                    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
                    self.logger.info("SAM loaded from segment_anything")
                except ImportError:
                    self.logger.warning("SAM not available - segmentation disabled")
                    return False
            
            # Load model
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            
            if checkpoint and os.path.exists(checkpoint):
                self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
            else:
                self.logger.warning("SAM checkpoint not found - using auto-download")
                return False
            
            self.sam_model.to(device)
            self.sam_predictor = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=16,  # Reduced for speed
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=100
            )
            
            self._sam_initialized = True
            self.logger.info(f"SAM initialized on {device}")
            return True
            
        except Exception as e:
            self.logger.error(f"SAM initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start video capture.
        
        Returns:
            True if capture started successfully
        """
        if self._running:
            return True
        
        if cv2 is None:
            self.logger.error("OpenCV not available")
            return False
        
        # Open camera
        if not self._open_camera():
            return False
        
        self._running = True
        self.start_time = time.time()
        
        # Start capture thread
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        self.logger.info(f"Video capture started: {self.camera_config.name}")
        return True
    
    def stop(self) -> None:
        """Stop video capture."""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=5.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.logger.info("Video capture stopped")
    
    def _open_camera(self) -> bool:
        """Open the camera device."""
        if cv2 is None:
            return False
        
        try:
            # Parse source (can be int or string URL)
            source = self.camera_config.source
            try:
                source = int(source)
            except ValueError:
                pass  # Keep as string (RTSP/file)
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera: {source}")
                return False
            
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(
                f"Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Camera open error: {e}")
            return False
    
    def _capture_loop(self) -> None:
        """Main capture loop running in a separate thread."""
        target_interval = 1.0 / self.target_fps
        
        while self._running:
            loop_start = time.time()
            
            # Capture frame
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Process frame
                    result = self._process_frame(frame)
                    
                    # Update state
                    self.current_frame = frame
                    self.current_result = result
                    
                    # Add to buffer
                    self.frame_buffer.append((frame.copy(), time.time()))
                    
                    # Notify callbacks
                    for callback in self._frame_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            self.logger.error(f"Frame callback error: {e}")
                else:
                    # Handle disconnect
                    self.logger.warning("Frame capture failed")
                    if self.auto_reconnect:
                        self._handle_reconnect()
            else:
                if self.auto_reconnect:
                    self._handle_reconnect()
                else:
                    break
            
            # Maintain target FPS
            elapsed = time.time() - loop_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
    
    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process a single frame."""
        start_time = time.time()
        self.frame_count += 1
        
        # Low-light enhancement
        if self.enable_enhancement:
            frame = self._enhance_frame(frame)
        
        # SAM segmentation
        detections = []
        combined_mask = None
        
        if self._sam_initialized and self.enable_sam:
            try:
                detections, combined_mask = self._run_sam_segmentation(frame)
            except Exception as e:
                self.logger.error(f"SAM error: {e}")
        
        # Anomaly detection
        anomaly_score = 0.0
        if self.anomaly_detector:
            object_count = len(detections)
            anomaly_score, event = self.anomaly_detector.process_frame(
                frame, combined_mask, object_count,
                [{'bbox': d.bbox} for d in detections]
            )
            
            # Notify anomaly callbacks
            if event:
                for callback in self._anomaly_callbacks:
                    try:
                        callback(anomaly_score, event.__dict__)
                    except Exception as e:
                        self.logger.error(f"Anomaly callback error: {e}")
        
        # Calculate timing
        processing_time = (time.time() - start_time) * 1000
        self._frame_times.append(processing_time)
        
        # Update stats
        self.stats['frames_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        
        if len(self._frame_times) > 0:
            avg_time = np.mean(list(self._frame_times))
            self.stats['avg_fps'] = 1000.0 / avg_time if avg_time > 0 else 0
        
        return FrameResult(
            frame=frame,
            timestamp=time.time(),
            frame_index=self.frame_count,
            detections=detections,
            combined_mask=combined_mask,
            anomaly_score=anomaly_score,
            processing_time_ms=processing_time,
            fps=self.stats['avg_fps']
        )
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply low-light enhancement using CLAHE."""
        if cv2 is None or self.clahe is None:
            return frame
        
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _run_sam_segmentation(
        self, 
        frame: np.ndarray
    ) -> Tuple[List[Detection], np.ndarray]:
        """Run SAM segmentation on frame."""
        if not self._sam_initialized or self.sam_predictor is None:
            return [], np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = self.sam_predictor.generate(rgb)
        
        # Process results
        detections = []
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for i, mask_data in enumerate(masks[:10]):  # Limit to top 10 masks
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # x, y, w, h
            
            detection = Detection(
                id=i,
                label='object',
                confidence=mask_data.get('predicted_iou', 0.9),
                bbox=tuple(int(v) for v in bbox),
                mask=mask.astype(np.uint8),
                track_id=None
            )
            detections.append(detection)
            
            # Add to combined mask with unique ID
            combined_mask[mask > 0] = i + 1
        
        return detections, combined_mask
    
    def _handle_reconnect(self) -> None:
        """Handle camera reconnection."""
        self.logger.warning("Attempting camera reconnect...")
        
        if self.cap:
            self.cap.release()
        
        time.sleep(2.0)
        
        if self._open_camera():
            self.stats['reconnects'] += 1
            self.logger.info("Camera reconnected")
        else:
            self.logger.error("Camera reconnect failed")
    
    def add_frame_callback(self, callback: Callable[[FrameResult], None]) -> None:
        """Add callback for new frames."""
        self._frame_callbacks.append(callback)
    
    def add_anomaly_callback(self, callback: Callable[[float, Dict], None]) -> None:
        """Add callback for anomaly events."""
        self._anomaly_callbacks.append(callback)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_result(self) -> Optional[FrameResult]:
        """Get the latest processing result."""
        return self.current_result
    
    def get_frames(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding frames for streaming."""
        while self._running:
            if self.current_frame is not None:
                yield self.current_frame.copy()
            time.sleep(1.0 / self.target_fps)
    
    def render_frame(
        self,
        frame: np.ndarray,
        result: Optional[FrameResult] = None,
        show_masks: bool = True,
        show_bboxes: bool = True,
        show_info: bool = True,
        mask_alpha: float = 0.4
    ) -> np.ndarray:
        """
        Render frame with overlays.
        
        Args:
            frame: Input frame
            result: Processing result
            show_masks: Show segmentation masks
            show_bboxes: Show bounding boxes
            show_info: Show info overlay
            mask_alpha: Mask transparency
        
        Returns:
            Rendered frame
        """
        if cv2 is None:
            return frame
        
        output = frame.copy()
        
        if result:
            # Draw masks
            if show_masks and result.combined_mask is not None:
                mask_overlay = np.zeros_like(output)
                for i in range(1, int(result.combined_mask.max()) + 1):
                    color = self.MASK_COLORS[i % len(self.MASK_COLORS)]
                    mask_overlay[result.combined_mask == i] = color
                
                output = cv2.addWeighted(output, 1.0, mask_overlay, mask_alpha, 0)
            
            # Draw bounding boxes
            if show_bboxes:
                for det in result.detections:
                    x, y, w, h = det.bbox
                    color = self.MASK_COLORS[det.id % len(self.MASK_COLORS)]
                    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"{det.label} ({det.confidence:.2f})"
                    cv2.putText(
                        output, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )
            
            # Draw info overlay
            if show_info:
                info_text = [
                    f"FPS: {result.fps:.1f}",
                    f"Objects: {len(result.detections)}",
                    f"Anomaly: {result.anomaly_score:.2f}",
                    f"Processing: {result.processing_time_ms:.1f}ms"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(
                        output, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    y_offset += 25
        
        return output
    
    def save_clip(
        self,
        duration_before: int = 5,
        duration_after: int = 5,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Save a video clip from the buffer.
        
        Args:
            duration_before: Seconds before event
            duration_after: Seconds after event
            filename: Output filename
        
        Returns:
            Path to saved clip
        """
        if cv2 is None or len(self.frame_buffer) == 0:
            return None
        
        try:
            # Create output path
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clip_{timestamp}.mp4"
            
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.clips_dir / filename
            
            # Get frames from buffer
            current_time = time.time()
            frames_to_save = [
                (f, t) for f, t in self.frame_buffer
                if current_time - t <= duration_before + duration_after
            ]
            
            if not frames_to_save:
                return None
            
            # Create video writer
            first_frame = frames_to_save[0][0]
            height, width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path), fourcc, self.target_fps, (width, height)
            )
            
            for frame, _ in frames_to_save:
                writer.write(frame)
            
            writer.release()
            
            self.logger.info(f"Clip saved: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving clip: {e}")
            return None
    
    def capture_snapshot(self, filename: Optional[str] = None) -> Optional[Path]:
        """
        Capture and save a snapshot.
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved snapshot
        """
        if cv2 is None or self.current_frame is None:
            return None
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
            
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.clips_dir / filename
            
            cv2.imwrite(str(output_path), self.current_frame)
            
            self.logger.info(f"Snapshot saved: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving snapshot: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'camera_connected': self.cap is not None and self.cap.isOpened() if self.cap else False,
            'sam_enabled': self._sam_initialized,
            'buffer_size': len(self.frame_buffer)
        }
