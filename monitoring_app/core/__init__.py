"""
Core modules for video and audio processing.
"""

try:
    from .video_processor import VideoProcessor
    from .audio_processor import AudioProcessor
    from .anomaly_detector import AnomalyDetector
except ImportError:
    pass  # Allow partial imports

__all__ = ['VideoProcessor', 'AudioProcessor', 'AnomalyDetector']
