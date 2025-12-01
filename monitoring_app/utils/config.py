"""
Application configuration module.
Supports YAML configuration files and environment variables.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CameraConfig:
    """Configuration for a camera source."""
    id: int = 0
    name: str = "Default Camera"
    source: str = "0"  # Can be device ID or RTSP URL
    width: int = 1280
    height: int = 720
    fps: int = 30
    enabled: bool = True


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    device_index: Optional[int] = None
    vad_mode: int = 2  # WebRTC VAD aggressiveness (0-3)
    noise_suppression: bool = True


@dataclass
class AlertConfig:
    """Alert system configuration."""
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_sender: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    
    sms_enabled: bool = False
    twilio_sid: str = ""
    twilio_token: str = ""
    twilio_from: str = ""
    sms_recipients: List[str] = field(default_factory=list)
    
    desktop_enabled: bool = True
    sound_enabled: bool = True


@dataclass
class AnomalyConfig:
    """Anomaly detection thresholds."""
    motion_threshold: float = 0.3
    anomaly_threshold: float = 0.75
    object_change_weight: float = 0.3
    motion_weight: float = 0.4
    mask_change_weight: float = 0.3
    
    # Audio classification thresholds
    scream_threshold: float = 0.85
    glass_break_threshold: float = 0.80
    gunshot_threshold: float = 0.90
    door_slam_threshold: float = 0.75
    speech_threshold: float = 0.60


@dataclass
class AIConfig:
    """AI/ML model configuration."""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    scene_summary_interval: int = 30  # seconds
    sam_model_type: str = "vit_b"  # vit_b, vit_l, vit_h
    use_gpu: bool = True


@dataclass
class Config:
    """Main application configuration."""
    
    # App settings
    app_name: str = "AI Vision Monitoring System"
    debug: bool = False
    demo_mode: bool = False
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    logs_dir: Path = field(default_factory=lambda: Path("data/logs"))
    clips_dir: Path = field(default_factory=lambda: Path("data/clips"))
    events_dir: Path = field(default_factory=lambda: Path("data/events"))
    
    # Performance
    target_fps: int = 30
    max_latency_ms: int = 150
    max_memory_gb: float = 4.0
    frame_skip_threshold: float = 0.8  # Skip frames if CPU > 80%
    
    # Sub-configurations
    cameras: List[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    audio: AudioConfig = field(default_factory=AudioConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from a dictionary."""
        config = cls()
        
        # Basic settings
        config.app_name = data.get('app_name', config.app_name)
        config.debug = data.get('debug', config.debug)
        config.demo_mode = data.get('demo_mode', config.demo_mode)
        
        # Paths
        if 'data_dir' in data:
            config.data_dir = Path(data['data_dir'])
        if 'models_dir' in data:
            config.models_dir = Path(data['models_dir'])
        if 'logs_dir' in data:
            config.logs_dir = Path(data['logs_dir'])
        if 'clips_dir' in data:
            config.clips_dir = Path(data['clips_dir'])
        if 'events_dir' in data:
            config.events_dir = Path(data['events_dir'])
        
        # Performance
        config.target_fps = data.get('target_fps', config.target_fps)
        config.max_latency_ms = data.get('max_latency_ms', config.max_latency_ms)
        config.max_memory_gb = data.get('max_memory_gb', config.max_memory_gb)
        
        # Cameras
        if 'cameras' in data:
            config.cameras = [
                CameraConfig(**cam) for cam in data['cameras']
            ]
        
        # Audio
        if 'audio' in data:
            config.audio = AudioConfig(**data['audio'])
        
        # Alerts
        if 'alerts' in data:
            config.alerts = AlertConfig(**data['alerts'])
        
        # Anomaly
        if 'anomaly' in data:
            config.anomaly = AnomalyConfig(**data['anomaly'])
        
        # AI
        if 'ai' in data:
            config.ai = AIConfig(**data['ai'])
        
        # Load from environment variables
        config._load_from_env()
        
        return config
    
    def _load_from_env(self) -> None:
        """Load sensitive values from environment variables."""
        # OpenAI
        if os.environ.get('OPENAI_API_KEY'):
            self.ai.openai_api_key = os.environ['OPENAI_API_KEY']
        
        # Email
        if os.environ.get('SMTP_PASSWORD'):
            self.alerts.email_password = os.environ['SMTP_PASSWORD']
        
        # Twilio
        if os.environ.get('TWILIO_SID'):
            self.alerts.twilio_sid = os.environ['TWILIO_SID']
        if os.environ.get('TWILIO_TOKEN'):
            self.alerts.twilio_token = os.environ['TWILIO_TOKEN']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'app_name': self.app_name,
            'debug': self.debug,
            'demo_mode': self.demo_mode,
            'data_dir': str(self.data_dir),
            'models_dir': str(self.models_dir),
            'logs_dir': str(self.logs_dir),
            'clips_dir': str(self.clips_dir),
            'events_dir': str(self.events_dir),
            'target_fps': self.target_fps,
            'max_latency_ms': self.max_latency_ms,
            'max_memory_gb': self.max_memory_gb,
            'cameras': [
                {
                    'id': cam.id,
                    'name': cam.name,
                    'source': cam.source,
                    'width': cam.width,
                    'height': cam.height,
                    'fps': cam.fps,
                    'enabled': cam.enabled
                }
                for cam in self.cameras
            ],
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'chunk_size': self.audio.chunk_size,
                'channels': self.audio.channels,
                'device_index': self.audio.device_index,
                'vad_mode': self.audio.vad_mode,
                'noise_suppression': self.audio.noise_suppression
            },
            'alerts': {
                'email_enabled': self.alerts.email_enabled,
                'desktop_enabled': self.alerts.desktop_enabled,
                'sms_enabled': self.alerts.sms_enabled
            },
            'anomaly': {
                'motion_threshold': self.anomaly.motion_threshold,
                'anomaly_threshold': self.anomaly.anomaly_threshold
            },
            'ai': {
                'openai_model': self.ai.openai_model,
                'scene_summary_interval': self.ai.scene_summary_interval,
                'sam_model_type': self.ai.sam_model_type,
                'use_gpu': self.ai.use_gpu
            }
        }
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to a YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, 
                         self.clips_dir, self.events_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = Config()
