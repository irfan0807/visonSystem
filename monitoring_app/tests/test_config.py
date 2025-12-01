"""
Tests for configuration module.
"""

import pytest
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import Config, CameraConfig, AudioConfig, AlertConfig, AnomalyConfig, AIConfig


class TestCameraConfig:
    """Tests for CameraConfig dataclass."""
    
    def test_default_values(self):
        """Test default camera configuration values."""
        config = CameraConfig()
        
        assert config.id == 0
        assert config.name == "Default Camera"
        assert config.source == "0"
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30
        assert config.enabled is True
    
    def test_custom_values(self):
        """Test custom camera configuration."""
        config = CameraConfig(
            id=1,
            name="Test Camera",
            source="rtsp://example.com/stream",
            width=1920,
            height=1080,
            fps=60,
            enabled=False
        )
        
        assert config.id == 1
        assert config.name == "Test Camera"
        assert config.source == "rtsp://example.com/stream"
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 60
        assert config.enabled is False


class TestAudioConfig:
    """Tests for AudioConfig dataclass."""
    
    def test_default_values(self):
        """Test default audio configuration values."""
        config = AudioConfig()
        
        assert config.sample_rate == 16000
        assert config.chunk_size == 1024
        assert config.channels == 1
        assert config.device_index is None
        assert config.vad_mode == 2
        assert config.noise_suppression is True


class TestAlertConfig:
    """Tests for AlertConfig dataclass."""
    
    def test_default_values(self):
        """Test default alert configuration values."""
        config = AlertConfig()
        
        assert config.email_enabled is False
        assert config.sms_enabled is False
        assert config.desktop_enabled is True
        assert config.sound_enabled is True
        assert config.email_recipients == []
        assert config.sms_recipients == []


class TestAnomalyConfig:
    """Tests for AnomalyConfig dataclass."""
    
    def test_default_thresholds(self):
        """Test default anomaly thresholds."""
        config = AnomalyConfig()
        
        assert config.motion_threshold == 0.3
        assert config.anomaly_threshold == 0.75
        assert config.scream_threshold == 0.85
        assert config.glass_break_threshold == 0.80
        assert config.gunshot_threshold == 0.90


class TestConfig:
    """Tests for main Config class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        
        assert config.app_name == "AI Vision Monitoring System"
        assert config.debug is False
        assert config.demo_mode is False
        assert config.target_fps == 30
        assert config.max_latency_ms == 150
    
    def test_from_dict(self):
        """Test configuration from dictionary."""
        data = {
            'app_name': 'Test App',
            'debug': True,
            'demo_mode': True,
            'target_fps': 60,
            'cameras': [
                {'id': 0, 'name': 'Camera 1', 'source': '0'}
            ],
            'audio': {
                'sample_rate': 22050,
                'channels': 2
            }
        }
        
        config = Config.from_dict(data)
        
        assert config.app_name == 'Test App'
        assert config.debug is True
        assert config.demo_mode is True
        assert config.target_fps == 60
        assert len(config.cameras) == 1
        assert config.cameras[0].name == 'Camera 1'
        assert config.audio.sample_rate == 22050
        assert config.audio.channels == 2
    
    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config()
        config.app_name = "Serialization Test"
        
        data = config.to_dict()
        
        assert data['app_name'] == "Serialization Test"
        assert 'cameras' in data
        assert 'audio' in data
        assert 'alerts' in data
        assert 'anomaly' in data
        assert 'ai' in data
    
    def test_yaml_round_trip(self):
        """Test YAML save and load."""
        config = Config()
        config.app_name = "YAML Test"
        config.debug = True
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            config.save_yaml(config_path)
            loaded_config = Config.from_yaml(config_path)
            
            assert loaded_config.app_name == "YAML Test"
            assert loaded_config.debug is True
        finally:
            os.unlink(config_path)
    
    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.data_dir = Path(tmpdir) / "data"
            config.logs_dir = Path(tmpdir) / "logs"
            config.clips_dir = Path(tmpdir) / "clips"
            config.events_dir = Path(tmpdir) / "events"
            config.models_dir = Path(tmpdir) / "models"
            
            config.ensure_directories()
            
            assert config.data_dir.exists()
            assert config.logs_dir.exists()
            assert config.clips_dir.exists()
            assert config.events_dir.exists()
            assert config.models_dir.exists()
    
    def test_env_variable_loading(self):
        """Test loading from environment variables."""
        os.environ['OPENAI_API_KEY'] = 'test-key-123'
        
        try:
            config = Config.from_dict({})
            assert config.ai.openai_api_key == 'test-key-123'
        finally:
            del os.environ['OPENAI_API_KEY']


class TestConfigIntegration:
    """Integration tests for configuration."""
    
    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create config
        config = Config()
        config.app_name = "Integration Test"
        
        # Add camera
        config.cameras = [
            CameraConfig(id=0, name="Test Cam", source="0")
        ]
        
        # Configure audio
        config.audio.sample_rate = 22050
        
        # Configure alerts
        config.alerts.desktop_enabled = True
        
        # Configure anomaly detection
        config.anomaly.anomaly_threshold = 0.8
        
        # Verify
        data = config.to_dict()
        assert data['app_name'] == "Integration Test"
        assert len(data['cameras']) == 1
        assert data['cameras'][0]['name'] == "Test Cam"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
