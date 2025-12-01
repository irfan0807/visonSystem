"""
Audio processing module for real-time audio analysis and classification.
Supports sound event detection, noise suppression, and feature extraction.
"""

import os
import time
import threading
import struct
from pathlib import Path
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from collections import deque
from datetime import datetime
import pickle
import json

import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

try:
    from ..utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger


@dataclass
class AudioEvent:
    """Represents a detected audio event."""
    timestamp: float
    event_type: str
    confidence: float
    duration_ms: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class AudioClassification:
    """Audio classification result."""
    label: str
    confidence: float
    all_probabilities: Dict[str, float] = field(default_factory=dict)


class AudioProcessor:
    """
    Real-time audio processor with classification and event detection.
    
    Features:
    - Real-time microphone capture
    - Voice Activity Detection (WebRTC VAD)
    - Feature extraction (MFCC, Spectral, ZCR)
    - Sound event classification
    - Noise suppression
    """
    
    # Default classification thresholds
    THRESHOLDS = {
        'scream': 0.85,
        'glass_break': 0.80,
        'gunshot': 0.90,
        'door_slam': 0.75,
        'speech': 0.60,
        'dog_bark': 0.75,
        'car_horn': 0.80,
        'siren': 0.85
    }
    
    # Class labels
    CLASSES = [
        'normal', 'speech', 'scream', 'glass_break', 
        'gunshot', 'door_slam', 'dog_bark', 'car_horn', 'siren'
    ]
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_index: Optional[int] = None,
        vad_mode: int = 2,
        noise_suppression: bool = True,
        model_path: Optional[Path] = None,
        buffer_duration: float = 10.0
    ):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Samples per chunk
            channels: Number of audio channels
            device_index: Audio input device index
            vad_mode: WebRTC VAD aggressiveness (0-3)
            noise_suppression: Enable noise suppression
            model_path: Path to classification model
            buffer_duration: Audio buffer duration (seconds)
        """
        self.logger = get_logger("audio")
        
        # Audio config
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        
        # VAD config
        self.vad_mode = vad_mode
        self.noise_suppression = noise_suppression
        
        # Initialize VAD
        self.vad = None
        if webrtcvad and noise_suppression:
            try:
                self.vad = webrtcvad.Vad(vad_mode)
            except Exception as e:
                self.logger.warning(f"VAD init failed: {e}")
        
        # PyAudio
        self.audio = None
        self.stream = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        
        # Audio buffer
        buffer_samples = int(sample_rate * buffer_duration)
        self.audio_buffer: deque = deque(maxlen=buffer_samples)
        self.raw_buffer: deque = deque(maxlen=int(sample_rate * 2))  # 2 seconds
        
        # Classification
        self.classifier = None
        self.scaler = None
        if model_path and model_path.exists():
            self._load_model(model_path)
        
        # Processing queue
        self.process_queue: Queue = Queue(maxsize=100)
        self._process_thread: Optional[threading.Thread] = None
        
        # Results
        self.current_classification: Optional[AudioClassification] = None
        self.current_features: Dict[str, float] = {}
        self.events: deque = deque(maxlen=100)
        
        # Waveform data for visualization
        self.waveform_buffer: deque = deque(maxlen=int(sample_rate * 5))
        
        # Callbacks
        self._event_callbacks: List[Callable[[AudioEvent], None]] = []
        self._classification_callbacks: List[Callable[[AudioClassification], None]] = []
        
        # Stats
        self.stats = {
            'chunks_processed': 0,
            'events_detected': 0,
            'total_duration': 0.0,
            'vad_active_ratio': 0.0
        }
        
        self._vad_active_count = 0
        self._vad_total_count = 0
    
    def _load_model(self, model_path: Path) -> bool:
        """Load classification model."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data.get('classifier')
            self.scaler = model_data.get('scaler')
            
            self.logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model load failed: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start audio capture and processing.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
        
        if pyaudio is None:
            self.logger.error("PyAudio not available")
            return False
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Open stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index
            )
            
            self._running = True
            
            # Start capture thread
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self._capture_thread.start()
            
            # Start processing thread
            self._process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self._process_thread.start()
            
            self.logger.info("Audio capture started")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio start failed: {e}")
            return False
    
    def stop(self) -> None:
        """Stop audio capture and processing."""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=5.0)
        
        if self._process_thread:
            self._process_thread.join(timeout=5.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        self.logger.info("Audio capture stopped")
    
    def _capture_loop(self) -> None:
        """Audio capture loop."""
        while self._running:
            try:
                if self.stream:
                    # Read audio data
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    samples = samples / 32768.0  # Normalize to [-1, 1]
                    
                    # Add to buffers
                    self.audio_buffer.extend(samples)
                    self.raw_buffer.extend(samples)
                    self.waveform_buffer.extend(samples)
                    
                    # Queue for processing
                    self.process_queue.put(samples)
                    
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                time.sleep(0.1)
    
    def _process_loop(self) -> None:
        """Audio processing loop."""
        analysis_interval = 0.5  # Analyze every 0.5 seconds
        last_analysis = time.time()
        
        while self._running:
            try:
                # Collect chunks
                chunk = self.process_queue.get(timeout=1.0)
                self.stats['chunks_processed'] += 1
                self.stats['total_duration'] = len(self.audio_buffer) / self.sample_rate
                
                # VAD check
                is_voice = self._check_vad(chunk)
                self._vad_total_count += 1
                if is_voice:
                    self._vad_active_count += 1
                
                # Update VAD stats
                if self._vad_total_count > 0:
                    self.stats['vad_active_ratio'] = (
                        self._vad_active_count / self._vad_total_count
                    )
                
                # Periodic analysis
                if time.time() - last_analysis >= analysis_interval:
                    self._analyze_audio()
                    last_analysis = time.time()
                    
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
    
    def _check_vad(self, samples: np.ndarray) -> bool:
        """Check Voice Activity Detection."""
        if self.vad is None:
            return True
        
        try:
            # Convert to 16-bit PCM
            pcm = (samples * 32768).astype(np.int16).tobytes()
            
            # WebRTC VAD requires specific frame sizes
            frame_duration = 30  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000)
            
            if len(samples) >= frame_size:
                frame = pcm[:frame_size * 2]  # 2 bytes per sample
                return self.vad.is_speech(frame, self.sample_rate)
            
            return True
            
        except Exception:
            return True
    
    def _analyze_audio(self) -> None:
        """Analyze buffered audio."""
        if len(self.raw_buffer) < self.sample_rate:
            return
        
        # Get recent audio
        audio_data = np.array(list(self.raw_buffer))
        
        # Extract features
        features = self._extract_features(audio_data)
        self.current_features = features
        
        # Classify
        if self.classifier is not None:
            classification = self._classify(features)
            self.current_classification = classification
            
            # Notify callbacks
            for callback in self._classification_callbacks:
                try:
                    callback(classification)
                except Exception as e:
                    self.logger.error(f"Classification callback error: {e}")
            
            # Check for events
            self._check_events(classification)
    
    def _extract_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract audio features.
        
        Features:
        - MFCC (13 coefficients)
        - Spectral Centroid
        - Spectral Rolloff
        - Zero Crossing Rate
        - RMS Energy
        """
        features = {}
        
        if librosa is None:
            # Basic features without librosa
            features['rms'] = float(np.sqrt(np.mean(audio ** 2)))
            features['zcr'] = float(np.mean(np.abs(np.diff(np.signbit(audio)))))
            features['max_amplitude'] = float(np.max(np.abs(audio)))
            return features
        
        try:
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            for i, mfcc_mean in enumerate(np.mean(mfccs, axis=1)):
                features[f'mfcc_{i}'] = float(mfcc_mean)
            
            # Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )
            features['spectral_centroid'] = float(np.mean(spectral_centroid))
            
            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate
            )
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zcr'] = float(np.mean(zcr))
            
            # RMS Energy
            rms = librosa.feature.rms(y=audio)
            features['rms'] = float(np.mean(rms))
            
            # Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate
            )
            features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            # Fallback features
            features['rms'] = float(np.sqrt(np.mean(audio ** 2)))
            features['zcr'] = float(np.mean(np.abs(np.diff(np.signbit(audio)))))
        
        return features
    
    def _classify(self, features: Dict[str, float]) -> AudioClassification:
        """Classify audio using trained model."""
        try:
            # Prepare feature vector
            feature_names = sorted(features.keys())
            feature_vector = np.array([[features[f] for f in feature_names]])
            
            # Scale features
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Predict
            probabilities = self.classifier.predict_proba(feature_vector)[0]
            classes = self.classifier.classes_
            
            # Get best prediction
            best_idx = np.argmax(probabilities)
            best_label = classes[best_idx]
            best_confidence = probabilities[best_idx]
            
            # Build probability dict
            prob_dict = {
                classes[i]: float(probabilities[i]) 
                for i in range(len(classes))
            }
            
            return AudioClassification(
                label=best_label,
                confidence=float(best_confidence),
                all_probabilities=prob_dict
            )
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return AudioClassification(
                label='unknown',
                confidence=0.0,
                all_probabilities={}
            )
    
    def _check_events(self, classification: AudioClassification) -> None:
        """Check for audio events based on classification."""
        label = classification.label
        confidence = classification.confidence
        
        # Get threshold for this class
        threshold = self.THRESHOLDS.get(label, 0.7)
        
        # Check if event should be triggered
        if label != 'normal' and label != 'speech' and confidence >= threshold:
            event = AudioEvent(
                timestamp=time.time(),
                event_type=label,
                confidence=confidence,
                duration_ms=len(self.raw_buffer) / self.sample_rate * 1000,
                features=self.current_features
            )
            
            self.events.append(event)
            self.stats['events_detected'] += 1
            
            # Notify callbacks
            for callback in self._event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Event callback error: {e}")
            
            self.logger.info(
                f"Audio event: {label} (confidence: {confidence:.2f})"
            )
    
    def add_event_callback(self, callback: Callable[[AudioEvent], None]) -> None:
        """Add callback for audio events."""
        self._event_callbacks.append(callback)
    
    def add_classification_callback(
        self, 
        callback: Callable[[AudioClassification], None]
    ) -> None:
        """Add callback for classifications."""
        self._classification_callbacks.append(callback)
    
    def get_waveform(self, duration: float = 2.0) -> np.ndarray:
        """
        Get recent waveform data for visualization.
        
        Args:
            duration: Duration in seconds
        
        Returns:
            Audio waveform array
        """
        samples_needed = int(self.sample_rate * duration)
        waveform = list(self.waveform_buffer)
        
        if len(waveform) >= samples_needed:
            return np.array(waveform[-samples_needed:])
        else:
            return np.array(waveform) if waveform else np.zeros(samples_needed)
    
    def get_spectrum(self, n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frequency spectrum for visualization.
        
        Args:
            n_fft: FFT size
        
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        if len(self.raw_buffer) < n_fft:
            return np.zeros(n_fft // 2), np.zeros(n_fft // 2)
        
        audio = np.array(list(self.raw_buffer)[-n_fft:])
        
        # Apply window
        window = np.hanning(len(audio))
        audio_windowed = audio * window
        
        # FFT
        spectrum = np.abs(np.fft.rfft(audio_windowed))
        frequencies = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        
        return frequencies, spectrum
    
    def get_classification(self) -> Optional[AudioClassification]:
        """Get current audio classification."""
        return self.current_classification
    
    def get_features(self) -> Dict[str, float]:
        """Get current audio features."""
        return self.current_features.copy()
    
    def get_events(self, limit: int = 10) -> List[AudioEvent]:
        """Get recent audio events."""
        events = list(self.events)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            **self.stats,
            'model_loaded': self.classifier is not None,
            'vad_enabled': self.vad is not None,
            'stream_active': self.stream is not None and self.stream.is_active()
        }
    
    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """List available audio input devices."""
        if pyaudio is None:
            return []
        
        devices = []
        audio = pyaudio.PyAudio()
        
        try:
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
        finally:
            audio.terminate()
        
        return devices
