"""
AI-powered scene summarization using OpenAI GPT models.
Generates human-readable descriptions of detected scenes and events.
"""

import os
import time
import base64
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from ..utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger


@dataclass
class SceneSummary:
    """Represents a scene summary."""
    timestamp: float
    description: str
    objects_detected: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    frame_index: int = 0


class SceneDescriber:
    """
    AI-powered scene description generator using OpenAI GPT models.
    
    Features:
    - Visual scene analysis using GPT-4 Vision
    - Context-aware summaries
    - Activity detection
    - Periodic and event-triggered summaries
    """
    
    SYSTEM_PROMPT = """You are an AI security monitoring assistant. 
Analyze the scene and provide a concise, security-focused description.

Focus on:
1. Number and types of people/vehicles present
2. Notable activities or behaviors
3. Any potential security concerns
4. Changes from normal patterns

Keep responses under 100 words. Be specific and factual.
Format: One clear sentence describing the main scene, followed by bullet points for details if needed."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        summary_interval: int = 30,
        max_history: int = 100,
        enable_vision: bool = True
    ):
        """
        Initialize the scene describer.
        
        Args:
            api_key: OpenAI API key (defaults to env var)
            model: OpenAI model to use
            summary_interval: Seconds between automatic summaries
            max_history: Maximum summaries to keep in history
            enable_vision: Enable image analysis
        """
        self.logger = get_logger("scene")
        
        # OpenAI config
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model
        self.enable_vision = enable_vision
        
        # Initialize client
        self.client = None
        if OpenAI and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.logger.info(f"OpenAI client initialized with model: {model}")
            except Exception as e:
                self.logger.error(f"OpenAI init failed: {e}")
        
        # Timing
        self.summary_interval = summary_interval
        self.last_summary_time = 0
        
        # History
        self.history: deque = deque(maxlen=max_history)
        self.current_summary: Optional[SceneSummary] = None
        
        # State
        self._running = False
        self._summary_thread: Optional[threading.Thread] = None
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_context: Dict[str, Any] = {}
        
        # Callbacks
        self._callbacks: List[Callable[[SceneSummary], None]] = []
        
        # Stats
        self.stats = {
            'summaries_generated': 0,
            'api_calls': 0,
            'api_errors': 0,
            'avg_latency_ms': 0.0
        }
        
        self._latencies: deque = deque(maxlen=50)
    
    def start_periodic(self) -> None:
        """Start periodic summary generation."""
        if self._running:
            return
        
        self._running = True
        self._summary_thread = threading.Thread(
            target=self._periodic_loop, daemon=True
        )
        self._summary_thread.start()
        self.logger.info("Periodic scene description started")
    
    def stop(self) -> None:
        """Stop periodic summary generation."""
        self._running = False
        if self._summary_thread:
            self._summary_thread.join(timeout=5.0)
        self.logger.info("Scene description stopped")
    
    def _periodic_loop(self) -> None:
        """Periodic summary generation loop."""
        while self._running:
            try:
                current_time = time.time()
                
                if current_time - self.last_summary_time >= self.summary_interval:
                    if self._pending_frame is not None:
                        self.generate_summary(
                            self._pending_frame,
                            self._pending_context
                        )
                    self.last_summary_time = current_time
                
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Periodic summary error: {e}")
    
    def update_frame(
        self, 
        frame: np.ndarray, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the current frame for analysis.
        
        Args:
            frame: Current video frame
            context: Additional context (detections, anomaly score, etc.)
        """
        self._pending_frame = frame.copy() if frame is not None else None
        self._pending_context = context or {}
    
    def generate_summary(
        self,
        frame: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Optional[SceneSummary]:
        """
        Generate a scene summary.
        
        Args:
            frame: Video frame to analyze
            context: Additional context
            force: Force generation even if client unavailable
        
        Returns:
            SceneSummary if successful
        """
        if self.client is None:
            return self._generate_fallback_summary(frame, context)
        
        start_time = time.time()
        context = context or {}
        
        try:
            # Build messages
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
            
            # Build user message
            user_content = []
            
            # Add context
            context_text = self._build_context_text(context)
            if context_text:
                user_content.append({"type": "text", "text": context_text})
            
            # Add image if available and vision enabled
            if frame is not None and self.enable_vision and cv2:
                image_data = self._encode_frame(frame)
                if image_data:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "low"
                        }
                    })
            
            if not user_content:
                user_content.append({
                    "type": "text", 
                    "text": "Describe the current scene."
                })
            
            messages.append({"role": "user", "content": user_content})
            
            # Call API with timeout for production reliability
            self.stats['api_calls'] += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.3,
                timeout=30.0  # 30 second timeout
            )
            
            description = response.choices[0].message.content
            
            # Parse response
            summary = self._parse_response(description, context)
            
            # Update timing
            latency = (time.time() - start_time) * 1000
            self._latencies.append(latency)
            self.stats['avg_latency_ms'] = np.mean(list(self._latencies))
            
            # Store and notify
            self.history.append(summary)
            self.current_summary = summary
            self.stats['summaries_generated'] += 1
            
            for callback in self._callbacks:
                try:
                    callback(summary)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
            
            self.logger.debug(f"Scene summary generated: {description[:100]}...")
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            self.stats['api_errors'] += 1
            return self._generate_fallback_summary(frame, context)
    
    def _encode_frame(self, frame: np.ndarray) -> Optional[str]:
        """Encode frame to base64 JPEG."""
        if cv2 is None:
            return None
        
        try:
            # Resize for efficiency
            max_dim = 512
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # Encode
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Frame encoding error: {e}")
            return None
    
    def _build_context_text(self, context: Dict[str, Any]) -> str:
        """Build context text for the prompt."""
        parts = ["Analyze this security camera scene."]
        
        if 'detections' in context:
            detections = context['detections']
            if detections:
                labels = [d.get('label', 'object') for d in detections]
                parts.append(f"Detected objects: {', '.join(labels)}")
        
        if 'anomaly_score' in context:
            score = context['anomaly_score']
            if score > 0.5:
                parts.append(f"Anomaly score: {score:.2f} (elevated)")
        
        if 'audio_event' in context:
            event = context['audio_event']
            parts.append(f"Audio detected: {event}")
        
        if 'previous_summary' in context:
            parts.append(f"Previous: {context['previous_summary']}")
        
        return " ".join(parts)
    
    def _parse_response(
        self, 
        description: str, 
        context: Dict[str, Any]
    ) -> SceneSummary:
        """Parse API response into SceneSummary."""
        # Extract objects from context
        objects = []
        if 'detections' in context:
            objects = [d.get('label', 'object') for d in context['detections']]
        
        # Extract activities (simple heuristics)
        activities = []
        lower_desc = description.lower()
        if 'walking' in lower_desc:
            activities.append('walking')
        if 'standing' in lower_desc:
            activities.append('standing')
        if 'entering' in lower_desc:
            activities.append('entering')
        if 'leaving' in lower_desc:
            activities.append('leaving')
        if 'talking' in lower_desc or 'conversation' in lower_desc:
            activities.append('conversation')
        
        # Check for alerts
        alerts = []
        alert_keywords = ['suspicious', 'unusual', 'concern', 'alert', 'warning']
        for keyword in alert_keywords:
            if keyword in lower_desc:
                alerts.append(f"Potential {keyword} activity")
                break
        
        return SceneSummary(
            timestamp=time.time(),
            description=description,
            objects_detected=objects,
            activities=activities,
            alerts=alerts,
            confidence=0.9,
            frame_index=context.get('frame_index', 0)
        )
    
    def _generate_fallback_summary(
        self, 
        frame: Optional[np.ndarray], 
        context: Optional[Dict[str, Any]]
    ) -> SceneSummary:
        """Generate fallback summary without API."""
        context = context or {}
        
        # Build description from context
        parts = []
        
        if 'detections' in context:
            num_objects = len(context['detections'])
            if num_objects > 0:
                parts.append(f"{num_objects} object(s) detected")
        
        if 'anomaly_score' in context:
            score = context['anomaly_score']
            if score > 0.75:
                parts.append("High anomaly detected")
            elif score > 0.5:
                parts.append("Moderate activity")
            else:
                parts.append("Normal activity")
        
        if 'audio_event' in context:
            parts.append(f"Audio: {context['audio_event']}")
        
        description = ". ".join(parts) if parts else "Scene monitoring active"
        
        return SceneSummary(
            timestamp=time.time(),
            description=description,
            objects_detected=[
                d.get('label', 'object') 
                for d in context.get('detections', [])
            ],
            activities=[],
            alerts=[],
            confidence=0.5,
            frame_index=context.get('frame_index', 0)
        )
    
    def add_callback(self, callback: Callable[[SceneSummary], None]) -> None:
        """Add callback for new summaries."""
        self._callbacks.append(callback)
    
    def trigger_summary(
        self, 
        frame: np.ndarray, 
        context: Optional[Dict[str, Any]] = None,
        reason: str = "event"
    ) -> Optional[SceneSummary]:
        """
        Trigger an immediate summary (e.g., on anomaly).
        
        Args:
            frame: Video frame
            context: Additional context
            reason: Reason for trigger
        
        Returns:
            SceneSummary if successful
        """
        self.logger.info(f"Summary triggered: {reason}")
        return self.generate_summary(frame, context, force=True)
    
    def get_current_summary(self) -> Optional[SceneSummary]:
        """Get the most recent summary."""
        return self.current_summary
    
    def get_history(self, limit: int = 10) -> List[SceneSummary]:
        """Get summary history."""
        summaries = list(self.history)
        summaries.sort(key=lambda s: s.timestamp, reverse=True)
        return summaries[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get describer statistics."""
        return {
            **self.stats,
            'client_available': self.client is not None,
            'vision_enabled': self.enable_vision,
            'history_size': len(self.history)
        }
    
    def search_history(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[SceneSummary]:
        """
        Search history for matching summaries.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            Matching summaries
        """
        query_lower = query.lower()
        matches = [
            s for s in self.history
            if query_lower in s.description.lower()
            or any(query_lower in obj.lower() for obj in s.objects_detected)
            or any(query_lower in act.lower() for act in s.activities)
        ]
        
        matches.sort(key=lambda s: s.timestamp, reverse=True)
        return matches[:limit]
