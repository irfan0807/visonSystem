"""
AI Vision Monitoring System - Main Streamlit Application

A comprehensive monitoring solution with:
- Live video feed with SAM 3 segmentation
- Real-time audio analysis and classification
- Anomaly detection and alerting
- AI-powered scene summarization
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import streamlit as st
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

from utils.config import Config
from utils.logger import setup_logger, get_logger
from utils.alerts import AlertManager, Alert
from core.video_processor import VideoProcessor, FrameResult
from core.audio_processor import AudioProcessor, AudioClassification
from core.anomaly_detector import AnomalyDetector
from models.scene_descriptions import SceneDescriber, SceneSummary


# Page configuration
st.set_page_config(
    page_title="AI Vision Monitor",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .alert-critical { color: #ff4444; font-weight: bold; }
    .alert-high { color: #ff8800; font-weight: bold; }
    .alert-medium { color: #ffcc00; }
    .alert-low { color: #44ff44; }
    .status-active { color: #00ff00; }
    .status-inactive { color: #ff0000; }
    .video-feed {
        border: 2px solid #333;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.config = None
        st.session_state.video_processor = None
        st.session_state.audio_processor = None
        st.session_state.alert_manager = None
        st.session_state.scene_describer = None
        st.session_state.running = False
        st.session_state.demo_mode = True
        st.session_state.alerts = []
        st.session_state.events_timeline = []
        st.session_state.anomaly_scores = []
        st.session_state.current_frame = None
        st.session_state.current_summary = None


def load_config() -> Config:
    """Load application configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
    else:
        config = Config()
        config.demo_mode = True
    
    # Ensure directories exist
    config.ensure_directories()
    
    return config


def initialize_system(config: Config):
    """Initialize all system components."""
    logger = setup_logger(
        "monitoring",
        log_dir=config.logs_dir,
        level=20 if not config.debug else 10
    )
    
    # Initialize alert manager
    alert_manager = AlertManager(
        email_enabled=config.alerts.email_enabled,
        email_smtp_server=config.alerts.email_smtp_server,
        email_smtp_port=config.alerts.email_smtp_port,
        email_sender=config.alerts.email_sender,
        email_password=config.alerts.email_password,
        email_recipients=config.alerts.email_recipients,
        sms_enabled=config.alerts.sms_enabled,
        twilio_sid=config.alerts.twilio_sid,
        twilio_token=config.alerts.twilio_token,
        twilio_from=config.alerts.twilio_from,
        sms_recipients=config.alerts.sms_recipients,
        desktop_enabled=config.alerts.desktop_enabled,
        sound_enabled=config.alerts.sound_enabled
    )
    
    # Initialize video processor
    video_processor = VideoProcessor(
        camera_config=config.cameras[0] if config.cameras else None,
        target_fps=config.target_fps,
        enable_sam=config.ai.use_gpu,
        enable_enhancement=True,
        enable_anomaly_detection=True,
        clips_dir=config.clips_dir
    )
    
    # Initialize SAM if available
    if config.ai.use_gpu:
        checkpoint = config.models_dir / "sam_vit_b.pth"
        if checkpoint.exists():
            video_processor.initialize_sam(
                model_type=config.ai.sam_model_type,
                checkpoint=str(checkpoint)
            )
    
    # Initialize audio processor
    audio_processor = AudioProcessor(
        sample_rate=config.audio.sample_rate,
        chunk_size=config.audio.chunk_size,
        channels=config.audio.channels,
        device_index=config.audio.device_index,
        vad_mode=config.audio.vad_mode,
        noise_suppression=config.audio.noise_suppression,
        model_path=config.models_dir / "audio_classifier.pkl"
    )
    
    # Initialize scene describer
    scene_describer = SceneDescriber(
        api_key=config.ai.openai_api_key,
        model=config.ai.openai_model,
        summary_interval=config.ai.scene_summary_interval,
        enable_vision=True
    )
    
    # Wire up callbacks
    def on_anomaly(score, event):
        if score > config.anomaly.anomaly_threshold:
            alert_manager.trigger_alert(
                alert_type="anomaly",
                message=event.get('description', 'Anomaly detected'),
                severity="high" if score > 0.85 else "medium",
                data={'score': score, **event}
            )
    
    def on_audio_event(event):
        alert_manager.trigger_alert(
            alert_type="audio",
            message=f"Audio event: {event.event_type}",
            severity="high" if event.confidence > 0.9 else "medium",
            data={
                'event_type': event.event_type,
                'confidence': event.confidence
            }
        )
    
    video_processor.add_anomaly_callback(on_anomaly)
    audio_processor.add_event_callback(on_audio_event)
    
    return {
        'config': config,
        'alert_manager': alert_manager,
        'video_processor': video_processor,
        'audio_processor': audio_processor,
        'scene_describer': scene_describer
    }


def render_sidebar():
    """Render the sidebar with controls and settings."""
    with st.sidebar:
        st.title("🔴 AI Vision Monitor")
        st.markdown("---")
        
        # Status indicators
        st.subheader("System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            video_status = "🟢" if st.session_state.running else "🔴"
            st.markdown(f"**Video:** {video_status}")
        with col2:
            audio_status = "🟢" if st.session_state.running else "🔴"
            st.markdown(f"**Audio:** {audio_status}")
        
        st.markdown("---")
        
        # Controls
        st.subheader("Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start" if not st.session_state.running else "⏹️ Stop", 
                        use_container_width=True):
                toggle_monitoring()
        
        with col2:
            if st.button("📸 Snapshot", use_container_width=True):
                capture_snapshot()
        
        # Demo mode toggle
        demo_mode = st.checkbox(
            "Demo Mode", 
            value=st.session_state.demo_mode,
            help="Use sample data instead of live camera"
        )
        if demo_mode != st.session_state.demo_mode:
            st.session_state.demo_mode = demo_mode
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        
        with st.expander("🎥 Video Settings"):
            target_fps = st.slider("Target FPS", 10, 60, 30)
            show_masks = st.checkbox("Show Segmentation Masks", True)
            show_bboxes = st.checkbox("Show Bounding Boxes", True)
            enhance_low_light = st.checkbox("Low-Light Enhancement", True)
        
        with st.expander("🎤 Audio Settings"):
            audio_sensitivity = st.slider("Sensitivity", 0.0, 1.0, 0.5)
            noise_suppression = st.checkbox("Noise Suppression", True)
        
        with st.expander("🚨 Alert Settings"):
            anomaly_threshold = st.slider(
                "Anomaly Threshold", 0.5, 1.0, 0.75
            )
            email_alerts = st.checkbox("Email Alerts", False)
            desktop_alerts = st.checkbox("Desktop Alerts", True)
        
        st.markdown("---")
        
        # Stats
        st.subheader("Statistics")
        
        if st.session_state.video_processor:
            stats = st.session_state.video_processor.get_stats()
            st.metric("Frames Processed", stats.get('frames_processed', 0))
            st.metric("Avg FPS", f"{stats.get('avg_fps', 0):.1f}")
        
        if st.session_state.alert_manager:
            alert_stats = st.session_state.alert_manager.get_stats()
            st.metric("Total Alerts", alert_stats.get('total_alerts', 0))


def toggle_monitoring():
    """Toggle monitoring on/off."""
    if st.session_state.running:
        # Stop
        if st.session_state.video_processor:
            st.session_state.video_processor.stop()
        if st.session_state.audio_processor:
            st.session_state.audio_processor.stop()
        if st.session_state.scene_describer:
            st.session_state.scene_describer.stop()
        if st.session_state.alert_manager:
            st.session_state.alert_manager.stop()
        st.session_state.running = False
    else:
        # Start
        if st.session_state.video_processor:
            st.session_state.video_processor.start()
        if st.session_state.audio_processor:
            st.session_state.audio_processor.start()
        if st.session_state.scene_describer:
            st.session_state.scene_describer.start_periodic()
        if st.session_state.alert_manager:
            st.session_state.alert_manager.start()
        st.session_state.running = True


def capture_snapshot():
    """Capture and save a snapshot."""
    if st.session_state.video_processor:
        path = st.session_state.video_processor.capture_snapshot()
        if path:
            st.success(f"Snapshot saved: {path}")
        else:
            st.warning("No frame available for snapshot")


def render_video_feed():
    """Render the main video feed."""
    st.subheader("🔴 Live Feed")
    
    # Video display
    video_placeholder = st.empty()
    
    if st.session_state.demo_mode:
        # Demo mode - show placeholder
        demo_frame = create_demo_frame()
        video_placeholder.image(demo_frame, channels="BGR", use_container_width=True)
    elif st.session_state.video_processor and st.session_state.running:
        # Live feed
        result = st.session_state.video_processor.get_result()
        if result:
            rendered = st.session_state.video_processor.render_frame(
                result.frame, result,
                show_masks=True,
                show_bboxes=True,
                show_info=True
            )
            video_placeholder.image(rendered, channels="BGR", use_container_width=True)
    else:
        st.info("Click Start to begin monitoring")


def create_demo_frame() -> np.ndarray:
    """Create a demo frame for demonstration."""
    if cv2 is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create gradient background
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        frame[y, :] = [30 + y//10, 30, 40]
    
    # Add text
    cv2.putText(frame, "DEMO MODE", (220, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.putText(frame, "Connect camera to view live feed", (150, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add simulated detection box
    cv2.rectangle(frame, (100, 150), (300, 400), (0, 255, 0), 2)
    cv2.putText(frame, "person (0.95)", (100, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame


def render_audio_panel():
    """Render the audio analysis panel."""
    st.subheader("🎤 Audio Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Waveform visualization
        if st.session_state.audio_processor and st.session_state.running:
            waveform = st.session_state.audio_processor.get_waveform(2.0)
            if len(waveform) > 0 and px:
                fig = px.line(
                    y=waveform[::10],  # Downsample for display
                    title="Audio Waveform"
                )
                fig.update_layout(
                    showlegend=False,
                    height=150,
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Placeholder waveform
            st.line_chart(np.random.randn(100) * 0.3, height=150)
    
    with col2:
        # Classification result
        st.markdown("**Classification**")
        
        if st.session_state.audio_processor:
            classification = st.session_state.audio_processor.get_classification()
            if classification:
                st.metric("Detected", classification.label.title())
                st.metric("Confidence", f"{classification.confidence:.0%}")
            else:
                st.metric("Detected", "Normal")
                st.metric("Confidence", "N/A")
        else:
            st.metric("Detected", "Normal")
            st.metric("Confidence", "N/A")


def render_anomaly_gauge():
    """Render the anomaly score gauge."""
    st.subheader("🎯 Anomaly Score")
    
    # Get current score
    score = 0.23  # Default demo value
    if st.session_state.video_processor and st.session_state.running:
        result = st.session_state.video_processor.get_result()
        if result:
            score = result.anomaly_score
    
    # Create gauge
    if go:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Anomaly Level (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Simple progress bar fallback
        st.progress(score)
        st.metric("Score", f"{score * 100:.1f}%")


def render_event_timeline():
    """Render the event timeline."""
    st.subheader("📋 Event Timeline")
    
    events = []
    
    # Get alerts from alert manager
    if st.session_state.alert_manager:
        alerts = st.session_state.alert_manager.get_history(limit=10)
        for alert in alerts:
            events.append({
                'time': alert.timestamp.strftime("%H:%M:%S"),
                'type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message
            })
    
    # Demo events if empty
    if not events:
        events = [
            {'time': '14:23:45', 'type': 'Motion', 'severity': 'medium', 'message': 'Motion detected in zone A'},
            {'time': '14:21:30', 'type': 'Audio', 'severity': 'low', 'message': 'Speech detected'},
            {'time': '14:20:15', 'type': 'Person', 'severity': 'low', 'message': 'Person entered frame'},
        ]
    
    # Display events
    for event in events:
        severity_class = f"alert-{event['severity']}"
        st.markdown(
            f"""<div class="metric-card">
                <span class="{severity_class}">[{event['severity'].upper()}]</span> 
                <strong>{event['type']}</strong> - {event['message']}
                <br><small>{event['time']}</small>
            </div>""",
            unsafe_allow_html=True
        )


def render_metrics_dashboard():
    """Render the metrics dashboard."""
    st.subheader("📊 Dashboard Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get stats
    video_stats = {}
    if st.session_state.video_processor:
        video_stats = st.session_state.video_processor.get_stats()
    
    audio_stats = {}
    if st.session_state.audio_processor:
        audio_stats = st.session_state.audio_processor.get_stats()
    
    alert_stats = {}
    if st.session_state.alert_manager:
        alert_stats = st.session_state.alert_manager.get_stats()
    
    with col1:
        objects = 5  # Demo default
        if st.session_state.video_processor and st.session_state.running:
            result = st.session_state.video_processor.get_result()
            if result:
                objects = len(result.detections)
        st.metric("Objects Tracked", objects, delta="🟢")
    
    with col2:
        score = 23  # Demo default
        if st.session_state.video_processor and st.session_state.running:
            result = st.session_state.video_processor.get_result()
            if result:
                score = int(result.anomaly_score * 100)
        status = "🟢" if score < 50 else ("🟡" if score < 75 else "🔴")
        st.metric("Anomaly Score", f"{score}%", delta=status)
    
    with col3:
        events = alert_stats.get('total_alerts', 2)
        st.metric("Events/Hour", events, delta="🟢")
    
    with col4:
        uptime = video_stats.get('uptime_seconds', 0)
        uptime_pct = 99.8 if uptime > 0 else 0.0
        st.metric("Uptime", f"{uptime_pct}%", delta="🟢")


def render_scene_summary():
    """Render the AI scene summary panel."""
    st.subheader("✨ AI Scene Summary")
    
    # Get current summary
    summary_text = "3 people entering room, one carrying bag, loud conversation"
    
    if st.session_state.scene_describer:
        summary = st.session_state.scene_describer.get_current_summary()
        if summary:
            summary_text = summary.description
    
    st.markdown(f"""
    <div class="metric-card">
        <p><strong>Current Scene:</strong></p>
        <p>{summary_text}</p>
        <small>Updated: {datetime.now().strftime('%H:%M:%S')}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Predictive analytics
    st.markdown("**🎯 Predictive Analytics**")
    st.info("• Unusual activity pattern detected\n• Person loitering >2 minutes")


def render_alert_history():
    """Render the alert history table."""
    st.subheader("🚨 Alert History")
    
    # Get alerts
    alerts_data = []
    
    if st.session_state.alert_manager:
        alerts = st.session_state.alert_manager.get_history(limit=20)
        for alert in alerts:
            alerts_data.append({
                'Time': alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'Type': alert.alert_type,
                'Severity': alert.severity.upper(),
                'Message': alert.message,
                'Status': '✓' if alert.acknowledged else '⚠'
            })
    
    # Demo data if empty
    if not alerts_data:
        alerts_data = [
            {'Time': '2024-01-15 14:23:45', 'Type': 'Motion', 'Severity': 'HIGH', 'Message': 'Motion detected zone A', 'Status': '✓'},
            {'Time': '2024-01-15 14:21:30', 'Type': 'Audio', 'Severity': 'MEDIUM', 'Message': 'Loud noise detected', 'Status': '⚠'},
            {'Time': '2024-01-15 14:20:15', 'Type': 'Person', 'Severity': 'LOW', 'Message': 'Person detected', 'Status': '✓'},
        ]
    
    st.dataframe(alerts_data, use_container_width=True)


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Initialize system if needed
    if not st.session_state.initialized:
        config = load_config()
        components = initialize_system(config)
        
        st.session_state.config = components['config']
        st.session_state.alert_manager = components['alert_manager']
        st.session_state.video_processor = components['video_processor']
        st.session_state.audio_processor = components['audio_processor']
        st.session_state.scene_describer = components['scene_describer']
        st.session_state.initialized = True
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🔴 AI Vision Monitoring System")
    
    # Top row - Video feed and metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_video_feed()
    
    with col2:
        render_anomaly_gauge()
        render_scene_summary()
    
    # Middle row - Audio and events
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_audio_panel()
    
    with col2:
        render_event_timeline()
    
    # Bottom row - Dashboard and alerts
    st.markdown("---")
    render_metrics_dashboard()
    
    st.markdown("---")
    render_alert_history()
    
    # Auto-refresh using Streamlit's built-in mechanism
    # Use st.empty with a placeholder for more efficient updates
    if st.session_state.running:
        # Refresh every 1 second instead of 0.1s to reduce CPU usage
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
