"""
Alert system module for the monitoring application.
Supports email, SMS (Twilio), and desktop notifications.
"""

import os
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from queue import Queue, Empty
from collections import deque

from .logger import get_logger


@dataclass
class Alert:
    """Represents an alert event."""
    id: str
    timestamp: datetime
    alert_type: str  # 'anomaly', 'audio', 'person', 'custom'
    severity: str    # 'low', 'medium', 'high', 'critical'
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    clip_path: Optional[Path] = None
    image_path: Optional[Path] = None
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'data': self.data,
            'clip_path': str(self.clip_path) if self.clip_path else None,
            'image_path': str(self.image_path) if self.image_path else None,
            'acknowledged': self.acknowledged
        }


class AlertManager:
    """
    Manages alert generation, notification delivery, and history.
    Supports multiple notification channels and queuing for offline mode.
    """
    
    def __init__(
        self,
        email_enabled: bool = False,
        email_smtp_server: str = "",
        email_smtp_port: int = 587,
        email_sender: str = "",
        email_password: str = "",
        email_recipients: List[str] = None,
        sms_enabled: bool = False,
        twilio_sid: str = "",
        twilio_token: str = "",
        twilio_from: str = "",
        sms_recipients: List[str] = None,
        desktop_enabled: bool = True,
        sound_enabled: bool = True,
        max_history: int = 1000,
        cooldown_seconds: float = 5.0
    ):
        """
        Initialize the alert manager.
        
        Args:
            email_enabled: Enable email notifications
            email_smtp_server: SMTP server address
            email_smtp_port: SMTP port
            email_sender: Sender email address
            email_password: SMTP password
            email_recipients: List of recipient emails
            sms_enabled: Enable SMS notifications
            twilio_sid: Twilio account SID
            twilio_token: Twilio auth token
            twilio_from: Twilio phone number
            sms_recipients: List of recipient phone numbers
            desktop_enabled: Enable desktop notifications
            sound_enabled: Enable sound alerts
            max_history: Maximum alerts to keep in history
            cooldown_seconds: Minimum time between alerts of same type
        """
        self.logger = get_logger("alerts")
        
        # Email config
        self.email_enabled = email_enabled
        self.email_smtp_server = email_smtp_server
        self.email_smtp_port = email_smtp_port
        self.email_sender = email_sender
        self.email_password = email_password
        self.email_recipients = email_recipients or []
        
        # SMS config
        self.sms_enabled = sms_enabled
        self.twilio_sid = twilio_sid
        self.twilio_token = twilio_token
        self.twilio_from = twilio_from
        self.sms_recipients = sms_recipients or []
        
        # Desktop/sound config
        self.desktop_enabled = desktop_enabled
        self.sound_enabled = sound_enabled
        
        # Alert history and queue
        self.history: deque = deque(maxlen=max_history)
        self.alert_queue: Queue = Queue()
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_times: Dict[str, float] = {}
        
        # Callbacks
        self._callbacks: List[Callable[[Alert], None]] = []
        
        # Worker thread
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Stats
        self.stats = {
            'total_alerts': 0,
            'emails_sent': 0,
            'sms_sent': 0,
            'desktop_shown': 0,
            'errors': 0
        }
    
    def start(self) -> None:
        """Start the alert worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        self.logger.info("Alert manager started")
    
    def stop(self) -> None:
        """Stop the alert worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        self.logger.info("Alert manager stopped")
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback function to be called on new alerts."""
        self._callbacks.append(callback)
    
    def trigger_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "medium",
        data: Dict[str, Any] = None,
        clip_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
        bypass_cooldown: bool = False
    ) -> Optional[Alert]:
        """
        Trigger a new alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity
            data: Additional alert data
            clip_path: Path to video clip
            image_path: Path to image snapshot
            bypass_cooldown: Skip cooldown check
        
        Returns:
            Alert object if triggered, None if rate-limited
        """
        # Check cooldown
        if not bypass_cooldown:
            last_time = self._last_alert_times.get(alert_type, 0)
            if time.time() - last_time < self.cooldown_seconds:
                return None
        
        # Create alert
        alert = Alert(
            id=f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            data=data or {},
            clip_path=clip_path,
            image_path=image_path
        )
        
        # Update cooldown
        self._last_alert_times[alert_type] = time.time()
        
        # Add to history
        self.history.append(alert)
        self.stats['total_alerts'] += 1
        
        # Queue for processing
        self.alert_queue.put(alert)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
        
        self.logger.info(f"Alert triggered: [{severity.upper()}] {alert_type} - {message}")
        return alert
    
    def _process_queue(self) -> None:
        """Process alerts from the queue."""
        while self._running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                self._send_notifications(alert)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
                self.stats['errors'] += 1
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        # Desktop notification
        if self.desktop_enabled:
            self._send_desktop_notification(alert)
        
        # Email notification (for high/critical)
        if self.email_enabled and alert.severity in ['high', 'critical']:
            self._send_email_notification(alert)
        
        # SMS notification (for critical only)
        if self.sms_enabled and alert.severity == 'critical':
            self._send_sms_notification(alert)
    
    def _send_desktop_notification(self, alert: Alert) -> None:
        """Send desktop notification."""
        try:
            # Try to use platform-specific notification
            title = f"🚨 {alert.alert_type.upper()} Alert"
            body = alert.message
            
            # Try different notification backends
            if os.name == 'nt':  # Windows
                self._windows_notify(title, body)
            elif os.name == 'posix':
                self._linux_notify(title, body)
            
            self.stats['desktop_shown'] += 1
            self.logger.debug(f"Desktop notification sent: {title}")
            
        except Exception as e:
            self.logger.warning(f"Desktop notification failed: {e}")
    
    def _windows_notify(self, title: str, body: str) -> None:
        """Send Windows notification."""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(title, body, duration=5, threaded=True)
        except ImportError:
            # Fallback to console
            print(f"\n{'='*50}\n{title}\n{body}\n{'='*50}\n")
    
    def _linux_notify(self, title: str, body: str) -> None:
        """Send Linux notification."""
        try:
            import subprocess
            subprocess.run(['notify-send', title, body], check=False, 
                         capture_output=True, timeout=5)
        except Exception:
            # Fallback to console
            print(f"\n{'='*50}\n{title}\n{body}\n{'='*50}\n")
    
    def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification."""
        if not self.email_recipients:
            return
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.alert_type} Alert - AI Vision Monitor"
            msg['From'] = self.email_sender
            msg['To'] = ', '.join(self.email_recipients)
            
            # HTML content
            html = self._generate_email_html(alert)
            msg.attach(MIMEText(html, 'html'))
            
            # Attach image if present
            if alert.image_path and alert.image_path.exists():
                with open(alert.image_path, 'rb') as f:
                    img_attachment = MIMEBase('image', 'jpeg')
                    img_attachment.set_payload(f.read())
                    encoders.encode_base64(img_attachment)
                    img_attachment.add_header(
                        'Content-Disposition', 
                        f'attachment; filename="snapshot.jpg"'
                    )
                    msg.attach(img_attachment)
            
            # Send email
            with smtplib.SMTP(self.email_smtp_server, self.email_smtp_port) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)
            
            self.stats['emails_sent'] += 1
            self.logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
            self.stats['errors'] += 1
    
    def _generate_email_html(self, alert: Alert) -> str:
        """Generate HTML email content."""
        severity_colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        color = severity_colors.get(alert.severity, '#6c757d')
        
        data_rows = ''.join(
            f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
            for k, v in alert.data.items()
        )
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ padding: 20px; background-color: #f8f9fa; border-radius: 5px; margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 {alert.alert_type.upper()} Alert</h2>
                <p>Severity: {alert.severity.upper()}</p>
            </div>
            <div class="content">
                <h3>Alert Details</h3>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <table>{data_rows}</table>
            </div>
            <p style="color: #6c757d; font-size: 12px;">
                This alert was generated by AI Vision Monitoring System
            </p>
        </body>
        </html>
        """
    
    def _send_sms_notification(self, alert: Alert) -> None:
        """Send SMS notification via Twilio."""
        if not self.sms_recipients:
            return
        
        try:
            from twilio.rest import Client
            
            client = Client(self.twilio_sid, self.twilio_token)
            
            body = f"🚨 {alert.severity.upper()} ALERT\n{alert.alert_type}: {alert.message}"
            
            for recipient in self.sms_recipients:
                message = client.messages.create(
                    body=body,
                    from_=self.twilio_from,
                    to=recipient
                )
                self.logger.debug(f"SMS sent to {recipient}: {message.sid}")
            
            self.stats['sms_sent'] += len(self.sms_recipients)
            self.logger.info(f"SMS notification sent for alert {alert.id}")
            
        except ImportError:
            self.logger.warning("Twilio library not installed")
        except Exception as e:
            self.logger.error(f"SMS notification failed: {e}")
            self.stats['errors'] += 1
    
    def get_history(
        self,
        limit: int = 50,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Alert]:
        """
        Get alert history with optional filters.
        
        Args:
            limit: Maximum alerts to return
            alert_type: Filter by alert type
            severity: Filter by severity
            since: Filter alerts after this time
        
        Returns:
            List of alerts matching filters
        """
        alerts = list(self.history)
        
        # Apply filters
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        # Sort by timestamp (newest first) and limit
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
        
        Returns:
            True if alert was found and acknowledged
        """
        for alert in self.history:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return {
            **self.stats,
            'unacknowledged': sum(1 for a in self.history if not a.acknowledged),
            'critical_count': sum(1 for a in self.history if a.severity == 'critical'),
            'high_count': sum(1 for a in self.history if a.severity == 'high')
        }
