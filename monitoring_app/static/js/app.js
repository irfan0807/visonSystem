// AI Vision Monitoring System - Client-side JavaScript

// Utility functions for the monitoring UI
const MonitoringApp = {
    // Initialize the application
    init: function() {
        console.log('AI Vision Monitoring System initialized');
        this.setupEventListeners();
    },

    // Setup event listeners
    setupEventListeners: function() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyPress(e);
        });
    },

    // Handle keyboard shortcuts
    handleKeyPress: function(e) {
        // Space to toggle monitoring
        if (e.code === 'Space' && !this.isInputFocused()) {
            e.preventDefault();
            this.toggleMonitoring();
        }
        
        // S for snapshot
        if (e.code === 'KeyS' && e.ctrlKey) {
            e.preventDefault();
            this.captureSnapshot();
        }
    },

    // Check if an input element is focused
    isInputFocused: function() {
        const activeElement = document.activeElement;
        return activeElement && 
               (activeElement.tagName === 'INPUT' || 
                activeElement.tagName === 'TEXTAREA');
    },

    // Toggle monitoring state (triggers Streamlit button)
    toggleMonitoring: function() {
        const button = document.querySelector('[data-testid="stButton"] button');
        if (button) {
            button.click();
        }
    },

    // Capture snapshot (triggers Streamlit button)
    captureSnapshot: function() {
        const buttons = document.querySelectorAll('[data-testid="stButton"] button');
        buttons.forEach(button => {
            if (button.textContent.includes('Snapshot')) {
                button.click();
            }
        });
    },

    // Play alert sound
    playAlertSound: function(severity = 'medium') {
        const sounds = {
            critical: 'alert-critical.mp3',
            high: 'alert-high.mp3',
            medium: 'alert-medium.mp3',
            low: 'alert-low.mp3'
        };
        
        const audio = new Audio(`/static/sounds/${sounds[severity] || sounds.medium}`);
        audio.volume = 0.5;
        audio.play().catch(e => console.log('Audio play failed:', e));
    },

    // Show browser notification
    showNotification: function(title, body, severity = 'medium') {
        if (!('Notification' in window)) {
            console.log('Browser does not support notifications');
            return;
        }

        if (Notification.permission === 'granted') {
            this.createNotification(title, body, severity);
        } else if (Notification.permission !== 'denied') {
            Notification.requestPermission().then(permission => {
                if (permission === 'granted') {
                    this.createNotification(title, body, severity);
                }
            });
        }
    },

    // Create notification
    createNotification: function(title, body, severity) {
        const icons = {
            critical: '🚨',
            high: '⚠️',
            medium: '📢',
            low: 'ℹ️'
        };

        new Notification(title, {
            body: `${icons[severity] || '📢'} ${body}`,
            icon: '/static/img/logo.png',
            badge: '/static/img/badge.png',
            tag: `alert-${Date.now()}`
        });
    },

    // Format timestamp
    formatTime: function(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },

    // Format duration
    formatDuration: function(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        
        if (h > 0) {
            return `${h}h ${m}m ${s}s`;
        } else if (m > 0) {
            return `${m}m ${s}s`;
        }
        return `${s}s`;
    },

    // Calculate color based on value
    getColorForValue: function(value, min = 0, max = 100) {
        const normalized = (value - min) / (max - min);
        
        if (normalized < 0.5) {
            // Green to Yellow
            const r = Math.round(normalized * 2 * 255);
            return `rgb(${r}, 255, 0)`;
        } else {
            // Yellow to Red
            const g = Math.round((1 - (normalized - 0.5) * 2) * 255);
            return `rgb(255, ${g}, 0)`;
        }
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => MonitoringApp.init());
} else {
    MonitoringApp.init();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MonitoringApp;
}
