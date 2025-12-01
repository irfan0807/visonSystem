"""
Models and AI components for the monitoring application.
"""

try:
    from .scene_descriptions import SceneDescriber
except ImportError:
    pass  # Allow partial imports

__all__ = ['SceneDescriber']
