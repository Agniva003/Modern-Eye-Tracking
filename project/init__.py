"""
Eye Tracker - Modular Eye Tracking System

A modular, extensible eye-tracking solution using OpenCV and Python.
"""

__version__ = "1.0.0"
__author__ = "Stepan Filonov"
__email__ = "stepanfilonov@gmail.com"

from .eye_detector import EyeDetector, SimpleEyeDetector
from .frame_source import create_frame_source, FrameSource
from .utils import FPSCounter, draw_text, save_debug_frame

__all__ = [
    "EyeDetector",
    "SimpleEyeDetector",
    "create_frame_source",
    "FrameSource",
    "FPSCounter",
    "draw_text",
    "save_debug_frame",
]
