"""
Utility functions for Eye Tracker

Provides helper functions for:
- FPS calculation and display
- Text rendering with background
- Debug frame saving
- Performance monitoring
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple

from settings import (
    DEBUG_DUMP, 
    DEBUG_DUMP_LOCATION,
    FPS_SMOOTHING,
    TEXT_SCALE,
    TEXT_THICKNESS,
    DEBUG_MODE
)


class FPSCounter:
    """
    Calculate and smooth FPS (Frames Per Second)
    
    Uses exponential moving average for smooth FPS readings.
    
    Args:
        smoothing: Smoothing factor (0.0 to 1.0)
                   Higher values = smoother but less responsive
    """
    
    def __init__(self, smoothing: float = FPS_SMOOTHING):
        self.smoothing = max(0.0, min(1.0, smoothing))  # Clamp between 0 and 1
        self.fps = 0.0
        self.last_time = time.time()
        self.frame_count = 0
    
    def update(self) -> float:
        """
        Update FPS calculation
        
        Should be called once per frame.
        
        Returns:
            float: Current smoothed FPS value
        """
        current_time = time.time()
        delta = current_time - self.last_time
        
        if delta > 0:
            # Calculate instantaneous FPS
            current_fps = 1.0 / delta
            
            # Apply exponential moving average
            if self.frame_count == 0:
                # First frame: use instantaneous FPS
                self.fps = current_fps
            else:
                # Smooth with previous FPS
                self.fps = self.smoothing * self.fps + (1 - self.smoothing) * current_fps
        
        self.last_time = current_time
        self.frame_count += 1
        
        return self.fps
    
    def get_fps(self) -> float:
        """
        Get current FPS without updating
        
        Returns:
            float: Current FPS value
        """
        return self.fps
    
    def reset(self):
        """Reset FPS counter"""
        self.fps = 0.0
        self.last_time = time.time()
        self.frame_count = 0


def draw_text(
    frame: np.ndarray, 
    text: str, 
    position: Tuple[int, int], 
    color: Tuple[int, int, int] = (0, 255, 0), 
    scale: float = TEXT_SCALE,
    thickness: int = TEXT_THICKNESS,
    background: bool = True,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    background_alpha: float = 0.7
) -> None:
    """
    Draw text on frame with optional background
    
    Args:
        frame: Image to draw on (modified in place)
        text: Text string to draw
        position: (x, y) position for bottom-left corner of text
        color: Text color in BGR format
        scale: Font scale factor
        thickness: Text thickness
        background: Whether to draw background rectangle
        background_color: Background color in BGR format
        background_alpha: Background transparency (0.0 to 1.0)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    
    x, y = position
    
    if background:
        # Calculate background rectangle coordinates
        padding = 5
        bg_x1 = x - padding
        bg_y1 = y - text_height - baseline - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            background_color,
            -1
        )
        
        # Blend overlay with original frame
        cv2.addWeighted(
            overlay, background_alpha, 
            frame, 1 - background_alpha, 
            0, frame
        )
    
    # Draw text
    cv2.putText(
        frame, text, (x, y), 
        font, scale, color, thickness, 
        cv2.LINE_AA
    )


def draw_multiline_text(
    frame: np.ndarray,
    lines: list,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    scale: float = TEXT_SCALE,
    thickness: int = TEXT_THICKNESS,
    line_spacing: int = 30,
    background: bool = True
) -> None:
    """
    Draw multiple lines of text
    
    Args:
        frame: Image to draw on
        lines: List of text strings
        position: Starting position (x, y) for first line
        color: Text color in BGR format
        scale: Font scale factor
        thickness: Text thickness
        line_spacing: Vertical spacing between lines in pixels
        background: Whether to draw background
    """
    x, y = position
    
    for i, line in enumerate(lines):
        current_y = y + i * line_spacing
        draw_text(
            frame, line, (x, current_y), 
            color, scale, thickness, background
        )


def save_debug_frame(
    frame: np.ndarray, 
    prefix: str = "frame",
    suffix: str = ""
) -> Path:
    """
    Save frame to debug dump location
    
    Saves frame with timestamp in filename for debugging purposes.
    Only saves if DEBUG_DUMP is enabled in settings.
    
    Args:
        frame: Image to save
        prefix: Filename prefix
        suffix: Optional suffix to add before extension
    
    Returns:
        Path: Path to saved file, or None if not saved
    """
    if not DEBUG_DUMP:
        if DEBUG_MODE:
            print("Debug dump is disabled. Set DEBUG_DUMP=true to enable.")
        return None
    
    # Ensure dump directory exists
    DEBUG_DUMP_LOCATION.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = int(time.time() * 1000)
    if suffix:
        filename = f"{prefix}_{suffix}_{timestamp}.jpg"
    else:
        filename = f"{prefix}_{timestamp}.jpg"
    
    filepath = DEBUG_DUMP_LOCATION / filename
    
    # Save image
    success = cv2.imwrite(str(filepath), frame)
    
    if success:
        print(f"Debug frame saved: {filepath}")
        return filepath
    else:
        print(f"Failed to save debug frame: {filepath}")
        return None


def create_info_overlay(
    frame: np.ndarray,
    fps: float = 0.0,
    detection_count: int = 0,
    extra_info: dict = None
) -> np.ndarray:
    """
    Create an information overlay on the frame
    
    Args:
        frame: Input frame
        fps: Current FPS
        detection_count: Number of detections
        extra_info: Dictionary of additional info to display
    
    Returns:
        Frame with info overlay
    """
    output = frame.copy()
    
    # Build info lines
    lines = []
    
    if fps > 0:
        lines.append(f"FPS: {fps:.1f}")
    
    lines.append(f"Eyes detected: {detection_count}")
    
    if extra_info:
        for key, value in extra_info.items():
            lines.append(f"{key}: {value}")
    
    # Draw info
    draw_multiline_text(
        output, 
        lines, 
        position=(10, 30),
        color=(0, 255, 0),
        background=True
    )
    
    return output


def resize_frame(
    frame: np.ndarray, 
    max_width: int = None, 
    max_height: int = None,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize frame to fit within maximum dimensions
    
    Args:
        frame: Input frame
        max_width: Maximum width (None = no limit)
        max_height: Maximum height (None = no limit)
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if max_width is None and max_height is None:
        return frame
    
    if maintain_aspect:
        # Calculate scale factor
        scale = 1.0
        
        if max_width and w > max_width:
            scale = min(scale, max_width / w)
        
        if max_height and h > max_height:
            scale = min(scale, max_height / h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w = max_width if max_width else w
        new_h = max_height if max_height else h
        
        if new_w != w or new_h != h:
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return frame


def draw_crosshair(
    frame: np.ndarray,
    center: Tuple[int, int],
    size: int = 20,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """
    Draw a crosshair at specified position
    
    Useful for marking pupil centers or points of interest.
    
    Args:
        frame: Image to draw on (modified in place)
        center: (x, y) position of crosshair center
        size: Size of crosshair arms
        color: Color in BGR format
        thickness: Line thickness
    """
    x, y = center
    
    # Draw horizontal line
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    
    # Draw vertical line
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


class PerformanceMonitor:
    """
    Monitor performance metrics
    
    Tracks timing information for different processing stages.
    """
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, name: str):
        """Start timing a section"""
        self.start_times[name] = time.time()
    
    def end(self, name: str):
        """End timing a section and record duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            
            if name not in self.timings:
                self.timings[name] = []
            
            self.timings[name].append(duration)
            del self.start_times[name]
    
    def get_average(self, name: str) -> float:
        """Get average time for a section"""
        if name in self.timings and self.timings[name]:
            return sum(self.timings[name]) / len(self.timings[name])
        return 0.0
    
    def get_report(self) -> dict:
        """Get performance report"""
        report = {}
        for name, times in self.timings.items():
            if times:
                report[name] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        return report
    
    def reset(self):
        """Reset all timings"""
        self.timings.clear()
        self.start_times.clear()
    
    def print_report(self):
        """Print performance report to console"""
        report = self.get_report()
        
        if not report:
            print("No performance data collected")
            return
        
        print("\n" + "=" * 60)
        print("Performance Report")
        print("=" * 60)
        
        for name, stats in report.items():
            print(f"\n{name}:")
            print(f"  Average: {stats['average']*1000:.2f} ms")
            print(f"  Min: {stats['min']*1000:.2f} ms")
            print(f"  Max: {stats['max']*1000:.2f} ms")
            print(f"  Count: {stats['count']}")
        
        print("=" * 60 + "\n")
