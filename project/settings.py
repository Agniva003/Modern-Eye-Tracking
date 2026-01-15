"""
Configuration settings for Eye Tracker

All application settings are centralized here for easy customization.
Settings can be overridden using environment variables.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Debug dump location
DEBUG_DUMP_LOCATION = PROJECT_ROOT / 'dump'

# Static file path for 'file' frame source
STATIC_FILE_PATH = PROJECT_ROOT / 'test_images' / 'eye.jpg'

# ============================================================================
# DEBUG SETTINGS
# ============================================================================

# Enable debug mode (verbose logging)
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# Enable saving debug frames on errors
DEBUG_DUMP = os.getenv('DEBUG_DUMP', 'False').lower() == 'true'

# ============================================================================
# CAMERA SETTINGS
# ============================================================================

# Camera device index (0 for default camera, 1 for external, etc.)
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Camera FPS (if supported by camera)
CAMERA_FPS = 30

# ============================================================================
# FRAME SOURCE SETTINGS
# ============================================================================

# Refresh period for folder source (seconds between images)
REFRESH_PERIOD = 0.1

# ============================================================================
# EYE DETECTION SETTINGS
# ============================================================================

# Blob detection area limits (in pixels)
MIN_BLOB_AREA = 100
MAX_BLOB_AREA = 5000

# Threshold range for blob detection
BLOB_THRESHOLD_START = 50
BLOB_THRESHOLD_END = 220
BLOB_THRESHOLD_STEP = 5

# Blob circularity (0.0 to 1.0, where 1.0 is perfect circle)
MIN_CIRCULARITY = 0.7

# Blob convexity (0.0 to 1.0)
MIN_CONVEXITY = 0.8

# Blob inertia ratio (0.0 to 1.0)
MIN_INERTIA_RATIO = 0.5

# Haar Cascade parameters
EYE_CASCADE_SCALE_FACTOR = 1.1
EYE_CASCADE_MIN_NEIGHBORS = 5
EYE_CASCADE_MIN_SIZE = (30, 30)

# CLAHE parameters for contrast enhancement
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Gaussian blur kernel size (must be odd number)
GAUSSIAN_BLUR_KERNEL = (7, 7)

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Main window name
WINDOW_NAME = 'Eye Tracker'

# Display FPS counter on screen
DISPLAY_FPS = True

# Display detection count
DISPLAY_DETECTION_COUNT = True

# Colors (BGR format)
COLOR_EYE_BOX = (0, 255, 0)      # Green for eye region
COLOR_PUPIL = (0, 0, 255)         # Red for pupil circle
COLOR_PUPIL_CENTER = (255, 0, 0)  # Blue for pupil center
COLOR_TEXT = (0, 255, 0)          # Green for text

# Text settings
TEXT_FONT = 'FONT_HERSHEY_SIMPLEX'
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# FPS smoothing factor (0.0 to 1.0, higher = smoother but less responsive)
FPS_SMOOTHING = 0.9

# Maximum frames to process per second (0 = unlimited)
MAX_FPS = 0

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize():
    """Initialize application settings and create necessary directories"""
    # Create dump directory if debug mode is enabled
    if DEBUG_DUMP:
        DEBUG_DUMP_LOCATION.mkdir(parents=True, exist_ok=True)
        print(f"Debug dump location: {DEBUG_DUMP_LOCATION}")
    
    # Create test_images directory if it doesn't exist
    test_images_dir = PROJECT_ROOT / 'test_images'
    test_images_dir.mkdir(parents=True, exist_ok=True)
    
    if DEBUG_MODE:
        print("=" * 60)
        print("Eye Tracker Configuration")
        print("=" * 60)
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Camera index: {CAMERA_INDEX}")
        print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"Debug mode: {DEBUG_MODE}")
        print(f"Debug dump: {DEBUG_DUMP}")
        print("=" * 60)


# Auto-initialize when module is imported
initialize()
