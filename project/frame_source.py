"""
Frame source handlers for different input types

Provides abstract and concrete implementations for various frame sources:
- CameraSource: Real-time webcam capture
- FileSource: Static image file
- FolderSource: Sequence of images from a folder
"""

import cv2
import time
from pathlib import Path
from typing import Optional, List
import numpy as np

from settings import (
    CAMERA_INDEX, 
    CAMERA_WIDTH, 
    CAMERA_HEIGHT,
    CAMERA_FPS,
    STATIC_FILE_PATH, 
    DEBUG_DUMP_LOCATION, 
    REFRESH_PERIOD,
    DEBUG_MODE
)


class FrameSource:
    """
    Base class for frame sources
    
    All frame sources should inherit from this class and implement
    the get_frame() method.
    """
    
    def __init__(self):
        self.current_frame = None
        self.frame_count = 0
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame
        
        Returns:
            numpy.ndarray: The frame as BGR image, or None if unavailable
        """
        raise NotImplementedError("Subclasses must implement get_frame()")
    
    def release(self):
        """Release resources held by this frame source"""
        pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.release()


class CameraSource(FrameSource):
    """
    Capture frames from webcam or camera device
    
    Args:
        camera_index: Camera device index (0 for default camera)
    """
    
    def __init__(self, camera_index: int = CAMERA_INDEX):
        super().__init__()
        self.camera_index = camera_index
        
        if DEBUG_MODE:
            print(f"Initializing camera {camera_index}...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera {camera_index}. "
                f"Please check if camera is connected and not in use."
            )
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Verify actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        if DEBUG_MODE:
            print(f"Camera opened successfully")
            print(f"Resolution: {actual_width}x{actual_height}")
            print(f"FPS: {actual_fps}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture and return the next frame from camera
        
        Returns:
            numpy.ndarray: Captured frame, or None if capture failed
        """
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            self.frame_count += 1
            return frame
        else:
            if DEBUG_MODE:
                print(f"Failed to read frame from camera {self.camera_index}")
            return None
    
    def release(self):
        """Release camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            if DEBUG_MODE:
                print(f"Camera {self.camera_index} released")


class FileSource(FrameSource):
    """
    Load frame from a static image file
    
    Args:
        file_path: Path to the image file
    """
    
    def __init__(self, file_path: Path = STATIC_FILE_PATH):
        super().__init__()
        self.file_path = Path(file_path)
        
        if DEBUG_MODE:
            print(f"Loading image from: {self.file_path}")
        
        # Check if file exists
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {file_path}\n"
                f"Please place an image at this location or update STATIC_FILE_PATH in settings.py"
            )
        
        # Load image
        self.current_frame = cv2.imread(str(self.file_path))
        
        if self.current_frame is None:
            raise ValueError(
                f"Failed to load image from {file_path}\n"
                f"Please check if the file is a valid image format (jpg, png, bmp, etc.)"
            )
        
        if DEBUG_MODE:
            h, w = self.current_frame.shape[:2]
            print(f"Image loaded successfully: {w}x{h}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Return the static frame (always returns the same image)
        
        Returns:
            numpy.ndarray: Copy of the loaded image
        """
        self.frame_count += 1
        return self.current_frame.copy()


class FolderSource(FrameSource):
    """
    Iterate through images in a folder
    
    Images are loaded sequentially with a configurable delay between frames.
    
    Args:
        folder_path: Path to folder containing images
    """
    
    def __init__(self, folder_path: Path = DEBUG_DUMP_LOCATION):
        super().__init__()
        self.folder_path = Path(folder_path)
        
        if DEBUG_MODE:
            print(f"Loading images from folder: {self.folder_path}")
        
        # Check if folder exists
        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder_path}\n"
                f"Please create this folder and add images, or update the path."
            )
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        self.image_files: List[Path] = []
        
        # Collect all images
        for ext in extensions:
            self.image_files.extend(self.folder_path.glob(ext))
            # Also check uppercase extensions
            self.image_files.extend(self.folder_path.glob(ext.upper()))
        
        # Sort files by name
        self.image_files = sorted(set(self.image_files))
        
        if not self.image_files:
            raise ValueError(
                f"No images found in {folder_path}\n"
                f"Supported formats: jpg, jpeg, png, bmp, tiff, webp"
            )
        
        if DEBUG_MODE:
            print(f"Found {len(self.image_files)} images")
        
        self.current_index = 0
        self.last_update = time.time()
        
        # Load first image
        self._load_current_image()
    
    def _load_current_image(self):
        """Load the image at current index"""
        image_path = self.image_files[self.current_index]
        self.current_frame = cv2.imread(str(image_path))
        
        if self.current_frame is None:
            print(f"Warning: Failed to load {image_path}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame from folder
        
        Advances to next image based on REFRESH_PERIOD setting.
        
        Returns:
            numpy.ndarray: Current frame
        """
        current_time = time.time()
        
        # Check if it's time to advance to next image
        if current_time - self.last_update >= REFRESH_PERIOD:
            self.last_update = current_time
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self._load_current_image()
            self.frame_count += 1
            
            if DEBUG_MODE:
                print(f"Loading image {self.current_index + 1}/{len(self.image_files)}")
        
        return self.current_frame.copy() if self.current_frame is not None else None


def create_frame_source(source_type: str, **kwargs) -> FrameSource:
    """
    Factory function to create frame sources
    
    Args:
        source_type: Type of frame source ('camera', 'file', or 'folder')
        **kwargs: Additional arguments to pass to the frame source constructor
    
    Returns:
        FrameSource: Instance of the requested frame source
    
    Raises:
        ValueError: If source_type is not recognized
    
    Examples:
        >>> source = create_frame_source('camera')
        >>> source = create_frame_source('file', file_path='my_image.jpg')
        >>> source = create_frame_source('folder', folder_path='my_images/')
    """
    sources = {
        'camera': CameraSource,
        'file': FileSource,
        'folder': FolderSource
    }
    
    if source_type not in sources:
        raise ValueError(
            f"Unknown source type: '{source_type}'. "
            f"Available options: {', '.join(sources.keys())}"
        )
    
    try:
        return sources[source_type](**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create {source_type} source: {e}"
        ) from e
