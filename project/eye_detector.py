"""
Eye detection and tracking module

Provides two detection modes:
1. EyeDetector: Full face/eye detection with Haar Cascades + blob detection
2. SimpleEyeDetector: Direct blob detection for close-up eye images
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

from settings import (
    MIN_BLOB_AREA, 
    MAX_BLOB_AREA,
    BLOB_THRESHOLD_START, 
    BLOB_THRESHOLD_END, 
    BLOB_THRESHOLD_STEP,
    MIN_CIRCULARITY,
    MIN_CONVEXITY,
    MIN_INERTIA_RATIO,
    EYE_CASCADE_SCALE_FACTOR,
    EYE_CASCADE_MIN_NEIGHBORS,
    EYE_CASCADE_MIN_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    GAUSSIAN_BLUR_KERNEL,
    COLOR_EYE_BOX,
    COLOR_PUPIL,
    COLOR_PUPIL_CENTER,
    DEBUG_MODE
)


class EyeDetector:
    """
    Detects and tracks eye pupils using Haar Cascade + blob detection
    
    This detector:
    1. Finds eye regions using Haar Cascade
    2. Preprocesses each eye region
    3. Detects pupils using SimpleBlobDetector
    """
    
    def __init__(self):
        """Initialize eye detector with Haar Cascade and blob detector"""
        
        if DEBUG_MODE:
            print("Initializing EyeDetector...")
        
        # Setup SimpleBlobDetector parameters
        self.blob_params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        self.blob_params.filterByArea = True
        self.blob_params.minArea = MIN_BLOB_AREA
        self.blob_params.maxArea = MAX_BLOB_AREA
        
        # Filter by circularity
        self.blob_params.filterByCircularity = True
        self.blob_params.minCircularity = MIN_CIRCULARITY
        
        # Filter by convexity
        self.blob_params.filterByConvexity = True
        self.blob_params.minConvexity = MIN_CONVEXITY
        
        # Filter by inertia
        self.blob_params.filterByInertia = True
        self.blob_params.minInertiaRatio = MIN_INERTIA_RATIO
        
        # Create blob detector
        self.detector = cv2.SimpleBlobDetector_create(self.blob_params)
        
        # Load Haar Cascade for eye detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.eye_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from {cascade_path}\n"
                f"Please check OpenCV installation."
            )
        
        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        
        if DEBUG_MODE:
            print("EyeDetector initialized successfully")
            print(f"Blob area range: {MIN_BLOB_AREA} - {MAX_BLOB_AREA}")
            print(f"Circularity threshold: {MIN_CIRCULARITY}")
    
    def detect_eye_region(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect eye regions using Haar Cascade
        
        Args:
            frame: Input image in BGR format
        
        Returns:
            List of (x, y, width, height) tuples for each detected eye
        """
        # Convert to grayscale for cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=EYE_CASCADE_SCALE_FACTOR,
            minNeighbors=EYE_CASCADE_MIN_NEIGHBORS,
            minSize=EYE_CASCADE_MIN_SIZE
        )
        
        return eyes
    
    def preprocess_eye_region(self, eye_region: np.ndarray) -> np.ndarray:
        """
        Preprocess eye region for better blob detection
        
        Steps:
        1. Convert to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Enhance contrast using CLAHE
        
        Args:
            eye_region: Eye region in BGR format
        
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
        
        # Enhance contrast using CLAHE
        enhanced = self.clahe.apply(blurred)
        
        return enhanced
    
    def detect_pupil(self, eye_region: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Detect pupil in eye region using blob detection
        
        Tries multiple threshold values to find the best blob.
        
        Args:
            eye_region: Eye region in BGR format
        
        Returns:
            Tuple of (x, y, radius) if pupil found, None otherwise
        """
        # Preprocess the eye region
        preprocessed = self.preprocess_eye_region(eye_region)
        
        best_blob = None
        best_blob_size = 0
        
        # Try multiple threshold values
        for threshold in range(BLOB_THRESHOLD_START, BLOB_THRESHOLD_END, BLOB_THRESHOLD_STEP):
            # Apply threshold (inverse: dark objects on light background)
            _, binary = cv2.threshold(
                preprocessed, 
                threshold, 
                255, 
                cv2.THRESH_BINARY_INV
            )
            
            # Detect blobs
            keypoints = self.detector.detect(binary)
            
            if keypoints:
                # Find the largest blob
                for kp in keypoints:
                    if kp.size > best_blob_size:
                        best_blob = kp
                        best_blob_size = kp.size
        
        if best_blob:
            return (best_blob.pt[0], best_blob.pt[1], best_blob.size / 2)
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process frame and detect eyes with pupils
        
        Args:
            frame: Input frame in BGR format
        
        Returns:
            Tuple of (annotated_frame, list_of_detections)
            Each detection is a dict with 'eye_region', 'pupil_center', 'pupil_radius'
        """
        output_frame = frame.copy()
        detections = []
        
        # Detect eye regions
        eyes = self.detect_eye_region(frame)
        
        for (ex, ey, ew, eh) in eyes:
            # Draw eye region rectangle
            cv2.rectangle(
                output_frame, 
                (ex, ey), 
                (ex + ew, ey + eh), 
                COLOR_EYE_BOX, 
                2
            )
            
            # Extract eye region
            eye_region = frame[ey:ey+eh, ex:ex+ew]
            
            if eye_region.size == 0:
                continue
            
            # Detect pupil in eye region
            pupil = self.detect_pupil(eye_region)
            
            if pupil:
                px, py, pr = pupil
                
                # Convert to frame coordinates
                px_frame = int(ex + px)
                py_frame = int(ey + py)
                pr_int = int(pr)
                
                # Draw pupil circle
                cv2.circle(
                    output_frame, 
                    (px_frame, py_frame), 
                    pr_int, 
                    COLOR_PUPIL, 
                    2
                )
                
                # Draw pupil center point
                cv2.circle(
                    output_frame, 
                    (px_frame, py_frame), 
                    2, 
                    COLOR_PUPIL_CENTER, 
                    -1
                )
                
                detections.append({
                    'eye_region': (ex, ey, ew, eh),
                    'pupil_center': (px_frame, py_frame),
                    'pupil_radius': pr
                })
        
        return output_frame, detections


class SimpleEyeDetector:
    """
    Simplified eye detector for close-up eye images
    
    This detector assumes the entire frame is an eye and directly
    applies blob detection without Haar Cascade.
    
    Useful for:
    - Pre-cropped eye images
    - Close-up eye shots
    - Testing and development
    """
    
    def __init__(self):
        """Initialize simple eye detector with blob detector only"""
        
        if DEBUG_MODE:
            print("Initializing SimpleEyeDetector...")
        
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = MIN_BLOB_AREA
        params.maxArea = MAX_BLOB_AREA
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = MIN_CIRCULARITY
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = MIN_CONVEXITY
        
        # Filter by inertia
        params.filterByInertia = True
        params.minInertiaRatio = MIN_INERTIA_RATIO
        
        # Create detector
        self.detector = cv2.SimpleBlobDetector_create(params)
        
        if DEBUG_MODE:
            print("SimpleEyeDetector initialized successfully")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process frame assuming it's already an eye image
        
        Args:
            frame: Input frame (assumed to be close-up of eye)
        
        Returns:
            Tuple of (annotated_frame, list_of_detections)
        """
        output_frame = frame.copy()
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
        
        best_blob = None
        best_blob_size = 0
        
        # Try multiple thresholds
        for threshold in range(BLOB_THRESHOLD_START, BLOB_THRESHOLD_END, BLOB_THRESHOLD_STEP):
            # Apply threshold
            _, binary = cv2.threshold(
                blurred, 
                threshold, 
                255, 
                cv2.THRESH_BINARY_INV
            )
            
            # Detect blobs
            keypoints = self.detector.detect(binary)
            
            if keypoints:
                for kp in keypoints:
                    if kp.size > best_blob_size:
                        best_blob = kp
                        best_blob_size = kp.size
        
        if best_blob:
            px, py = best_blob.pt
            pr = best_blob.size / 2
            
            px_int = int(px)
            py_int = int(py)
            pr_int = int(pr)
            
            # Draw pupil circle
            cv2.circle(output_frame, (px_int, py_int), pr_int, COLOR_PUPIL, 2)
            
            # Draw pupil center
            cv2.circle(output_frame, (px_int, py_int), 2, COLOR_PUPIL_CENTER, -1)
            
            detections.append({
                'pupil_center': (px_int, py_int),
                'pupil_radius': pr
            })
        
        return output_frame, detections
