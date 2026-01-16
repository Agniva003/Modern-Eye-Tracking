"""
Main entry point for Eye Tracker application

Usage:
    python main.py                          # Use camera (default)
    python main.py --frame-source file      # Use static file
    python main.py --frame-source folder    # Use folder of images
    python main.py --simple                 # Use simple detector
    python main.py --debug                  # Enable debug mode
"""

import argparse
import sys
import os
from pathlib import Path

# Try to import cv2 and provide helpful error message
try:
    import cv2
except ImportError as e:
    print("\n" + "="*60)
    print("ERROR: OpenCV import failed!")
    print("="*60)
    if "libGL.so.1" in str(e):
        print("\nYou're in a headless environment (like Codespaces).")
        print("\nTo fix this, run:")
        print("  chmod +x install.sh")
        print("  ./install.sh")
        print("\nOr manually:")
        print("  pip uninstall opencv-python")
        print("  pip install opencv-python-headless")
    else:
        print(f"\nError details: {e}")
    print("="*60 + "\n")
    sys.exit(1)

from settings import (
    WINDOW_NAME, 
    DISPLAY_FPS, 
    DISPLAY_DETECTION_COUNT,
    DEBUG_MODE
)
from frame_source import create_frame_source
from eye_detector import EyeDetector, SimpleEyeDetector
from utils import (
    FPSCounter, 
    draw_text, 
    save_debug_frame,
    create_info_overlay,
    PerformanceMonitor
)


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Eye Tracker - Modular eye tracking using OpenCV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Use camera (default)
  python main.py --frame-source file      # Use static file
  python main.py --frame-source folder    # Use folder of images
  python main.py --simple                 # Use simple detector
  python main.py --debug                  # Enable debug mode
  python main.py --perf                   # Show performance metrics
  
Keyboard Controls:
  q - Quit application
  s - Save debug frame
  p - Print performance report
  r - Reset performance stats
        """
    )
    
    parser.add_argument(
        '--frame-source',
        type=str,
        choices=['camera', 'file', 'folder'],
        default='camera',
        help='Source of frames (default: camera)'
    )
    
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple eye detector (for close-up eye images)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose output'
    )
    
    parser.add_argument(
        '--perf',
        action='store_true',
        help='Enable performance monitoring'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without displaying window (for testing)'
    )
    
    parser.add_argument(
        '--camera-index',
        type=int,
        default=None,
        help='Camera device index (overrides settings)'
    )
    
    parser.add_argument(
        '--file-path',
        type=str,
        default=None,
        help='Path to image file (for file source)'
    )
    
    parser.add_argument(
        '--folder-path',
        type=str,
        default=None,
        help='Path to folder with images (for folder source)'
    )
    
    return parser.parse_args()


def print_startup_info(args):
    """Print startup information"""
    print("\n" + "=" * 60)
    print("Eye Tracker - Starting...")
    print("=" * 60)
    print(f"Frame source: {args.frame_source}")
    print(f"Detector type: {'Simple' if args.simple else 'Full (Haar Cascade + Blob)'}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"Performance monitoring: {'Enabled' if args.perf else 'Disabled'}")
    print(f"Display mode: {'Headless (no window)' if args.no_display else 'GUI'}")
    
    # Detect if running in headless environment
    if not args.no_display:
        if os.environ.get('DISPLAY') is None and sys.platform != 'win32':
            print("\n⚠️  WARNING: No DISPLAY detected (headless environment)")
            print("   GUI display may not work. Consider using --no-display flag.")
    
    print("=" * 60)
    print("\nKeyboard Controls:")
    if not args.no_display:
        print("  q - Quit application")
        print("  s - Save debug frame")
    else:
        print("  Ctrl+C - Stop application")
    if args.perf:
        print("  p - Print performance report")
        print("  r - Reset performance stats")
    print("=" * 60 + "\n")


def create_components(args):
    """
    Create application components based on arguments
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        tuple: (frame_source, detector, fps_counter, perf_monitor)
    
    Raises:
        RuntimeError: If component initialization fails
    """
    try:
        # Create frame source
        source_kwargs = {}
        
        if args.frame_source == 'camera' and args.camera_index is not None:
            source_kwargs['camera_index'] = args.camera_index
        elif args.frame_source == 'file' and args.file_path is not None:
            source_kwargs['file_path'] = Path(args.file_path)
        elif args.frame_source == 'folder' and args.folder_path is not None:
            source_kwargs['folder_path'] = Path(args.folder_path)
        
        frame_source = create_frame_source(args.frame_source, **source_kwargs)
        
        # Create detector
        detector = SimpleEyeDetector() if args.simple else EyeDetector()
        
        # Create FPS counter
        fps_counter = FPSCounter()
        
        # Create performance monitor
        perf_monitor = PerformanceMonitor() if args.perf else None
        
        return frame_source, detector, fps_counter, perf_monitor
    
    except Exception as e:
        raise RuntimeError(f"Failed to initialize components: {e}") from e


def process_keyboard_input(key: int, frame, perf_monitor=None) -> bool:
    """
    Process keyboard input
    
    Args:
        key: Key code from cv2.waitKey()
        frame: Current frame (for saving)
        perf_monitor: Performance monitor instance
    
    Returns:
        bool: True to continue, False to quit
    """
    if key == ord('q') or key == 27:  # 'q' or ESC
        print("\nQuitting...")
        return False
    
    elif key == ord('s'):
        save_debug_frame(frame, prefix="manual_save")
    
    elif key == ord('p') and perf_monitor:
        perf_monitor.print_report()
    
    elif key == ord('r') and perf_monitor:
        perf_monitor.reset()
        print("Performance stats reset")
    
    return True


def main():
    """
    Main application loop
    
    Returns:
        int: Exit code (0 = success, 1 = error)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print startup info
    if args.debug or DEBUG_MODE:
        print_startup_info(args)
    
    # Initialize components
    try:
        frame_source, detector, fps_counter, perf_monitor = create_components(args)
    except Exception as e:
        print(f"\nError during initialization: {e}")
        return 1
    
    # Main processing loop
    frame_count = 0
    
    try:
        while True:
            if perf_monitor:
                perf_monitor.start('total')
            
            # Get frame
            if perf_monitor:
                perf_monitor.start('frame_acquisition')
            
            frame = frame_source.get_frame()
            
            if perf_monitor:
                perf_monitor.end('frame_acquisition')
            
            if frame is None:
                print("\nFailed to get frame. Exiting...")
                break
            
            # Process frame
            if perf_monitor:
                perf_monitor.start('detection')
            
            processed_frame, detections = detector.process_frame(frame)
            
            if perf_monitor:
                perf_monitor.end('detection')
            
            # Update FPS
            if perf_monitor:
                perf_monitor.start('rendering')
            
            fps = fps_counter.update()
            
            # Create info overlay
            if DISPLAY_FPS or DISPLAY_DETECTION_COUNT:
                extra_info = {}
                
                if args.perf:
                    extra_info['Frame'] = frame_count
                
                processed_frame = create_info_overlay(
                    processed_frame,
                    fps=fps if DISPLAY_FPS else 0,
                    detection_count=len(detections) if DISPLAY_DETECTION_COUNT else 0,
                    extra_info=extra_info if extra_info else None
                )
            
            # Display frame
            if not args.no_display:
                try:
                    cv2.imshow(WINDOW_NAME, processed_frame)
                except cv2.error as e:
                    if frame_count == 1:  # Only show error once
                        print(f"\n⚠️  Cannot display window: {e}")
                        print("Running in headless mode. Use --no-display to suppress this warning.\n")
                    args.no_display = True  # Switch to headless mode
            else:
                # In headless mode, save frame periodically
                if frame_count % 30 == 0:  # Every 30 frames
                    save_debug_frame(processed_frame, prefix=f"processed_frame_{frame_count}")
            
            if perf_monitor:
                perf_monitor.end('rendering')
            
            # Handle keyboard input
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not process_keyboard_input(key, processed_frame, perf_monitor):
                        break
            else:
                # In headless mode, run for limited frames or until interrupted
                if args.frame_source == 'file' and frame_count >= 1:
                    print(f"Processed static image. Saved to debug folder.")
                    break
                elif frame_count >= 100 and args.frame_source == 'folder':
                    print(f"Processed {frame_count} frames from folder.")
                    break
            
            if perf_monitor:
                perf_monitor.end('total')
            
            frame_count += 1
            
            # Debug output every 100 frames
            if args.debug and frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, FPS: {fps:.1f}, "
                      f"Detections: {len(detections)}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        
        # Try to save debug frame
        try:
            if frame is not None:
                save_debug_frame(frame, prefix="error", suffix="crash")
        except:
            pass
        
        if args.debug:
            import traceback
            traceback.print_exc()
        
        return 1
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        
        # Print final performance report
        if perf_monitor and frame_count > 0:
            print(f"\nProcessed {frame_count} frames total")
            perf_monitor.print_report()
        
        # Release resources
        frame_source.release()
        cv2.destroyAllWindows()
        
        print("Cleanup complete")
    
    print("\nEye Tracker finished successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
