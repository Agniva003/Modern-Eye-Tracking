# Eye-Tracker

Modular, Extensible Eye-Tracking solution using OpenCV and Python.

![Eye Tracker Demo](https://i.imgur.com/DQRmibk.png)

## Overview

A **very** accurate eye-tracking software that works with:
- ✅ Cross-platform (Windows, macOS, Linux)
- ✅ Works with glasses
- ✅ Low hardware requirements (works with 640×480 webcam)
- ✅ Blob detection algorithm
- ✅ Highly extensible/flexible

## Features

- Real-time eye tracking using webcam
- Pupil detection using blob detection
- Multiple frame sources (camera, file, folder)
- FPS counter
- Debug mode with frame saving
- Simple and advanced detection modes

## Requirements

- Python 3.8 or higher
- Webcam (for camera mode)

## Installation

### Windows

```bash
# Create and navigate to project directory
mkdir eye-tracker
cd eye-tracker

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
# Create and navigate to project directory
mkdir eye-tracker
cd eye-tracker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Basic Usage (Camera)

```bash
cd project
python main.py
```

### Command Line Options

```bash
# Use a static image file
python main.py --frame-source file

# Use folder of images
python main.py --frame-source folder

# Use simple detector (for close-up eye images)
python main.py --simple
```

### Keyboard Controls

- **q** - Quit application
- **s** - Save debug frame

## Project Structure

```
eye-tracker/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
└── project/
    ├── __init__.py        # Package initialization
    ├── main.py            # Main application entry point
    ├── settings.py        # Configuration settings
    ├── eye_detector.py    # Eye detection algorithms
    ├── frame_source.py    # Frame source handlers
    └── utils.py           # Utility functions
```

## Configuration

Edit `project/settings.py` to customize:

- Camera settings (resolution, index)
- Detection parameters (blob size, thresholds)
- Debug settings (dump location, debug mode)
- Display settings (FPS display, window name)

### Environment Variables

```bash
# Enable debug mode
export DEBUG_MODE=true

# Enable debug frame dumping
export DEBUG_DUMP=true

# Set camera index (default: 0)
export CAMERA_INDEX=0
```

## How It Works

1. **Frame Acquisition**: Captures frames from camera, file, or folder
2. **Eye Region Detection**: Uses Haar Cascade to locate eye regions
3. **Preprocessing**: Applies Gaussian blur and CLAHE contrast enhancement
4. **Pupil Detection**: Uses SimpleBlobDetector to find pupil
5. **Visualization**: Draws bounding boxes and pupil circles on frame

## Troubleshooting

### Camera Not Opening

```bash
# Try different camera index
python main.py --frame-source camera
# Or set environment variable
export CAMERA_INDEX=1
python main.py
```

### No Eyes Detected

- Ensure good lighting
- Position face 30-60cm from camera
- Try adjusting detection parameters in `settings.py`
- Use `--simple` mode for close-up eye images

### Performance Issues

- Lower camera resolution in `settings.py`
- Close other applications using webcam
- Update graphics drivers

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Formatting

```bash
black project/
flake8 project/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Credits

- Original project by Stepan Filonov (@stepacool)
- Updated for Python 3.8+ compatibility
- Uses OpenCV for computer vision operations

## Links

- [GitHub Repository](https://github.com/stepacool/Eye-Tracker)
- [YouTube Demo](https://youtu.be/zDN-wwd5cfo)
- [OpenCV Documentation](https://docs.opencv.org/)

## Support

For issues, questions, or contributions, please visit the [GitHub Issues](https://github.com/stepacool/Eye-Tracker/issues) page.
