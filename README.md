# python-face-recognition-machine-learning

# Face Detection Project

## Overview
This is a simple machine learning-based face detection project that can detect specific faces in videos. Users need to provide the path of the image containing the face to detect and the video file where the face detection will be performed. The project can be extended to work with real-time face detection by integrating it with a CCTV camera feed.

## Features
- **Face Detection in Videos**: Detect specific faces in video files by providing a reference image.
- **Future Expansion**: Potential for real-time face detection using CCTV cameras.
- **Face Recognition Module**: Recognize faces in images using the `faceRecognition` module.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required Python libraries (listed in `requirements.txt`)
- OpenCV (for video processing)
- A compatible operating system (Linux, Windows, or macOS)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-detection-project.git
   cd face-detection-project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Detect Faces in Videos
To use the `FaceDetectorModule`:
1. Place the reference image (containing the face to detect) in a known location.
2. Ensure the video file for detection is available.
3. Run the module with the paths to the reference image and video file:
   ```bash
   python FaceDetectorModule.py --image_path path/to/face_image.jpg --video_path path/to/video.mp4
   ```

### 2. Recognize Faces in Images
To use the `faceRecognition` module:
1. Place the image containing the face to recognize in a known location.
2. Run the module with the image path:
   ```bash
   python faceRecognition.py --image_path path/to/image.jpg
   ```

## Configuration
### Parameters for `FaceDetectorModule`
- `--image_path`: Path to the image of the face to detect (required).
- `--video_path`: Path to the video file for face detection (required).

### Parameters for `faceRecognition`
- `--image_path`: Path to the image for face recognition (required).

## Future Improvements
- Integration with CCTV cameras for real-time face detection.
- Enhanced face recognition capabilities for handling multiple faces.
- Improved accuracy with advanced models.
- User-friendly GUI for easier interaction.

## Contributing
Contributions are welcome! If you have suggestions for improvements or want to add new features:
1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes and push to your branch.
4. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or support, please contact:
- Name: Mugisha Alain
- Email: mugishalain81@gmail.com


---
We hope you find this project helpful! Happy coding!

