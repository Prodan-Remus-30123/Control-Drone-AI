# Control-Drone-AI
This project implements a real-time face detection and hand gesture-based control system for a drone. Using MediaPipe for gesture recognition and OpenCV for computer vision, the drone can recognize six different hand gestures to execute specific commands. The system enables autonomous navigation, tracking, and command execution using AI-powered vision.
Features

    🚁 Face Detection & Tracking: The drone locks onto and follows a detected face using real-time computer vision.
    ✋ AI Hand Gesture Recognition: Trained an AI model based on Google's MediaPipe to recognize six hand gestures.
    📡 Gesture-Controlled Drone Navigation: The system interprets hand gestures to control drone movement:
        Move Forward
        Move Backward
        Move Up
        Move Down
        Stop
        Take a Picture
    🤖 Actuator Implementation: Additional actuators integrated for enhanced control and automation.
    🖼️ Image Capture & Processing: The drone can capture images upon receiving a hand gesture command.
    🔗 Python-based Implementation: Developed using Python, OpenCV, MediaPipe, and Drone SDKs for seamless interaction.

Technologies Used

    Python 🐍
    MediaPipe (for real-time hand tracking & gesture recognition) ✋
    OpenCV (for image processing & face tracking) 📸
    Drone SDK (for drone control and navigation) 🚁
    Machine Learning (trained model for hand gesture classification) 🤖
