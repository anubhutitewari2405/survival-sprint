# Survival-Sprint
Survial Sprint or Red-Green Light Game using OpenCV, YOLOv8, DeepSORT, CNN, and reinforcement learning (DQN). Tracks player movement in real time and eliminates motion during red-light phases.<br>
# project description<br>
This project implements an AI-powered Red-Green Light Game in four progressive development phases. Inspired by the classic game, it simulates real-time motion regulation using computer vision, deep learning, and reinforcement learning. The system evolves from basic motion detection to an integrated, multi-agent, intelligent game environment.<br>
## *Phase 1* <br>
*Basic Motion Detection with OpenCV*<br>
1. Implements a single-player Red-Green Light game using a webcam.
2. Detects motion via frame differencing and background subtraction.
3. Penalizes movement during red-light periods.
4. Built with Tkinter for GUI and OpenCV for video processing.<br>
## *Phase 2*
*Multi-Player Detection with YOLOv8 and DeepSORT*
1. Adds YOLOv8 for real-time person detection.
2. Uses DeepSORT to assign unique IDs and track multiple players.
3. Players are monitored during red-light phases using patch comparison.
4. Enhanced feedback with audio cues and status overlays.<br>
## *Phase 3*<br>
*Movement Classification & RL Agent*<br>
1. Face images of tracked players are captured and labeled as movement or no_movement.
2. Trains a CNN (movement_classifier.h5) to predict movement from face images.
3. Evaluated with a test dataset created by converting YOLO-labeled data.
4. Implements a Gym environment for Red-Green Light logic.
5. Trains a DQN agent to learn optimal behavior during light transitions.<br>
## *Phase 4*<br>
*Full Integration & GUI Game*<br>
1.Combines all components into a Tkinter-based game with:
    a.Live camera feed
    b.Light switching logic with sound
    c.Real-time YOLOv8 + DeepSORT tracking
    d.CNN-based movement detection
    e.Player leaderboard and graph of active players over time
2.Runs entirely in real time and supports multiple players simultaneously.<br>
## *Technologies Used*<br>
Python, OpenCV, Tkinter,YOLOv8 (Ultralytics), DeepSORT,TensorFlow ,(DQN),Gymnasium,Matplotlib <br>
##**Acess of the entire code**<br>
https://drive.google.com/file/d/1KRmcDLFbJwd3FBnqNUuLf4ANPaEqZmEa/view?usp=sharing
