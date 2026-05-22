# Real-Time Drowsiness Detection System using MediaPipe Tasks API

A lightweight, robust, and modern computer vision application built to combat driver fatigue. This system tracks facial landmarks in real time via a webcam feed, calculates an average Eye Aspect Ratio (EAR) approximation across both eyes, and triggers an instant verbal audio alert when signs of drowsiness (extended eye closure) are detected.

---

## 🚀 Key Features & Research Contributions

Unlike legacy drowsiness detection tutorials that rely on outdated frameworks, this project incorporates modern architectural updates and native hardware optimizations:

* **Dual-Eye Tracking Integration:** Tracks both eyes simultaneously to ensure accuracy even if the head is tilted or if shadows obscure one side of the face.
* **MediaPipe Tasks Migration:** Upgraded from the deprecated `mp.solutions.face_mesh` pipeline to the modern, faster **MediaPipe Tasks API** using the localized `face_landmarker.task` model file.
* **Python 3.13 Compatibility:** Optimized for modern environments by dropping broken legacy packages (like `playsound`) that suffer from `OSError: could not get source code` wheel compilation bugs on Python 3.13.
* **Zero-Dependency Native Mac Audio Engine:** Integrated a non-blocking background `threading` structure that calls macOS's native Text-to-Speech (`say` utility) via system pipes. This eliminates audio playback lag and prevents webcam frame freezes.
* **Instant Thread-Lock Reset:** Customized state logic to immediately clear background audio locks (`sound_playing = False`) the exact millisecond the user opens their eyes, allowing seamless back-to-back alerts.

---

## 📐 How the Eye Tracking Logic Works

The application monitors highly specific coordinate points in the facial mesh structure. By tracking the vertical Euclidean distance between the upper and lower eyelids on both the left and right eyes, it calculates a real-time tracking metric.

### Landmark Mapping Structure

* **Left Eye Coordinates:** Top Vertex (`386`), Bottom Vertex (`374`)
* **Right Eye Coordinates:** Top Vertex (`159`), Bottom Vertex (`145`)

[159] (Right Top)                 [386] (Left Top)
        .-------.                         .-------.
       /         \                       /         \
      '-----------'                     '-----------'
      [145] (Right Bottom)              [374] (Left Bottom)
The system calculates the vertical Euclidean distance using the standard distance formula:
$$\\text{Distance} = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

If the average distance falls below the fine-tuned threshold of `0.02` for more than **1.0 consecutive second**, a thread is spawned to command the system to alert the user.

---

## 📊 Visual Performance & Testing

### 1. Alert Status (Eyes Open)
When the eyes are open, the system maps the points flawlessly and displays a blue safety indicator.

![Awake and Alert Status](Screenshot 2026-05-22 at 2.16.49 PM.jpg)

### 2. Drowsiness Captured (Eyes Closed)
Upon closing the eyes, the facial landmarker remains locked onto the eyelids, the distance drops below the calibrated threshold, and the state machine transitions to the warning phase.

![Eyes Closed Detection](Screenshot 2026-05-22 at 2.18.34 PM.jpg)

---

## 🛠️ Project Setup & Installation

### Prerequisites
* macOS (Tested successfully on Apple Silicon M1/M2/M3 chips)
* Python 3.13+

### 1. Repository Setup
Ensure your local project folder is structured correctly:
```text
app/
├── main.py
└── face_landmarker.task
I. Dependency Installation
Install the modern foundational computer vision and machine learning frameworks:
 -> python3 -m pip install opencv-python mediapipe
II. Fetch the Model Asset
Download Google's official pre-trained Face Landmarker task bundle into your local directory:
 -> curl -o face_landmarker.task [https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
III. Running the System
Execute the script directly from your terminal:
 -> python3 main.py
