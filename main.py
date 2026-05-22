import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import threading
import os

# Indices for Left Eye landmarks
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

# Indices for Right Eye landmarks
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Thresholds 
CLOSE_THRESHOLD = 0.02
CLOSED_TIME_LIMIT = 1.0   

# State variables
eye_closed_start_time = None
alarm_active = False
sound_playing = False

def play_alarm_sound():
    """Background worker function using native Mac text-to-speech."""
    global sound_playing
    sound_playing = True
    try:
        # Shortened phrase so the background thread resets faster
        os.system('say "Wake up!"')
    except Exception as e:
        print(f"Audio Error: {e}")
    sound_playing = False

# Initialize Face Landmarker API
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        
        # 1. Left Eye Calculation
        left_top = face_landmarks[LEFT_EYE_TOP]
        left_bot = face_landmarks[LEFT_EYE_BOTTOM]
        left_dist = ((left_top.x - left_bot.x)**2 + (left_top.y - left_bot.y)**2)**0.5
        
        # 2. Right Eye Calculation
        right_top = face_landmarks[RIGHT_EYE_TOP]
        right_bot = face_landmarks[RIGHT_EYE_BOTTOM]
        right_dist = ((right_top.x - right_bot.x)**2 + (right_top.y - right_bot.y)**2)**0.5
        
        # 3. Average the distance of both eyes
        avg_distance = (left_dist + right_dist) / 2
        
        # Draw green dots on LEFT eye
        cv2.circle(frame, (int(left_top.x * w), int(left_top.y * h)), 3, (0, 255, 0), -1)
        cv2.circle(frame, (int(left_bot.x * w), int(left_bot.y * h)), 3, (0, 255, 0), -1)
        
        # Draw green dots on RIGHT eye
        cv2.circle(frame, (int(right_top.x * w), int(right_top.y * h)), 3, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_bot.x * w), int(right_bot.y * h)), 3, (0, 255, 0), -1)

        # Alarm logic using the average distance
        if avg_distance < CLOSE_THRESHOLD:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            else:
                elapsed_time = time.time() - eye_closed_start_time
                if elapsed_time > CLOSED_TIME_LIMIT:
                    alarm_active = True
                    if not sound_playing:
                        threading.Thread(target=play_alarm_sound, daemon=True).start()
        else:
            eye_closed_start_time = None
            alarm_active = False
            sound_playing = False  # FIXED: Frees the audio lock immediately when eyes open!

        if alarm_active:
            cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif eye_closed_start_time is not None:
            cv2.putText(frame, "Eyes Closed...", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Awake and Alert", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Two-Eye Drowsiness Detector', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()