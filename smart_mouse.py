import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
import subprocess
import evdev
from evdev import UInput, ecodes as e

def get_screen_size():
    try:
        output = subprocess.check_output("xrandr | grep '\*' | awk '{print $1}'", shell=True).decode().split('\n')[0]
        w, h = map(int, output.split('x'))
        return w, h
    except:
        return 1920, 1080

def main():
    screen_w, screen_h = get_screen_size()
    
    # 1. Load the AI Models you just trained
    print("Loading AI Models...")
    try:
        with open('model_x.pkl', 'rb') as f:
            model_x = pickle.load(f)
        with open('model_y.pkl', 'rb') as f:
            model_y = pickle.load(f)
    except FileNotFoundError:
        print("Error: Models not found. Run train_model.py first!")
        return

    # 2. Initialize MediaPipe
    model_path = 'face_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    # 3. Set up Virtual Hardware Mouse via evdev
    cap_evdev = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
        e.EV_ABS: [
            (e.ABS_X, evdev.AbsInfo(value=0, min=0, max=screen_w, fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, evdev.AbsInfo(value=0, min=0, max=screen_h, fuzz=0, flat=0, resolution=0))
        ]
    }
    ui = UInput(cap_evdev, name='EyeTracker-Smart-Mouse')

    # Smoothing configuration (Adjust this from 0.05 to 1.0 depending on how jittery you want it)
    SMOOTHING_FACTOR = 0.2
    smooth_x = None
    smooth_y = None

    points_of_interest = {
        'L_Iris': 468,
        'L_Inner': 133, 
        'L_Outer': 33,
        'R_Iris': 473,
        'R_Inner': 362,
        'R_Outer': 263
    }

    print("\n===============================")
    print("      SMART MOUSE ACTIVE!      ")
    print("===============================\n")
    print("Press ESC in the video window to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = detector.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            # Extract the exact same 18 features used during training
            features = []
            for idx in points_of_interest.values():
                lm = landmarks[idx]
                features.extend([lm.x, lm.y, lm.z])
            
            # Predict Screen X and Y using your custom AI!
            pred_x = model_x.predict([features])[0]
            pred_y = model_y.predict([features])[0]
            
            target_x = max(0, min(screen_w, int(pred_x)))
            target_y = max(0, min(screen_h, int(pred_y)))

            # Apply Smoothing Filter
            if smooth_x is None:
                smooth_x, smooth_y = target_x, target_y
            else:
                smooth_x = smooth_x + (target_x - smooth_x) * SMOOTHING_FACTOR
                smooth_y = smooth_y + (target_y - smooth_y) * SMOOTHING_FACTOR

            # Move the mouse
            ui.write(e.EV_ABS, e.ABS_X, int(smooth_x))
            ui.write(e.EV_ABS, e.ABS_Y, int(smooth_y))
            ui.syn()

            # Draw visual feedback
            left_iris = landmarks[468]
            img_h, img_w = frame.shape[:2]
            lx, ly = int(left_iris.x * img_w), int(left_iris.y * img_h)
            cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)

            cv2.putText(frame, "Smart AI Tracking Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Smart Eye Tracker', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    ui.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
