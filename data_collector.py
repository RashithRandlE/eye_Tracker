import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random
import csv
import os
import subprocess

def get_screen_size():
    try:
        output = subprocess.check_output("xrandr | grep '\*' | awk '{print $1}'", shell=True).decode().split('\n')[0]
        w, h = map(int, output.split('x'))
        return w, h
    except:
        return 1920, 1080

def main():
    screen_w, screen_h = get_screen_size()
    print(f"Screen Size: {screen_w}x{screen_h}")
    
    # Initialize MediaPipe
    model_path = 'face_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    
    csv_file = open('eye_data.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # We will save the Screen Target Coordinates, and the 3D position of 6 key eye features.
    # This solves the head movement issue by giving the AI reference points (the corners of the eye)
    headers = ['Screen_X', 'Screen_Y']
    points_of_interest = {
        'L_Iris': 468,
        'L_Inner': 133, 
        'L_Outer': 33,
        'R_Iris': 473,
        'R_Inner': 362,
        'R_Outer': 263
    }
    for name in points_of_interest.keys():
        headers.extend([f'{name}_x', f'{name}_y', f'{name}_z'])
    
    csv_writer.writerow(headers)
    
    # Create Fullscreen Black Window
    cv2.namedWindow('Data Collector', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Data Collector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    target_x = random.randint(50, screen_w - 50)
    target_y = random.randint(50, screen_h - 50)
    
    collected_points = 0
    TOTAL_POINTS = 50 # 50 is a great start. You can increase it later for more accuracy.
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = detector.detect(mp_image)
        
        # Draw Black Screen
        display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        # Draw Target Dot
        cv2.circle(display, (target_x, target_y), 15, (0, 0, 255), -1)
        
        # Draw Instructions
        cv2.putText(display, f"Points collected: {collected_points}/{TOTAL_POINTS}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "1. STARE exactly at the RED DOT.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(display, "2. KEEP STARING at it and press SPACEBAR.", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(display, "3. Try moving your head slightly between dots to train it better!", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow('Data Collector', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            print("Exiting early...")
            break
        elif key == 32: # SPACEBAR
            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                row = [target_x, target_y]
                for idx in points_of_interest.values():
                    lm = landmarks[idx]
                    row.extend([lm.x, lm.y, lm.z])
                csv_writer.writerow(row)
                collected_points += 1
                
                if collected_points >= TOTAL_POINTS:
                    print(f"Data collection complete! Saved {TOTAL_POINTS} rows to eye_data.csv")
                    break
                
                # Move dot to new location
                target_x = random.randint(50, screen_w - 50)
                target_y = random.randint(50, screen_h - 50)
            else:
                print("Face not detected! Make sure you are in frame.")

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
