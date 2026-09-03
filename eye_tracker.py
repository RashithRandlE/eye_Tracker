import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
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

def download_model():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading Face Landmarker AI model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path

def main():
    model_path = download_model()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    screen_w, screen_h = get_screen_size()
    print(f"Screen resolution detected as: {screen_w}x{screen_h}")

    # Set up Virtual Hardware Mouse via Linux evdev
    cap_evdev = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
        e.EV_ABS: [
            (e.ABS_X, evdev.AbsInfo(value=0, min=0, max=screen_w, fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, evdev.AbsInfo(value=0, min=0, max=screen_h, fuzz=0, flat=0, resolution=0))
        ]
    }
    ui = UInput(cap_evdev, name='EyeTracker-Virtual-Mouse')

    # Calibration & Smoothing variables
    calibrated = False
    top_left_eye = None
    bottom_right_eye = None
    
    # Smoothing parameters (0.0 to 1.0, lower = smoother but slower to respond)
    SMOOTHING_FACTOR = 0.15 
    smooth_x = None
    smooth_y = None

    print("\n--- CALIBRATION INSTRUCTIONS ---")
    print("1. Look at the TOP-LEFT corner of your monitor and press '1'")
    print("2. Look at the BOTTOM-RIGHT corner of your monitor and press '2'")
    print("3. Once both are set, the mouse will start moving with your eyes!")
    print("Press 'ESC' to exit.\n")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = detector.detect(mp_image)
        
        current_iris_x, current_iris_y = None, None

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            img_h, img_w = image.shape[:2]
            
            # Left iris center
            left_iris = face_landmarks[468]
            current_iris_x = left_iris.x 
            current_iris_y = left_iris.y 
            
            lx, ly = int(current_iris_x * img_w), int(current_iris_y * img_h)
            cv2.circle(image, (lx, ly), 4, (0, 255, 0), -1)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('1') and current_iris_x is not None:
            top_left_eye = (current_iris_x, current_iris_y)
            print("Top-Left calibrated.")
        elif key == ord('2') and current_iris_x is not None:
            bottom_right_eye = (current_iris_x, current_iris_y)
            print("Bottom-Right calibrated.")

        if top_left_eye and bottom_right_eye:
            calibrated = True

        if calibrated and current_iris_x is not None:
            x_range = bottom_right_eye[0] - top_left_eye[0]
            y_range = bottom_right_eye[1] - top_left_eye[1]
            
            if x_range != 0 and y_range != 0:
                x_perc = (current_iris_x - top_left_eye[0]) / x_range
                y_perc = (current_iris_y - top_left_eye[1]) / y_range
                
                # We allow slight overshooting so you can comfortably reach the edges
                x_perc = max(-0.1, min(1.1, x_perc))
                y_perc = max(-0.1, min(1.1, y_perc))
                
                target_screen_x = int(x_perc * screen_w)
                target_screen_y = int(y_perc * screen_h)
                
                # Apply Smoothing Filter
                if smooth_x is None:
                    smooth_x, smooth_y = target_screen_x, target_screen_y
                else:
                    smooth_x = smooth_x + (target_screen_x - smooth_x) * SMOOTHING_FACTOR
                    smooth_y = smooth_y + (target_screen_y - smooth_y) * SMOOTHING_FACTOR
                
                # Inject Virtual Hardware Mouse events
                ui.write(e.EV_ABS, e.ABS_X, int(smooth_x))
                ui.write(e.EV_ABS, e.ABS_Y, int(smooth_y))
                ui.syn()
                
                cv2.putText(image, "Tracking Active (Smoothed)!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Look Top-Left + press '1'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(image, "Look Bottom-Right + press '2'", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Eye Tracker Preview', image)

    ui.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
