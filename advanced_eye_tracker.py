import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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
    
    model_path = 'face_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    cap_evdev = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
        e.EV_ABS: [
            (e.ABS_X, evdev.AbsInfo(value=int(screen_w/2), min=0, max=screen_w, fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, evdev.AbsInfo(value=int(screen_h/2), min=0, max=screen_h, fuzz=0, flat=0, resolution=0))
        ]
    }
    ui = UInput(cap_evdev, name='Perfect-Eye-Tracker')

    SMOOTHING_FACTOR = 0.1 # Very heavy smoothing for eyes
    smooth_x = screen_w / 2
    smooth_y = screen_h / 2

    calibrated = False
    top_left_eye = None
    bottom_right_eye = None

    print("\n===============================")
    print("   ADVANCED EYE TRACKER ACTIVE ")
    print("===============================\n")
    print("1. Keep your head completely still.")
    print("2. Look at TOP-LEFT corner of screen -> Press '1'")
    print("3. Look at BOTTOM-RIGHT corner -> Press '2'")
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
            
            # Left Eye Landmarks
            iris = landmarks[468]
            inner_corner = landmarks[133]
            outer_corner = landmarks[33]
            top_eyelid = landmarks[159]
            bottom_eyelid = landmarks[145]
            
            # Calculate the width and height of the eyeball
            eye_width = inner_corner.x - outer_corner.x
            eye_height = bottom_eyelid.y - top_eyelid.y
            
            if eye_width > 0 and eye_height > 0:
                # Calculate the percentage of where the iris is INSIDE the eyeball
                # This naturally cancels out your head moving around the room!
                rel_iris_x = (iris.x - outer_corner.x) / eye_width
                rel_iris_y = (iris.y - top_eyelid.y) / eye_height

                img_h, img_w = frame.shape[:2]
                cv2.circle(frame, (int(iris.x * img_w), int(iris.y * img_h)), 4, (0, 255, 0), -1)

                key = cv2.waitKey(5) & 0xFF
                if key == 27:
                    break
                elif key == ord('1'):
                    top_left_eye = (rel_iris_x, rel_iris_y)
                    print(f"Top-Left Calibrated: {top_left_eye}")
                elif key == ord('2'):
                    bottom_right_eye = (rel_iris_x, rel_iris_y)
                    print(f"Bottom-Right Calibrated: {bottom_right_eye}")

                if top_left_eye and bottom_right_eye:
                    calibrated = True

                if calibrated:
                    x_range = bottom_right_eye[0] - top_left_eye[0]
                    y_range = bottom_right_eye[1] - top_left_eye[1]
                    
                    if x_range != 0 and y_range != 0:
                        x_perc = (rel_iris_x - top_left_eye[0]) / x_range
                        y_perc = (rel_iris_y - top_left_eye[1]) / y_range
                        
                        x_perc = max(-0.1, min(1.1, x_perc))
                        y_perc = max(-0.1, min(1.1, y_perc))
                        
                        target_x = int(x_perc * screen_w)
                        target_y = int(y_perc * screen_h)

                        smooth_x = smooth_x + (target_x - smooth_x) * SMOOTHING_FACTOR
                        smooth_y = smooth_y + (target_y - smooth_y) * SMOOTHING_FACTOR

                        ui.write(e.EV_ABS, e.ABS_X, int(smooth_x))
                        ui.write(e.EV_ABS, e.ABS_Y, int(smooth_y))
                        ui.syn()

                        cv2.putText(frame, "Eye Tracking Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Look Top-Left (1) then Bottom-Right (2)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Advanced Eye Tracker', frame)
        if not results.face_landmarks and cv2.waitKey(5) & 0xFF == 27:
            break

    ui.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
