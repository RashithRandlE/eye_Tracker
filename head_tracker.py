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
    
    # Initialize MediaPipe
    model_path = 'face_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    # Set up Virtual Hardware Mouse via evdev
    cap_evdev = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
        e.EV_ABS: [
            (e.ABS_X, evdev.AbsInfo(value=int(screen_w/2), min=0, max=screen_w, fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, evdev.AbsInfo(value=int(screen_h/2), min=0, max=screen_h, fuzz=0, flat=0, resolution=0))
        ]
    }
    ui = UInput(cap_evdev, name='HeadTracker-Mouse')

    # Smoothing configuration
    SMOOTHING_FACTOR = 0.3
    smooth_x = screen_w / 2
    smooth_y = screen_h / 2

    # Calibration variables
    center_nose_x = None
    center_nose_y = None
    
    # SENSITIVITY: How far you have to move your head to reach the edge of the screen
    # Lower number = less head movement required
    SENSITIVITY_X = 0.08
    SENSITIVITY_Y = 0.08

    print("\n===============================")
    print("      HEAD TRACKER ACTIVE!     ")
    print("===============================\n")
    print("1. Look straight at the center of your monitor.")
    print("2. Press 'C' to calibrate your center point.")
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
            
            # We track the tip of the nose (Landmark 4)
            nose = landmarks[4]
            current_x = nose.x
            current_y = nose.y
            
            # Draw a red dot on the nose
            img_h, img_w = frame.shape[:2]
            nx, ny = int(current_x * img_w), int(current_y * img_h)
            cv2.circle(frame, (nx, ny), 8, (0, 0, 255), -1)

            key = cv2.waitKey(5) & 0xFF
            if key == 27: # ESC
                break
            elif key == ord('c') or key == ord('C'):
                center_nose_x = current_x
                center_nose_y = current_y
                print("Center calibrated!")

            if center_nose_x is not None:
                # Calculate how far the nose moved from center
                dx = current_x - center_nose_x
                dy = current_y - center_nose_y
                
                # Convert to percentage (-0.5 to 0.5 means full left to full right)
                perc_x = dx / SENSITIVITY_X
                perc_y = dy / SENSITIVITY_Y
                
                # Shift to 0.0 to 1.0 (where 0.5 is exactly the center of the screen)
                perc_x = perc_x + 0.5
                perc_y = perc_y + 0.5
                
                # Allow a tiny bit of overshoot so it's easy to hit the corners
                perc_x = max(-0.05, min(1.05, perc_x))
                perc_y = max(-0.05, min(1.05, perc_y))
                
                target_x = int(perc_x * screen_w)
                target_y = int(perc_y * screen_h)

                # Smoothing Filter
                smooth_x = smooth_x + (target_x - smooth_x) * SMOOTHING_FACTOR
                smooth_y = smooth_y + (target_y - smooth_y) * SMOOTHING_FACTOR

                # Move Virtual Mouse
                ui.write(e.EV_ABS, e.ABS_X, int(smooth_x))
                ui.write(e.EV_ABS, e.ABS_Y, int(smooth_y))
                ui.syn()

                cv2.putText(frame, "Head Tracking Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Look at screen center and press 'C'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Head Tracker', frame)
        
        # Handle case where no face is detected but user presses ESC
        if not results.face_landmarks:
             if cv2.waitKey(5) & 0xFF == 27:
                 break

    ui.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
