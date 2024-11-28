from imutils import face_utils
from scipy.spatial import distance as dist
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import os

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def draw_eye(frame, eye_points, color, is_left):
    if len(eye_points) == 6:
        for i, point in enumerate(eye_points):
            cv2.circle(frame, tuple(point), 2, color, -1)
            cv2.putText(frame, f"({point[0]},{point[1]})", 
                        (point[0]+5, point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        cv2.polylines(frame, [eye_points], True, color, 1)
        
        center = np.mean(eye_points, axis=0).astype("int")
        cv2.circle(frame, tuple(center), 3, (0, 255, 0), -1)
        
        ear = eye_aspect_ratio(eye_points)
        
        eye_label = f"{'Left' if is_left else 'Right'} Eye EAR: {ear:.2f}"
        cv2.putText(frame, eye_label, (center[0] - 20, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return center, ear
    return None, None

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
args = vars(ap.parse_args())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] opening video file...")
vs = cv2.VideoCapture(args["video"])

EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 60  

COUNTER = 0
ALARM_ON = False

start_time = time.time()  # プログラム開始時間

while True:
    ret, frame = vs.read()
    if not ret:
        print("End of video file")
        break
    
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    try:
        rects = detector(gray, 0)
    except Exception as e:
        print(f"Error in face detection: {e}")
        continue
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[42:48]
        rightEye = shape[36:42]
        
        _, left_ear = draw_eye(frame, leftEye, (0, 0, 255), True)
        _, right_ear = draw_eye(frame, rightEye, (255, 0, 0), False)
        
        if left_ear and right_ear:
            avg_ear = (left_ear + right_ear) / 2.0
            cv2.putText(frame, f"Avg EAR: {avg_ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if avg_ear < EAR_THRESHOLD:
                COUNTER += 1
                current_time = time.time() - start_time
                print(f"Drowsiness detected at {current_time:.2f} seconds")
                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        os.system('say "Wake up"')
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()