from imutils.video import VideoStream
from imutils import face_utils
from collections import deque
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import os
import tensorflow as tf  # Import TensorFlow for model loading
 
# Load the trained CNN model
print("[INFO] loading trained drowsiness detection CNN model...")
model = tf.keras.models.load_model('drowsiness_cnn_model.h5')
 
# Argument parser for the facial landmark predictor
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())
 
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
 
print("[INFO] camera sensor warming up...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)
 
# Constants
EAR_THRESHOLD = 0.2  # EAR threshold for drowsiness
EAR_BUFFER_SIZE = 60  # Number of frames for averaging
DROWSINESS_FRAME_THRESHOLD = 5  # Minimum number of frames for eyes closed to trigger drowsiness alert
 
# Variables
ear_buffer = deque(maxlen=EAR_BUFFER_SIZE)  # Circular buffer for EAR values
ALARM_ON = False
closed_eye_duration = 0  # Counter for duration of closed eyes
 
# Function to process eye images for CNN
def process_eye_for_cnn(eye_points, frame):
    # Crop and resize the eye region
    (x, y, w, h) = cv2.boundingRect(np.array(eye_points))
    eye_image = frame[y:y+h, x:x+w]
    eye_image = cv2.resize(eye_image, (64, 64))  # Resize to the input size expected by the CNN
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    eye_image = eye_image / 255.0  # Normalize the image
    eye_image = np.expand_dims(eye_image, axis=0)  # Add batch dimension
    return eye_image
 
# Start processing frames
while True:
    ret, frame = vs.read()
    if not ret:
        print("Failed to grab frame")
        break
   
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    try:
        rects = detector(gray, 0)
    except Exception as e:
        print(f"Error in face detection: {e}")
        continue
   
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
       
        leftEye = shape[:6]
        rightEye = shape[6:12]
       
        # Process the eyes for CNN input
        left_eye_image = process_eye_for_cnn(leftEye, frame)
        right_eye_image = process_eye_for_cnn(rightEye, frame)
       
        # Predict using the CNN model
        left_pred = model.predict(left_eye_image)
        right_pred = model.predict(right_eye_image)
       
        # Combine the predictions for both eyes
        avg_pred = (left_pred + right_pred) / 2.0
       
        # Check if both eyes are closed (avg_pred > 0.5 indicates closed eyes)
        if avg_pred > 0.5:
            closed_eye_duration += 1
        else:
            closed_eye_duration = 0  # Reset the duration if eyes are open
 
        # If closed eyes duration exceeds a threshold, trigger drowsiness alert
        if closed_eye_duration >= DROWSINESS_FRAME_THRESHOLD:
            if not ALARM_ON:
                ALARM_ON = True
                os.system('say "Wake up"')  # MacOS alert; replace with appropriate for your OS
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            ALARM_ON = False
 
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
   
    if key == ord("q"):
        break
 
vs.release()
cv2.destroyAllWindows()