from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def draw_eye(frame, eye_points, color, is_left):
    if len(eye_points) == 6:
        for i, point in enumerate(eye_points):
            cv2.circle(frame, tuple(point), 2, color, -1)
            # Display coordinates
            cv2.putText(frame, f"({point[0]},{point[1]})", 
                        (point[0]+5, point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw the eye contour
        cv2.polylines(frame, [eye_points], True, color, 1)
        
        # Calculate and draw the center
        center = np.mean(eye_points, axis=0).astype("int")
        cv2.circle(frame, tuple(center), 3, (0, 255, 0), -1)
        
        # Calculate EAR
        ear = eye_aspect_ratio(eye_points)
        
        # Label the eye with EAR
        eye_label = f"{'Left' if is_left else 'Right'} Eye EAR: {ear:.2f}"
        cv2.putText(frame, eye_label, (center[0] - 20, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return center, ear
    return None, None

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = cv2.VideoCapture(1)
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the video stream, resize it to have a
    # maximum width of 900 pixels, and convert it to grayscale
    ret, frame = vs.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame
    try:
        rects = detector(gray, 0)
    except Exception as e:
        print(f"Error in face detection: {e}")
        continue
    
    # loop over the face detections
    for rect in rects:
        # convert the dlib rectangle into an OpenCV bounding box and
        # draw a bounding box surrounding the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # use our custom dlib shape predictor to predict the location
        # of our landmark coordinates, then convert the prediction to
        # an easily parsable NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Assuming the first 6 points are for the left eye and the next 6 for the right eye
        leftEye = shape[:6]
        rightEye = shape[6:12]
        
        # Draw left eye and calculate EAR
        _, left_ear = draw_eye(frame, leftEye, (0, 0, 255), True)  # Red for left eye
        
        # Draw right eye and calculate EAR
        _, right_ear = draw_eye(frame, rightEye, (255, 0, 0), False)  # Blue for right eye
        
        # Calculate average EAR
        if left_ear and right_ear:
            avg_ear = (left_ear + right_ear) / 2.0
            cv2.putText(frame, f"Avg EAR: {avg_ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
""" this code works
# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = cv2.VideoCapture(1)
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the video stream, resize it to have a
    # maximum width of 400 pixels, and convert it to grayscale
    ret, frame = vs.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame
    try:
        rects = detector(gray, 0)
    except Exception as e:
        print(f"Error in face detection: {e}")
        continue
    
    # loop over the face detections
    for rect in rects:
        # convert the dlib rectangle into an OpenCV bounding box and
        # draw a bounding box surrounding the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # use our custom dlib shape predictor to predict the location
        # of our landmark coordinates, then convert the prediction to
        # an easily parsable NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # loop over the (x, y)-coordinates from our dlib shape
        # predictor model draw them on the image
        for (sX, sY) in shape:
            cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()"""