import os
import cv2
import dlib
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the shape predictor for eye landmarks
predictor = dlib.shape_predictor('/Users/nolanpettit/documents/data mining/cs350_drowsiness_detection-main/shape_predictor_68_face_landmarks.dat')

# Directories for open and closed eye images
open_eye_dir = '/Users/nolanpettit/documents/data mining/cs350_drowsiness_detection-main/Open_Eyes'
closed_eye_dir = '/Users/nolanpettit/documents/data mining/cs350_drowsiness_detection-main/Closed_Eyes'

# Initialize lists to hold EAR values and labels
data = []
labels = []

# Function to process eye images directly
def process_eye_images(image_dir, label):
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            print(f"Processing {filename}...")
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            
            # Resize the image for consistency
            image = cv2.resize(image, (640, 480))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Instead of detecting faces, directly use the predictor on the resized image
            # Create a fake rectangle for the face (assuming the eye is centered in the image)
            face_rect = dlib.rectangle(0, 0, image.shape[1], image.shape[0])

            # Find landmarks on the detected face
            shape = predictor(gray, face_rect)

            # Convert to numpy array
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])

            # Extract left and right eye coordinates (using specific landmark indices)
            left_eye = shape_np[36:42]  # Points 36-41 for the left eye
            right_eye = shape_np[42:48]  # Points 42-47 for the right eye

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            
            data.append(avg_ear)
            labels.append(label)

# Process open eye images
process_eye_images(open_eye_dir, 0)  # Label 0 for open eyes

# Process closed eye images
process_eye_images(closed_eye_dir, 1)  # Label 1 for closed eyes

# Create a DataFrame from the data and labels
df = pd.DataFrame({'EAR': data, 'Label': labels})

# Check if there's enough data to proceed
if df.empty:
    print("No data to train on. Please check the image directories.")
else:
    print(f"Total samples processed: {len(df)}")
    
    # Split the data into features and target labels
    X = df[['EAR']].values
    y = df['Label'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model (e.g., Random Forest)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'drowsiness_model.joblib')

    print("Model trained and saved as drowsiness_model.joblib")
