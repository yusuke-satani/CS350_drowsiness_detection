camera_test.py: see which index of cameras works
eye_predictor.dat : the custom_model which detect only eyes keypoints
train_shape_predictor.py : this is a python file used to train the eye  keypoints detector
predict_eyes.py : this is a main file which can detect the eye keypoints and calcurate EAR score.
(to run: python predict_eyes.py --shape-predictor eye_predictor.dat)
