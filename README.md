1. camera_test.py: see which index of cameras works
2. eye_predictor.dat : the custom_model which detect only eyes keypoints
3. train_shape_predictor.py : this is a python file used to train the eye  keypoints detector
4. predict_eyes.py : this is a main file which can detect the eye keypoints and calcurate EAR score.
   (to run: python predict_eyes.py --shape-predictor eye_predictor.dat)
