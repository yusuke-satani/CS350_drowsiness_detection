1. camera_test.py: see which index of cameras works
3. train_shape_predictor.py: this is a Python file used to fine_tune the eye key points detector
4. eye_predictor.dat : the fine_tuned which detect only eyes key points trained on train_shape_predictior.py.
5. FILEFORTRAININGCNNMODEL : ADD EXPLANATION
6. drowsiness_cnn_model.h5 : 
7. predict_eyes.py : this is a main file that can detect the eye key points and calculate the EAR score.
                      
   (to run: python predict_eyes.py --shape-predictor eye_predictor.dat)
