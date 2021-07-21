Question 1
Steps:
Use processing_data.py to preprocess data
Use tight_frame_feature.py to extract features
Use train_test_devide.py to divide train and test set
Use Feature_model to select features
Use Model.py to build model and do classification
------
processing_data.py:       Preprocess data: transform RGB images to gray images and so on.
train_test_devide.py:      Divide training/test set
tight_frame_feature.py:  Use tight frame method to extract features
gabor_feature.py:           Use Wavelet method to extract features
Feature_model.py:         Apply cross-one-validation on forward stage wise algorithm to select features
feature_selection.py:      Use stage-wise forward method to assist Feature_model.py to select features
Model.py:                       Training and test model

