import numpy as np 
import itertools as it 
import functools as ft 

import os 
import os.path as path 

import json 
import glob 

import pickle 
import joblib 

import operator as op 

from tools.utils import UImage as UI
from tools.utils import UModel as UM

if __name__ == '__main__':
    print(' ... [prediction] ... ')

    main_window = 'main screen'
    UI.create_window(main_window, 640, 480)

    current_location = path.dirname( path.realpath(__file__) )
    validation_location = path.join( current_location, '..', 'validation-data' )
    serialization_location = path.join( current_location, '..', 'serialization/Models')

    dct_model_names = glob.glob( path.join( serialization_location, 'dct_*.joblib') )
    hog_model_names = glob.glob( path.join( serialization_location, 'hog_*.joblib') )

    dct_models = [ UM.deserialize_model(dct_model_path, joblib) for dct_model_path in dct_model_names ]
    hog_models = [ UM.deserialize_model(hog_model_path, joblib) for hog_model_path in hog_model_names ]

    image_paths = glob.glob( path.join( validation_location, '*') )
    face_detector = UI.get_face_detector()
    hog_descriptor = UI.get_hog_descriptor(64, 64)

    for image_path in image_paths:
        current_image = UI.read_image(image_path)
        current_gray_image = UI.to_gray(current_image)
        roi_coordinates = UI.find_faces_roi(face_detector, current_gray_image)
        
        UI.draw_faces_on_image(current_image, roi_coordinates)

        for coordinates in roi_coordinates:
            face_roi = UI.extract_roi_from_image(current_gray_image, coordinates)
            resized_face_roi = UI.resize_image(face_roi, 64, 64)

            current_hog_features = UI.get_hog_features(resized_face_roi, hog_descriptor)
            current_dct_features = UI.dct_per_block(resized_face_roi)
            predictions = []

            for HOG_M, DCT_M in zip(hog_models, dct_models):
                hog_output = HOG_M.predict(current_hog_features[None, :])
                dct_output = DCT_M.predict(current_dct_features[None, :])
                predictions.append(hog_output)
                predictions.append(dct_output)
            
            print(predictions)
            
        UI.display_image(main_window, current_image)
        UI.pause()
    UI.free()