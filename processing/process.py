import itertools as it 
import functools as ft 

import numpy as np 
import operator as op 

import cv2 

import os 
import os.path as path 
import glob 


import json, pickle

from tools.utils import UFile as UF
from tools.utils import UImage as UI
 

if __name__ == '__main__':
    print(' ... [processing] face image | features extraction ... ')
    current_location = path.dirname( path.realpath(__file__) )
    location_to_faces_data = path.join( current_location, '..', 'source-data/Faces')
    location_to_features_dump = path.join( current_location, '..', 'features_dump/Features')

    face_data_paths = glob.glob( path.join( location_to_faces_data, '*.jpg') )
    
    hog_descriptor = UI.get_hog_descriptor(64, 64)
    
    hog_features_accumulator = []
    dct_features_accumulator = []

    index = 0
    for face_path in face_data_paths:
        path_body, path_head = path.split(face_path)
        filename, extension = path_head.split('.')
        current_gender = filename[0]  # get gender part | look the description file in source-data

        face_image = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
        resize_face_image = cv2.resize(face_image, (64, 64))

        current_hog_features = UI.get_hog_features(resize_face_image, hog_descriptor)
        current_dct_features = UI.dct_per_block(resize_face_image)

        hog_features_accumulator.append( (current_gender, current_hog_features) )
        dct_features_accumulator.append( (current_gender, current_dct_features) )

        print('Image number %03d was treated with success' % index)
        index = index + 1 

    features = {
        'hog': hog_features_accumulator, 
        'dct': dct_features_accumulator
    }

    UF.serialize_features(features, path.join(location_to_features_dump, 'features.pkl'), 'wb' ,pickle)
