import numpy as np 
import itertools as it 
import functools as ft 

from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.neural_network import MLPClassifier as MLP 
from sklearn.svm import SVC

import os 
import os.path as path 

import pickle 
import json 

import joblib 

from tools.utils import UFile as UF 
from tools.utils import UModel as UM 

if __name__ == '__main__':
    print(' ... [Modelization] | build and store model ... ')
    current_location = path.dirname( path.realpath(__file__) )
    features_location = path.join( current_location, '..', 'features_dump/Features' )
    
    serialization_location = path.join( current_location, '..', 'serialization/Models')

    dct_knn_predictor = KNN(n_neighbors=7)
    dct_mlp_predictor = MLP((128, 128), tol=1e-8, verbose=True, max_iter=10000)
    dct_svm_predictor = SVC(tol=1e-8, verbose=True, max_iter=10000)

    hog_knn_predictor = KNN(n_neighbors=7)
    hog_mlp_predictor = MLP((128, 128), tol=1e-8, verbose=True, max_iter=10000)
    hog_svm_predictor = SVC(tol=1e-8, verbose=True, max_iter=10000)

    features = UF.deserialize_features( path.join( features_location, 'features.pkl' ), 'rb', pickle )
    print('Type of features : ', list(features.keys()))

    dct_features = features['dct']
    hog_features = features['hog']

    dct_source_data, dct_target_data = UM.split_source_target(dct_features, source_pos=1, target_pos=0)
    hog_source_data, hog_target_data = UM.split_source_target(hog_features, source_pos=1, target_pos=0)

    dct_source_data = np.vstack(dct_source_data)
    dct_target_data = list(map(int, dct_target_data))

    hog_source_data = np.vstack(hog_source_data)
    hog_target_data = list(map(int, hog_target_data))



    dct_knn_predictor.fit(dct_source_data, dct_target_data)
    dct_mlp_predictor.fit(dct_source_data, dct_target_data)
    dct_svm_predictor.fit(dct_source_data, dct_target_data)
    

    hog_knn_predictor.fit(hog_source_data, hog_target_data)
    hog_mlp_predictor.fit(hog_source_data, hog_target_data)
    hog_svm_predictor.fit(hog_source_data, hog_target_data)


    UM.serialize_model(path.join(serialization_location, 'dct_knn.joblib'), dct_knn_predictor, joblib)
    UM.serialize_model(path.join(serialization_location, 'dct_mlp.joblib'), dct_mlp_predictor, joblib)
    UM.serialize_model(path.join(serialization_location, 'dct_svm.joblib'), dct_svm_predictor, joblib)
    
    UM.serialize_model(path.join(serialization_location, 'hog_knn.joblib'), hog_knn_predictor, joblib)
    UM.serialize_model(path.join(serialization_location, 'hog_mlp.joblib'), hog_mlp_predictor, joblib)
    UM.serialize_model(path.join(serialization_location, 'hog_svm.joblib'), hog_svm_predictor, joblib)

    
    