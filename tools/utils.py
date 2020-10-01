import numpy as np 
import itertools as it 
import functools as ft 

import cv2

import os 
import os.path as path 

import json 
import glob 

"""
    Ce fichier va servir de definition d'un ensemble de fonction
    De type utils : 
"""

class UModel:
    @staticmethod
    def serialize_model(path_to_storage):
        return None 

    @staticmethod
    def deserialize_model(path_to_model):
        return None 
    
class UFile:
    @staticmethod
    def rename_files_in_directory(directory_path, file_extension, prefix=''):
        directory_contents = glob.glob(path.join(directory_path, file_extension))
        index = 0 
        nb_items = len(directory_contents)
        for file_path in directory_contents:
            path_body, path_end = path.split(file_path)
            file_name, file_extension = path_end.split('.')
            new_file_name = '%s_%04d.%s' % (prefix, index, file_extension)
            new_file_path = path.join(path_body, new_file_name)
            os.rename(file_path, new_file_path) 
            index = index + 1

    @staticmethod
    def list_files(directory_path):
        return os.listdir(directory_path)
    
    def serialize_features(features_accumulator, target_file):
        json.dump(features_accumulator, open(target_file, 'w'))
    
    def deserialize_features(target_file_path):
        if path.exists(target_file_path):
            return json.load(open(target_file_path, 'r'))
        else:
            print('[warning]: this path is not valid')
            return None 

class UImage:
    @staticmethod
    def create_window(window_name, width, height):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    @staticmethod
    def display_image(window_name, image):
        cv2.imshow(window_name, image)

    @staticmethod
    def read_image(image_path):
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

    @staticmethod
    def to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def pause():
        cv2.waitKey(0)

    @staticmethod
    def resize_image(image, width, height):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def get_face_detector():
        path_to_cascade = 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + path_to_cascade
        ) 

    @staticmethod
    def get_hog_descriptor(width, height):
        return cv2.HOGDescriptor( (width, height), (8, 8), (4, 4), (8, 8), 9 )

    @staticmethod
    def get_hog_features(image, hog_desccriptor):
        hog_features = hog_desccriptor.compute(image)
        return hog_features

    @staticmethod
    def get_gabor_kernel():
        kernel_accumulator = []
        for sigma in np.linspace(4, 9, 5) * np.sqrt(2):
            for theta in np.linspace(0, np.pi / 2, 8):
                kernel_accumulator.append(
                    cv2.getGaborKernel(sigma, theha, 1.5, 1.5, np.pi / 2, cv2.CV_32F)
                )
        return kernel_accumulator

    @staticmethod
    def find_faces_roi(face_detector, image, wh_threshold=(24, 24)):
        min_w, min_h = wh_threshold
        min_valid_area = min_w * min_h

        boxe_coordinates = face_detector.detectMultiScale(image, 1.3, 5)
        if len(boxe_coordinates) > 0: 
            return [[x, y, w, h] for x, y, w, h in boxe_coordinates if w * h >= min_valid_area]
        return None 

    @staticmethod
    def draw_faces_on_image(image, face_roi_coordinates, boxe_color=(255, 0, 0)):
        for [x, y, w, h] in face_roi_coordinates: 
            cv2.rectangle(image, (x, y), (x + w, y + h), boxe_color, 2)
        return None 

    @staticmethod
    def extract_roi_from_image(image, roi_coordinates):
        [x, y, w, h] = roi_coordinates  # unpack the cooridnates 
        roi = image[y:y+h, x:x+w].copy()
        return roi 

    @staticmethod
    def split_image_to_cells(input_gray_image, vertical_split=8, horizontal_split=8):
        # this function expects an image of shape : (64, 64) 
        # this function is used in order to split the image into several cells 
        # so by default we gonna use the next configuration : (8, 8)
        vertical_split_outputs = np.vplsit(input_gray_image, vertical_split)
        cells = list(
            it.chain(
                *[ np.hsplit(sub_array, horizontal_split) for sub_array in vertical_split_outputs ]
            )
        )  # concatenate all cells_array 
        return cells 

    @staticmethod
    def dct_per_block(input_gray_image):
        cells = UImage.split_image_to_cells(input_gray_image.astype('float32'))+
        dct_features_accumulator = [] 
        for cell in cells: 
            dct_output = cv2.dct(cell)
            dct_diagonal_components = np.diag(dct_output)
            dct_features_accumulator.append(dct_diagonal_components)
        return np.hstack(dct_features_accumulator)  # a 512 dimensionals features vector 
