import argparse 
import os 
import os.path as path 

from tools.utils import UFile as UF 

import cv2 
import glob

if __name__ == '__main__':
    print(' ... [clean image name] ... ')

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', help='path to target directory for cleaning process', required=True)
    parser.add_argument('-m', '--mode', help='mode of script', default='rename')
    
    args_map = vars(parser.parse_args())

    mode = args_map['mode']
    
    current_location = path.dirname( path.realpath( __file__) )
    target_directory = path.join( current_location, '..', args_map['target'] )
    print(f'Lunch {mode} script in the directory : {target_directory}')

    if mode == 'rename':    
        UF.rename_files_in_directory(target_directory, '*', 'IMG')
    if mode == 'resize':
        image_paths = glob.glob( path.join( target_directory, '*') )
        for image_path in image_paths:
            current_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            resize_image = cv2.resize(current_image, (640, 480), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(image_path, resize_image)
