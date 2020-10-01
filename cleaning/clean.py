import argparse 
import os 
import os.path as path 

from tools.utils import UFile as UF 

if __name__ == '__main__':
    print(' ... [clean image name] ... ')

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', help='path to target directory for cleaning process', required=True)
    args_map = vars(parser.parse_args())

    current_location = path.dirname( path.realpath( __file__) )
    target_directory = path.join( current_location, '..', args_map['target'] )

    UF.rename_files_in_directory(target_directory, '*', 'IMG')