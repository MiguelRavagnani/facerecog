import argparse
import os
import shutil
import time
import numpy as np
import imutils
import math
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition

from tqdm import tqdm

import cv2

import dlib

"""
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
"""

def face_output(coord, i_1, i_2):
    x_1, y_1 = coord[i_1]
    x_2, y_2 = coord[i_2]
    DISTANCE = math.sqrt(((x_2-x_1)**2)+((y_2-y_1)**2))
    return DISTANCE

def mid_point(coord, i_1, i_2):
    x_1, y_1 = coord[i_1]
    x_2, y_2 = coord[i_2]
    MID = [(x_1 + x_2)/2, (y_1 + y_2)/2]
    return MID

def distance_points(i_1, i_2):
    x_1, y_1 = i_1[0], i_1[1]
    x_2, y_2 = i_2[0], i_2[1]
    DISTANCE = math.sqrt(((x_2-x_1)**2)+((y_2-y_1)**2))
    return DISTANCE   


PATH_TO_SCRIPT = os.path.dirname(os.path.realpath(__file__))
PATH = PATH_TO_SCRIPT + "/data/mydataset/raw/Miguel_Ravagnani/2.jpg"
PATH_2 = PATH_TO_SCRIPT + "/data/mydataset/raw/Miguel_Ravagnani/"

#print(PATH)

AP = argparse.ArgumentParser()
AP.add_argument("-p", "--shape-predictor", required=True,
                help="Caminho para Shape Predictor")
ARGS = vars(AP.parse_args()) # shape_predictor_68_face_landmarks.dat

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(ARGS["shape_predictor"])
FA = FaceAligner(PREDICTOR, desiredFaceWidth=480)


IMPORT = cv2.imread(PATH, 1)
#IMAGE = imutils.resize(IMPORT, width=480, height=360)
IMAGE = IMPORT
GRAY = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    
DETECT_RET = DETECTOR(GRAY, 1)
FACE_LIST = []
FACE_LANDMARKS_LIST = []
FACE_LANDMARKS_FEATURES = np.empty([68, 2], dtype = int)

i = 0

for RECT in DETECT_RET:               
    FACE_ALIGNED = FA.align(IMAGE, GRAY, RECT)
    
    for k, d in enumerate(DETECT_RET):  
        SHAPE = PREDICTOR(FACE_ALIGNED, d)
    for j in range(1, 68):
        FACE_LANDMARKS_FEATURES[j][0] = SHAPE.part(j).x
        FACE_LANDMARKS_FEATURES[j][1] = SHAPE.part(j).y
    
    J_ARRAY = []
    REF = 2*face_output(FACE_LANDMARKS_FEATURES, 45, 36)
    """
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 39, 42))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 36, 57))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 45, 57))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 39, 57))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 42, 57))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 27, 57))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 27, 33))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 31, 35))
    """ 
    PLACE_COORD_1 = mid_point(FACE_LANDMARKS_FEATURES, 36, 39)
    PLACE_COORD_2 = mid_point(FACE_LANDMARKS_FEATURES, 42, 45)
    J_ARRAY.append(distance_points(PLACE_COORD_1, PLACE_COORD_2))
    J_ARRAY.append(distance_points(PLACE_COORD_1, FACE_LANDMARKS_FEATURES[62]))
    J_ARRAY.append(distance_points(PLACE_COORD_2, FACE_LANDMARKS_FEATURES[62]))
    J_ARRAY.append(distance_points(PLACE_COORD_1, FACE_LANDMARKS_FEATURES[33]))
    J_ARRAY.append(distance_points(PLACE_COORD_2, FACE_LANDMARKS_FEATURES[33]))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 33, 62))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 27, 33))
    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 31, 35))

    J_NP = np.array(J_ARRAY)#.reshape((len(J_ARRAY), 1))
    J_NP = 2*(np.array(J_ARRAY)/REF) - 1
    J_NORM = np.array(J_NP).reshape((len(J_NP), 1))
    #J_MIN_MAX = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #J_NORM = J_MIN_MAX.fit_transform(J_NP)
    #J_NORM = np.tanh(J_NP/1000) 


    FILE_ENCODING = [] # Create an empty list for saving encoded files
    path = os.path.join(PATH_2)  
    #class_num = CATEGORIES.index(category)
    for img in tqdm(os.listdir(PATH_2)): # Loop over the folder to list individual files
        image = path + ("/{}".format(img))
        
        
        image = face_recognition.load_image_file(image) # Run your load command
        image_encoding = face_recognition.face_encodings(image) # Run your encoding command
        FILE_ENCODING.append(image_encoding[0]) # Append the results to encoding_for_file list
    FILE_ENCODING = np.array(FILE_ENCODING)
    FILE_ENCODING.resize((len(FILE_ENCODING), 128)) # Resize using your command

cv2.imshow('image', FACE_ALIGNED)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(FILE_ENCODING)
#print("Normalized: J1 {} J2 {} J3 {} J4 {} J5 {} J6 {} J7 {} J8 {}".format(J_NORM[0], J_NORM[1], J_NORM[2], J_NORM[3], J_NORM[4], J_NORM[5], J_NORM[6], J_NORM[7]))
