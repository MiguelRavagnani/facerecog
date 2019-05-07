import pickle
import os
import random
import math
import numpy as np
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition

from tqdm import tqdm

import cv2

import dlib

PATH_TO_SCRIPT = os.path.dirname(os.path.realpath(__file__))

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PATH_TO_SCRIPT + "/shape_predictor_68_face_landmarks.dat")
FA = FaceAligner(PREDICTOR, desiredFaceWidth=480)

PATH = PATH_TO_SCRIPT + "/data/mydataset/raw/"

CATEGORIES = []

for DIRECTORY in os.listdir(PATH):
    CATEGORIES.append(DIRECTORY)

training_data = []


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

"""
def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(PATH,category)  
        class_num = CATEGORIES.index(category)
        
        print(path)

        for img in tqdm(os.listdir(path)):
            try:
                IM_INDEX = path + ("/{}".format(img))
                IMPORT = cv2.imread(IM_INDEX, 1)
                IMAGE = cv2.flip(IMPORT, 1, 0)
                #IMAGE = imutils.resize(IMPORT, width=480, height=360)
                IMAGE = cv2.flip(IMAGE, 1, 0)
                GRAY = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)

                print(GRAY)

                DETECT_RET = DETECTOR(GRAY, 1)
                FACE_LIST = []
                FACE_LANDMARKS_LIST = []
                FACE_LANDMARKS_FEATURES = []

                i = 0

                for RECT in DETECT_RET:               
                    (x, y, w, h) = rect_to_bb(RECT)
                    FACE_ALIGNED = FA.align(IMAGE, GRAY, RECT)
                    
                    for k, d in enumerate(DETECT_RET):  
                        SHAPE = PREDICTOR(FACE_ALIGNED, d)
                    for j in range(68):
                        FACE_LANDMARKS_FEATURES.append((SHAPE.part(j).x, SHAPE.part(j).y))
                    
                    J_ARRAY = []
                    
                    REF = 2*face_output(FACE_LANDMARKS_FEATURES, 45, 36)
                    
                    #J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 39, 42))
                    #J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 36, 57))
                    #J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 45, 57))
                    #J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 39, 57))
                    #J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 42, 57))
                    #J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 27, 57))
                    #J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 27, 33))
                    #J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 31, 35))
                    

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
                
                training_data.append([J_NORM, class_num])
            except Exception as e:
                pass
"""

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(PATH,category)  
        class_num = CATEGORIES.index(category)
        
        print(path)

        for img in tqdm(os.listdir(path)):
            try:
                IM_INDEX = path + ("/{}".format(img))
                FILE_ENCODING = [] # Create an empty list for saving encoded files
                image = path + ("/{}".format(img))
                image = face_recognition.load_image_file(image) # Run your load command
                image_encoding = face_recognition.face_encodings(image) # Run your encoding command
                FILE_ENCODING.append(image_encoding[0]) # Append the results to encoding_for_file list
                FILE_ENCODING = np.array(FILE_ENCODING)
                FILE_ENCODING.resize((len(FILE_ENCODING), 128)) # Resize using your command
                training_data.append([image_encoding[0], class_num])
            except Exception as e:
                pass

create_training_data()

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
#print(X)

X_fix = np.squeeze(X)
Y_fix = np.squeeze(y)

X_train, X_test, y_train, y_test = train_test_split(X_fix,Y_fix,test_size=0.2)

pickle_out = open("X_train.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("Y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("Y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

print(X_test)
print(y_test)