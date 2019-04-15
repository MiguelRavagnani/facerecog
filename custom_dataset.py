import pickle
import os
import random
import math
import numpy as np
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from sklearn import preprocessing

from tqdm import tqdm

import cv2

import dlib

PATH_TO_SCRIPT = os.path.dirname(os.path.realpath(__file__))

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PATH_TO_SCRIPT + "/shape_predictor_68_face_landmarks.dat")
FA = FaceAligner(PREDICTOR, desiredFaceWidth=120)

PATH = PATH_TO_SCRIPT + "/data/mydataset/raw/"
CATEGORIES = ["person-1", "person-2"]

training_data = []


def face_output(coord, i_1, i_2):
    x_1, y_1 = coord[i_1]
    x_2, y_2 = coord[i_2]
    DISTANCE = math.sqrt(((x_2-x_1)**2)+((y_2-y_1)**2))
    return DISTANCE


def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(PATH,category)  
        class_num = CATEGORIES.index(category)
        
        print(path)

        for img in tqdm(os.listdir(path)):
            try:
                IM_INDEX = path + ("/{}".format(img))
                IMPORT = cv2.imread(IM_INDEX, 1)
                IMPORT = cv2.flip(IMPORT, 1, 0)
                IMAGE = imutils.resize(IMPORT, width=480, height=360)
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
                    
                    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 39, 42))
                    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 36, 57))
                    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 45, 57))
                    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 39, 33))
                    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 42, 33))
                    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 27, 57))
                    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 27, 33))
                    J_ARRAY.append(face_output(FACE_LANDMARKS_FEATURES, 31, 35))
                        
                    J_NP = np.array(J_ARRAY)
                    J_NP = np.array(J_NP).reshape((len(J_NP), 1))    
                    J_MIN_MAX = preprocessing.MinMaxScaler()
                    J_NORM = J_MIN_MAX.fit_transform(J_NP)

                training_data.append([J_NORM, class_num])
            except Exception as e:
                pass
                               
create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)
#print(X)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()