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

import cv2

import dlib


def save_crop(img):
    for j in range(0, len(img)):
        path_dir = os.path.dirname(os.path.abspath(__file__))
        directory = path_dir + "/face_{}".format(j+1)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(os.path.join(directory, 'face_{}.jpg'.format(j+1)), img[j])

def del_crop(img):
    for j in range(0, len(img)):
        path_dir = os.path.dirname(os.path.abspath(__file__))
        directory = path_dir + "/face_{}".format(j+1)
        shutil.rmtree(directory)

def show_crop(img, j, AUX):
    assert img[j] is not None
    cv2.imshow("Crop {}".format(j+1), img[j])
    cv2.moveWindow("Crop {}".format(j+1), 580, j*200)
    if (AUX-1) > 0:
        show_crop(img, j+1, AUX-1)

def face_output(coord, i_1, i_2):
    x_1, y_1 = coord[i_1]
    x_2, y_2 = coord[i_2]
    DISTANCE = math.sqrt(((x_2-x_1)**2)+((y_2-y_1)**2))
    return DISTANCE

AP = argparse.ArgumentParser()
AP.add_argument("-p", "--shape-predictor", required=True,
                help="Caminho para Shape Predictor")
ARGS = vars(AP.parse_args()) # shape_predictor_68_face_landmarks.dat

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(ARGS["shape_predictor"])
FA = FaceAligner(PREDICTOR, desiredFaceWidth=120)

print("[TIME SLEEP]: 3s..")
time.sleep(1.0)
print("[TIME SLEEP]: 2s..")
time.sleep(1.0)
print("[TIME SLEEP]: 1s..")
time.sleep(1.0)

CAP = cv2.VideoCapture(0) # Caputa video
CAP.set(5, 30)

while True: # Reproduz video ate que uma tecla definida seja pressionada
    RET, IMAGE = CAP.read()
    IMAGE = cv2.flip(IMAGE, 1, 0)
    RESIZE = imutils.resize(IMAGE, width=480, height=360)
    GRAY = cv2.cvtColor(RESIZE, cv2.COLOR_BGR2GRAY)

    DETECT_RET = DETECTOR(GRAY, 1)
    FACE_LIST = []
    FACE_LANDMARKS_LIST = []
    FACE_LANDMARKS_FEATURES = []

    i = 0

    for RECT in DETECT_RET:               
        (x, y, w, h) = rect_to_bb(RECT)
        FACE_LIST.append(i)
        FACE_LIST[i] = FA.align(RESIZE, GRAY, RECT)
        
        """
        SHAPE = PREDICTOR(FACE_LIST[i], RECT)
        SHAPE = face_utils.shape_to_np(SHAPE)

        for (name, (m, n)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            FACE_LANDMARKS_LIST.append(i)
            for (xf, yf) in SHAPE[m:n]:
                FACE_LANDMARKS_FEATURES.append((xf, yf))
        """
        
        for k, d in enumerate(DETECT_RET):  
            SHAPE = PREDICTOR(FACE_LIST[i], d)
        for j in range(0,67):
            FACE_LANDMARKS_FEATURES.append((SHAPE.part(j).x, SHAPE.part(j).y))
        """
        print("Nose bottom {}".format(FACE_LANDMARKS_FEATURES[33]))
        print("Left Eye {}".format(FACE_LANDMARKS_FEATURES[39]))
        print("Rigth Eye {}".format(FACE_LANDMARKS_FEATURES[42]))
        print("Mouth {}".format(FACE_LANDMARKS_FEATURES[57]))
        print("Mid Eyes (J1): {}".format(face_output(FACE_LANDMARKS_FEATURES, 39, 42)))
        print("Left Eye to Mouth (J2): {}".format(face_output(FACE_LANDMARKS_FEATURES, 36, 57)))
        print("Right Eye to Mouth (J3): {}".format(face_output(FACE_LANDMARKS_FEATURES, 45, 57)))
        print("Letf Eye to Nose (J4): {}".format(face_output(FACE_LANDMARKS_FEATURES, 39, 33)))
        print("Right Eye to Nose (J5): {}".format(face_output(FACE_LANDMARKS_FEATURES, 42, 33)))
        print("Top Nose to Mouth (J6): {}".format(face_output(FACE_LANDMARKS_FEATURES, 27, 57)))
        print("Top Nose to Nose (J7): {}".format(face_output(FACE_LANDMARKS_FEATURES, 27, 33)))
        print("Nose Width (J8): {}".format(face_output(FACE_LANDMARKS_FEATURES, 31, 35)))
        """

        if len(DETECT_RET) > 0:
            TEXT = "{} rosto(s) encontrado(s)".format(len(DETECT_RET))
            cv2.putText(RESIZE, TEXT, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.rectangle(RESIZE, (x, (y + h)), (x + (x + w) - x, (y + h) + y - (y + h)), (0, 255, 0), 1)

        TEXT = "Rosto {}".format(i+1)
        cv2.putText(RESIZE, TEXT, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        i = i + 1 
    cv2.imshow("Video Output Original", RESIZE)
    cv2.moveWindow("Video Output", 0, 0)
    
    if len(FACE_LIST) > 0:
        show_crop(FACE_LIST, 0, len(FACE_LIST))
        save_crop(FACE_LIST)

    k = cv2.waitKey(10) & 0xFF
    if k == ord('c'):
        break

cv2.destroyAllWindows()

time.sleep(1.0)

if len(FACE_LIST) > 0:
    DELET = ''
    while True:
        DELET = input("Manter pasta(s) criada(s)? [s/n] ")
        if DELET == 's' or DELET == 'n':
            break

    if DELET == 'n':
        del_crop(FACE_LIST)
        print("Pasta(s) apagada(s)")
    else:
        print("Pasta(s) conservada(s)")

CAP.release()
