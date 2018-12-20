import argparse
import os
import shutil
import time

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

AP = argparse.ArgumentParser()
AP.add_argument("-p", "--shape-predictor", required=True,
                help="Caminho para Shape Predictor")
ARGS = vars(AP.parse_args()) # shape_predictor_68_face_landmarks.dat

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(ARGS["shape_predictor"])

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
    RESIZE = cv2.resize(IMAGE, (240, 180))
    DISPLAY = cv2.resize(IMAGE, (480, 360))
    GRAY = cv2.cvtColor(RESIZE, cv2.COLOR_BGR2GRAY)

    DETECT_RET = DETECTOR(GRAY, 1)

    if len(DETECT_RET) > 0:
        TEXT = "{} rosto(s) encontrado(s)".format(len(DETECT_RET))
        cv2.putText(DISPLAY, TEXT, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            	0.7, (0, 0, 255), 2)

    CROP = []
    for i, dim in enumerate(DETECT_RET):
        print("Rosto {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, dim.left(), dim.top(), dim.right(), dim.bottom()))
        CROP.append(i)
        CROP[i] = RESIZE[max(0, dim.top()): min(dim.bottom(), 480),
                         max(0, dim.left()): min(dim.right(), 360)]

        cv2.rectangle(DISPLAY, (dim.left()*2, dim.top()*2), (dim.left()*2 + dim.right()*2 - dim.left()*2, dim.top()*2 + dim.bottom()*2 - dim.top()*2), (0, 255, 0), 1)

        TEXT = "Rosto {}".format(i+1)
        cv2.putText(DISPLAY, TEXT, (dim.left()*2, dim.top()*2-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Video Output - 480x360", DISPLAY)
    cv2.moveWindow("Video Output - 480x360", 0, 0)

    if len(CROP) > 0:
        show_crop(CROP, 0, len(CROP))
        save_crop(CROP)

    k = cv2.waitKey(10) & 0xFF
    if k == ord('c'):
        break

cv2.destroyAllWindows()

time.sleep(1.0)

if len(CROP) > 0:
    DELET = ''
    while True:
        DELET = input("Manter pasta(s) criada(s)? [s/n] ")
        if DELET == 's' or DELET == 'n':
            break

    if DELET == 'n':
        del_crop(CROP)
        print("Pasta(s) apagada(s)")
    else:
        print("Pasta(s) conservada(s)")

CAP.release()
