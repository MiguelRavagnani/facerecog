import argparse
import os
import shutil
import time

import cv2

import dlib


def saveCrop(img):
    for j in range(0, len(img)):
        pathDir = os.path.dirname(os.path.abspath(__file__))
        directory = pathDir + "/face_{}".format(j+1)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(os.path.join(directory , 'face_{}.jpg'.format(j+1)), img[j])

def delCrop(img):
    for j in range(0, len(img)):
        pathDir = os.path.dirname(os.path.abspath(__file__))
        directory = pathDir + "/face_{}".format(j+1)
        shutil.rmtree(directory)

def showCrop(img, j, aux):
    assert img[j] is not None
    cv2.imshow("Crop {}".format(j+1), img[j])
    cv2.moveWindow("Crop {}".format(j+1), 580, j*200)
    if (aux-1) > 0:
        showCrop(img, j+1, aux-1)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="Caminho para Shape Predictor")
args = vars(ap.parse_args()) # shape_predictor_68_face_landmarks.dat

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[TIME SLEEP]: 3s..")
time.sleep(1.0)
print("[TIME SLEEP]: 2s..")
time.sleep(1.0)
print("[TIME SLEEP]: 1s..")
time.sleep(1.0)

cap = cv2.VideoCapture(0) # Caputa video
cap.set(5,30)

while True: # Reproduz video ate que uma tecla definida seja pressionada
    ret, image = cap.read()
    image = cv2.flip(image, 1, 0)
    resize = cv2.resize(image, (240, 180))
    display = cv2.resize(image, (480, 360))
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    detectRect = detector(gray, 1)

    if len(detectRect) > 0:
        text = "{} rosto(s) encontrado(s)".format(len(detectRect))
        cv2.putText(display, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        	0.7, (0, 0, 255), 2)

    crop = []
    for i, dim in enumerate(detectRect):
        print("Rosto {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, dim.left(), dim.top(), dim.right(), dim.bottom()))
        crop.append(i)
        crop[i] = resize[max(0, dim.top()): min(dim.bottom(), 480),
                    max(0, dim.left()): min(dim.right(), 360)]

        cv2.rectangle(display, (dim.left()*2, dim.top()*2), (dim.left()*2 + dim.right()*2 - dim.left()*2, dim.top()*2 + dim.bottom()*2 - dim.top()*2), (0, 255, 0), 1)

        text = "Rosto {}".format(i+1)
        cv2.putText(display, text, (dim.left()*2, dim.top()*2-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Video Output - 480x360", display)
    cv2.moveWindow("Video Output - 480x360", 0, 0)

    if len(crop) > 0:
        showCrop(crop, 0, len(crop))
        saveCrop(crop)

    k = cv2.waitKey(10) & 0xFF
    if k == ord('c'):
        break

cv2.destroyAllWindows()

time.sleep(1.0)

if len(crop) > 0:
    delet = ''
    while True:
        delet = input("Manter pasta(s) criada(s)? [s/n] ")
        if delet == 's' or delet == 'n':
            break

    if delet == 'n':
        delCrop(crop)
        print("Pasta(s) apagada(s)")
    else:
        print("Pasta(s) conservada(s)")

cap.release()
