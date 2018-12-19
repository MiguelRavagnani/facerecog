from imutils.video import VideoStream
from imutils import face_utils
import imutils
import cv2
import argparse
import time
import dlib
import numpy as np

def convToBoundingBox(rectang): # Converte, para o formato (x, y, w, h), uma Bounding box encontrada

    x = rectang.left()
    y = rectang.top()
    w = rectang.right() - x
    h = rectang.bottom() - y

    return (x, y, w, h)

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

cap = cv2.VideoCapture(0) # Caputa vídeo
cap.set(5,30)

while True: # Reproduz vídeo até que uma tecla definida seja pressionada
    ret, image = cap.read()
    image = cv2.flip(image, 1, 0)
    resize = cv2.resize(image, (480, 360))
    display = cv2.resize(image, (480, 360))
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    detectRect = detector(gray, 1)

    if len(detectRect) > 0:
        text = "{} rosto(s) encontrado(s)".format(len(detectRect))
        cv2.putText(resize, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        	0.7, (0, 0, 255), 2)

    crop = []
    for i, dim in enumerate(detectRect):
        print("Rosto {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, dim.left(), dim.top(), dim.right(), dim.bottom()))
        crop.append(i)
        crop[i] = resize[max(0, dim.top()): min(dim.bottom(), 480),
                    max(0, dim.left()): min(dim.right(), 360)]

        cv2.rectangle(resize, (dim.left(), dim.top()), (dim.left() + dim.right() - dim.left(), dim.top() + dim.bottom() - dim.top()), (0, 255, 0), 1)

        text = "Rosto {}".format(i+1)
        cv2.putText(resize, text, (dim.left(), dim.top()-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    cv2.imshow("Video Output - 480x360", resize)
    cv2.moveWindow("Video Output - 480x360".format(1), 0, 0)

    if len(crop) > 0:
        assert crop[0] is not None
        cv2.imshow("Crop {}".format(1), crop[0])
        cv2.moveWindow("Crop {}".format(1), 580, 0)
        if len(crop) > 1:
            assert crop[1] is not None
            cv2.imshow("Crop {}".format(2), crop[1])
            cv2.moveWindow("Crop {}".format(2), 580, 0 + 300)
            if len(crop) > 2:
                assert crop[2] is not None
                cv2.imshow("Crop {}".format(3), crop[2])
                cv2.moveWindow("Crop {}".format(3), 580 + 300, 0)

    k = cv2.waitKey(10) & 0xFF
    if k == ord('c'):
        break

cv2.destroyAllWindows()
cap.release()
