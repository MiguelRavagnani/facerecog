from imutils.video import VideoStream
from imutils import face_utils
import imutils
import cv2
import argparse
import time
import dlib
import numpy as np
"""
def printBox(index, X, Y, W, H, img, aux, dim, boxDrawing):
    for i in range(0, index+1):
        dim.append(i)
        X.append(i)
        Y.append(i)
        W.append(i)
        H.append(i)

    dim[aux] = face_utils.rect_to_bb(boxDrawing)
    (X[index], Y[index], W[index], H[index]) = dim[index]

    cv2.rectangle(img, (X[index], Y[index]), (X[index] + W[index], Y[index] + H[index]),
	    (0, 255, 0),1)
    text = "SO FUNCIONA QUANDO QUER {}".format(aux+1)
    cv2.putText(img, text, (X[index], Y[index]-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if aux >= 1:
        return (printBox(index+1, X, Y, W, H, img, aux-1, dim, boxDrawing))
    else:
        return (dim[index])
"""

def convToBoundingBox(rectang): # Converte, para o formato (x, y, w, h), uma Bounding box encontrada

    x = rectang.left()
    y = rectang.top()
    w = rectang.right() - x
    h = rectang.bottom() - y

    return (x, y, w, h)

def imageProp(img): # Imprime a imagem em array, o tipo de imagem e seu tamanho
    #print("Imagem: ")
    #print(img)

    print("Tipo de imagem: ")
    print(type(img))

    print("Tamanho da imagem: ")
    print(img.shape)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="Caminho para Shape Predictor")
args = vars(ap.parse_args()) # shape_predictor_68_face_landmarks.dat

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


cap = cv2.VideoCapture(0) # Caputa vídeo

print("[TIME SLEEP]: 3s..")
time.sleep(1.0)
print("[TIME SLEEP]: 2s..")
time.sleep(1.0)
print("[TIME SLEEP]: 1s..")
time.sleep(1.0)


while True: # Reproduz vídeo até que uma tecla definida seja pressionada
    ret, image = cap.read()
    image = cv2.flip(image, 1)
    resize = cv2.resize(image, (480, 360))
    display = cv2.resize(image, (480, 360))
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    detectRect = detector(gray, 1)

    if len(detectRect) > 0:
            text = "{} rosto(s) encontrado(s)".format(len(detectRect))
            cv2.putText(resize, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            	0.7, (0, 0, 255), 2)

    boxNumber = []
    boxDim = []
    bX = []
    bY = []
    bW = []
    bH = []
#    X = []
#    Y = []
#    W = []
#    H = []


    for aux in range(0, len(detectRect)):
        boxDim.append(aux)
        bX.append(aux)
        bY.append(aux)
        bW.append(aux)
        bH.append(aux)

    #print(len(boxDim))

    for boxDrawing in detectRect:
        for aux in range(0, len(detectRect)):
            boxDim[aux] = face_utils.rect_to_bb(boxDrawing)
            (bX[aux], bY[aux], bW[aux], bH[aux]) = boxDim[aux]
        #(bX, bY, bW, bH) = face_utils.rect_to_bb(boxDrawing)

            cv2.rectangle(resize, (bX[aux], bY[aux]), (bX[aux] + bW[aux], bY[aux] + bH[aux]),
        	    (0, 255, 0),1)
            text = "SO FUNCIONA QUANDO QUER {}".format(aux+1)
            cv2.putText(resize, text, (bX[aux], bY[aux]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#        for j in range(0, len(detectRect)):
#            boxNumber.append(len(detectRect))
#            bX.append(len(detectRect))
#            bY.append(len(detectRect))
#            bW.append(len(detectRect))
#            bH.append(len(detectRect))
#            boxNumber[j] = printBox(0, X, Y, W, H, resize, j, boxDim, boxDrawing)
#            (bX[j], bY[j], bW[j], bH[j]) = boxNumber[j]
        shape = predictor(gray, boxDrawing)
        shape = face_utils.shape_to_np(shape)

        for (i, (x, y)) in enumerate(shape):
            cv2.circle(resize, (x, y), 1, (255, 0, 255), -1)
            cv2.putText(resize, str(i + 1), (x - 10, y - 10),
            	cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 255), 1)


    #cv2.imshow("Video Output - Original", image)
    #cv2.imshow("Video Output - Gray", gray)

    cv2.imshow("Video Output - 480x360", resize)

    roi = []

    if bY is not None and bX is not None and bH is not None and bW is not None:
        for i in range(0, len(detectRect)):
            roi.append(i)
            roi[i] = display[bY[i]:(bY[i] + bH[i]),bX[i]:(bX[i] + bW[i])]

    if roi:
        for i in range(0, len(roi)):
            cv2.imshow("Rosto {}".format(i+1), roi[i])

    k = cv2.waitKey(10) & 0xFF
    if k == ord('c'):
        break

imageProp(roi[0])
imageProp(roi[1])

cv2.destroyAllWindows()
cap.release()
