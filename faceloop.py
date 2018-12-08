from imutils import face_utils
import imutils
import cv2
import argparse
import time
import dlib

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

def convToBoundingBox(rectang): # Converte, para o formato (x, y, w, h), uma Bounding box encontrada

    x = rectang.left()
    y = rectang.top()
    w = rectang.right() - x
    h = rectang.bottom() - y

    return (x, y, w, h)

def imageProp(img): # Imprime a imagem em array, o tipo de imagem e seu tamanho
    print("Imagem: ")
    print(img)

    print("Tipo de imagem: ")
    print(type(img))

    print("Tamanho da imagem: ")
    print(img.shape)

while True: # Reproduz vídeo até que uma tecla definida seja pressionada
    ret, image = cap.read()
    resize = cv2.resize(image, (480, 360))
    display = cv2.resize(image, (480, 360))
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    detectRect = detector(gray, 0)

    if len(detectRect) > 0:
            text = "{} rosto(s) encontrado(s)".format(len(detectRect))
            cv2.putText(resize, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            	0.5, (0, 0, 255), 2)
    for boxDrawing in detectRect:

        (bX, bY, bW, bH) = face_utils.rect_to_bb(boxDrawing)

        cv2.rectangle(resize, (bX, bY), (bX + bW, bY + bH),
        	(0, 255, 0), 1)


        shape = predictor(gray, boxDrawing)
        shape = face_utils.shape_to_np(shape)

        for (i, (x, y)) in enumerate(shape):
            cv2.circle(resize, (x, y), 1, (255, 0, 255), -1)
            cv2.putText(resize, str(i + 1), (x - 10, y - 10),
            	cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)


    #cv2.imshow("Video Output - Original", image)
    #cv2.imshow("Video Output - Gray", gray)

    cv2.imshow("Video Output - 480x360", resize)
    #crop = display[bY:(bY + bH),bX:(bX + bW)]
    #cv2.imshow("Cropped Output", crop)


    k = cv2.waitKey(10) & 0xFF
    if k == 99:
        break

# imageProp(image)

cv2.destroyAllWindows()
cap.release()
