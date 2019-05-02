import argparse
import os
import shutil
import time
import numpy as np
import imutils
import math
import tensorflow as tf
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

import cv2

import dlib



n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 214
hm_data = 2521

batch_size = 100
hm_epochs = 15

X_PH = tf.placeholder('float')
Y_PH = tf.placeholder('float')


current_epoch = tf.Variable(1)

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([8, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    
    return output



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

def use_neural_network(input_data):
    prediction = neural_network_model(X_PH)
        
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess,"model.ckpt")

        features = np.array(input_data)
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={X_PH:[features]}),1)))
        print(result)

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
        
        for k, d in enumerate(DETECT_RET):  
            SHAPE = PREDICTOR(FACE_LIST[i], d)
        for j in range(68):
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

        #print("Normalized: J1 {} J2 {} J3 {} J4 {} J5 {} J6 {} J7 {} J8 {}".format(J_NORM[0], J_NORM[1], J_NORM[2], J_NORM[3], J_NORM[4], J_NORM[5], J_NORM[6], J_NORM[7]))
        
        use_neural_network(J_NORM)

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
