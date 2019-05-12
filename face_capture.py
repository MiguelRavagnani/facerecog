import argparse
import os
import shutil
import time
import numpy as np
import imutils
import math
import tensorflow as tf
import pickle
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition
import sklearn.preprocessing
import cv2
import dlib

PATH_TO_SCRIPT = os.path.dirname(os.path.realpath(__file__))

PATH = PATH_TO_SCRIPT + "/data/mydataset/raw/"

FILENAMES = os.listdir(PATH)
FILENAMES.sort()

NAME = []
for filename in FILENAMES:
    if os.path.isdir(os.path.join(os.path.abspath(PATH), filename)):
        NAME.append(filename)



PICKLE_IN = open("X_train.pickle","rb")
X_TRAIN = pickle.load(PICKLE_IN)

PICKLE_IN = open("Y_train.pickle","rb")
Y_TRAIN_RAW = pickle.load(PICKLE_IN)

PICKLE_IN = open("X_test.pickle","rb")
X_TEST = pickle.load(PICKLE_IN)

PICKLE_IN = open("Y_test.pickle","rb")
Y_TEST_RAW = pickle.load(PICKLE_IN)

"""
Y_AUX = Y_TRAIN_RAW
for index in range(len(Y_AUX)-1):
    if index == len(Y_AUX):
        Y_TRAIN_RAW[index] = Y_AUX[index]
    elif index == 0:
        Y_TRAIN_RAW[index] = Y_AUX[index]
    else:
        Y_TRAIN_RAW[index] = Y_AUX[index-1]
"""

label_binarizer_1 = sklearn.preprocessing.LabelBinarizer()
label_binarizer_1.fit(range(max(Y_TRAIN_RAW)+1))
Y_TRAIN = label_binarizer_1.transform(Y_TRAIN_RAW)
label_binarizer_2 = sklearn.preprocessing.LabelBinarizer()
label_binarizer_2.fit(range(max(Y_TEST_RAW)))
Y_TEST = label_binarizer_2.transform(Y_TEST_RAW)
#Y_TRAIN = (-1)*np.ones((len(Y_TRAIN_RAW), Y_TRAIN_RAW.max()+1))
#Y_TRAIN[np.arange(len(Y_TRAIN_RAW)), (Y_TRAIN_RAW)] = 1

#Y_TEST = (-1)*np.ones((len(Y_TEST_RAW), Y_TRAIN_RAW.max()+1))
#Y_TEST[np.arange(len(Y_TEST_RAW)), (Y_TEST_RAW)] = 1

N_IMPUT = len(X_TRAIN[0])
#N_HIDDEN1 = int((len(X_TRAIN) + Y_TRAIN_RAW.max())/2)
#N_HIDDEN2 = int((len(X_TRAIN) + N_HIDDEN1)/2)
N_HIDDEN1 = int((len(X_TRAIN)))
N_HIDDEN2 = int((N_HIDDEN1)/2)
N_OUTPUT = Y_TRAIN_RAW.max()+1

#hiperparâmetros
LEARNING_RATE = 1e-4
N_ITER = len(X_TRAIN)
HM_EPOCHS = 20
BATCH_SIZE = 128
DROPOUT = 0.5

tf.reset_default_graph()

#Gráfico do TensorFlow
X_PH = tf.placeholder('float', [None, N_IMPUT])
Y_PH = tf.placeholder('float', [None, N_OUTPUT])
DROP_OUT_CTRL = tf.placeholder(tf.float32)


def multilayer_perceptron(DATA):
    #tf.cast(DATA, tf.float32)
    LAYER_1 = tf.add(tf.matmul(DATA, WEIGHTS['w1']), BIASES['b1'])
    LAYER_1 = tf.nn.relu(LAYER_1)

    LAYER_2 = tf.add(tf.matmul(LAYER_1, WEIGHTS['w2']), BIASES['b2'])
    LAYER_2 = tf.nn.relu(LAYER_2)

    LAYER_DROP = tf.nn.dropout(LAYER_2, DROPOUT)
    LAYER_OUTPUT = tf.matmul(LAYER_DROP, WEIGHTS['out']) + BIASES['out']
    return LAYER_OUTPUT



WEIGHTS = {
    'w1' : tf.Variable(tf.random_normal([N_IMPUT, N_HIDDEN1])),
    'w2' : tf.Variable(tf.random_normal([N_HIDDEN1, N_HIDDEN2])),
    'out' : tf.Variable(tf.random_normal([N_HIDDEN2, N_OUTPUT])),
}

BIASES = {
    'b1' : tf.Variable(tf.random_normal([N_HIDDEN1])),
    'b2' : tf.Variable(tf.random_normal([N_HIDDEN2])),
    'out' : tf.Variable(tf.random_normal([N_OUTPUT])),
}


saver = tf.train.Saver()


def use_neural_network(DATA):
    prediction = multilayer_perceptron(X_PH)
    with tf.Session() as SESS:
        SESS.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('model.ckpt.meta')
        saver.restore(SESS, PATH_TO_SCRIPT + "/model.ckpt")

        features = np.array(DATA)
        result = (SESS.run(tf.argmax(prediction.eval(feed_dict={X_PH:[features], Y_PH:Y_TRAIN}),1)))
        return result

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
FA = FaceAligner(PREDICTOR, desiredFaceWidth=480)

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

        FILE_ENCODING = [] # Create an empty list for saving encoded files    
        #IMAGE = face_recognition.load_image_file(IMAGE) # Run your load command
        image_encoding = face_recognition.face_encodings(IMAGE) # Run your encoding command
        FILE_ENCODING.append(image_encoding[0]) # Append the results to encoding_for_file list
        FILE_ENCODING = np.array(FILE_ENCODING)
        FILE_ENCODING.resize((1, 128)) # Resize using your command
        FILE_ENCODING = np.squeeze(FILE_ENCODING)
        OUTPUT_NUM = use_neural_network(FILE_ENCODING)
        OUTPUT_NUM = np.asscalar(OUTPUT_NUM)
        #OUTPUT_NAME = NAME.index(OUTPUT_NUM)
        OUTPUT_NAME = NAME[OUTPUT_NUM]
        print(OUTPUT_NAME)
        print(OUTPUT_NUM)
        

        if len(DETECT_RET) > 0:
            TEXT = "{} rosto(s) encontrado(s)".format(len(DETECT_RET))
            cv2.putText(RESIZE, TEXT, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.rectangle(RESIZE, (x, (y + h)), (x + (x + w) - x, (y + h) + y - (y + h)), (0, 255, 0), 1)
        TEXT = "1"
        TEXT = "{}".format(OUTPUT_NAME)
        cv2.putText(RESIZE, TEXT, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        i = i + 1 
    cv2.imshow("Video Output Original", RESIZE)
    cv2.moveWindow("Video Output", 0, 0)
    """    
    if len(FACE_LIST) > 0:
        show_crop(FACE_LIST, 0, len(FACE_LIST))
        save_crop(FACE_LIST)
    """
    k = cv2.waitKey(10) & 0xFF
    if k == ord('c'):
        break

cv2.destroyAllWindows()

time.sleep(1.0)

CAP.release()
