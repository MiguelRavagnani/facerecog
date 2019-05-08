import tensorflow as tf
import pickle
import numpy as np
import os

import sklearn.preprocessing

PICKLE_IN = open("X_train.pickle","rb")
X_TRAIN = pickle.load(PICKLE_IN)

PICKLE_IN = open("Y_train.pickle","rb")
Y_TRAIN_RAW = pickle.load(PICKLE_IN)

PICKLE_IN = open("X_test.pickle","rb")
X_TEST = pickle.load(PICKLE_IN)

PICKLE_IN = open("Y_test.pickle","rb")
Y_TEST_RAW = pickle.load(PICKLE_IN)


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
N_HIDDEN1 = int((len(X_TRAIN))/2)
N_HIDDEN2 = int((N_HIDDEN1)/2)
N_OUTPUT = Y_TRAIN_RAW.max()+1

#hiperparâmetros
LEARNING_RATE = 1e-4
N_ITER = len(X_TRAIN)
HM_EPOCHS = 20
BATCH_SIZE = 128
DROPOUT = 0.5

#Gráfico do TensorFlow
x = tf.placeholder('float', [None, N_IMPUT])
y = tf.placeholder('float', [None, N_OUTPUT])
DROP_OUT_CTRL = tf.placeholder(tf.float32)


def multilayer_perceptron(DATA):

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
#função de perda: cross-entropy (log-loss)
#otimizador: AdamOptimizer

saver = tf.train.Saver()

def train_neural_network(DATA):
    PRED = multilayer_perceptron(DATA)

    LOG_LOSS = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = PRED))
    TRAIN_STEP = tf.train.AdamOptimizer(LEARNING_RATE).minimize(LOG_LOSS)

    #avaliação de precisão
    FIX_PREDICT = tf.equal(tf.argmax(PRED, 1), tf.argmax(y, 1))
    ACCURACY = tf.reduce_mean(tf.cast(FIX_PREDICT, tf.float32))

    #INIT = tf.global_variables_initializer()
    #SESS = tf.Session()
    #SESS.run(INIT)

    with tf.Session() as SESS:
        SESS.run(tf.global_variables_initializer())
        for EPOCH in range(HM_EPOCHS):
            print("Starting Epoch {}\n".format(EPOCH+1))
            for i in range(N_ITER):
                BATCH_STEP = 0
                while BATCH_STEP < len(X_TRAIN):
                    START = BATCH_STEP
                    END = BATCH_STEP+BATCH_SIZE

                    batch_x = X_TRAIN[START:END]
                    batch_y = Y_TRAIN[START:END]
                    
                    SESS.run(TRAIN_STEP, feed_dict={x: batch_x, y: batch_y})
                    BATCH_STEP += BATCH_SIZE
                if i%100 == 0:
                    MINIBATCH_LOSS, MINIBATCH_ACCURACY = SESS.run([LOG_LOSS, ACCURACY], feed_dict={x: batch_x, y: batch_y, DROP_OUT_CTRL:1.0})
                    print("Iteration", str(i), "\t| Loss =", str(MINIBATCH_LOSS), "\t| Accuracy =", str(MINIBATCH_ACCURACY))
                i += (BATCH_SIZE)

            TEST_ACCURACY = SESS.run(ACCURACY, feed_dict={x: X_TEST, y: Y_TEST, DROP_OUT_CTRL:1.0})
            print("\nEpoch {} | Accuracy on test: {}".format(EPOCH+1, TEST_ACCURACY))
        PATH_TO_SCRIPT = os.path.dirname(os.path.realpath(__file__))
        save_path = saver.save(SESS, PATH_TO_SCRIPT + "/model.ckpt")
        print("Model saved in path: %s" % save_path)            

train_neural_network(x)