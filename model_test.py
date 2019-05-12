import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
s = tf.InteractiveSession()
import pickle
import numpy as np

pickle_in = open("X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("Y_train.pickle","rb")
y_train_raw = pickle.load(pickle_in)

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("Y_test.pickle","rb")
y_test_raw = pickle.load(pickle_in)


y_train = (-1)*np.ones((len(y_train_raw), y_train_raw.max()))
y_train[np.arange(len(y_train_raw)), y_train_raw-1] = 1

y_test = (-1)*np.ones((len(y_test_raw), y_test_raw.max()))
y_test[np.arange(len(y_test_raw)), y_test_raw-1] = 1



## Defining various initialization parameters for 784-512-256-10 MLP model
N_CLASSES = y_train_raw.max()
N_FEATURES = X_train.shape[1]
N_OUTPUT = y_train.shape[1]
NEU_LAYER_1 = 512
NEU_LAYER_2 = 256
LEARNING_RATE = 0.01
REG_RATE = 0.1

# Placeholders for the input data
input_X = tf.placeholder('float32',shape =(None,N_FEATURES),name="input_X")
input_y = tf.placeholder('float32',shape = (None,N_CLASSES),name='input_Y')
## for dropout layer
KEEP_PROB = tf.placeholder(tf.float32)

## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
WEIGHTS_1 = tf.Variable(tf.random_normal([N_FEATURES,NEU_LAYER_1], stddev=(1/tf.sqrt(float(N_FEATURES)))))
BIAS_1 = tf.Variable(tf.random_normal([NEU_LAYER_1]))
WEIGHTS_2 = tf.Variable(tf.random_normal([NEU_LAYER_1,NEU_LAYER_2], stddev=(1/tf.sqrt(float(NEU_LAYER_1)))))
BIAS_2 = tf.Variable(tf.random_normal([NEU_LAYER_2]))
WEIGHTS_3 = tf.Variable(tf.random_normal([NEU_LAYER_2,N_OUTPUT], stddev=(1/tf.sqrt(float(NEU_LAYER_2)))))
BIAS_3 = tf.Variable(tf.random_normal([N_OUTPUT]))

## Initializing weigths and biases
HIDDEN_LAYER_1 = tf.nn.relu(tf.matmul(input_X,WEIGHTS_1)+BIAS_1)
HIDDEN_LAYER_1_1 = tf.nn.dropout(HIDDEN_LAYER_1, KEEP_PROB)
HIDDEN_LAYER_2 = tf.nn.relu(tf.matmul(HIDDEN_LAYER_1_1,WEIGHTS_2)+BIAS_2)
HIDDEN_LAYER_2_2 = tf.nn.dropout(HIDDEN_LAYER_2, KEEP_PROB)
OUTPUT_LAYER = tf.sigmoid(tf.matmul(HIDDEN_LAYER_2_2,WEIGHTS_3) + BIAS_3)

## Defining the LOSS function
LOSS = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=OUTPUT_LAYER,labels=input_y)) \
        + REG_RATE*(tf.reduce_sum(tf.square(BIAS_3)) + tf.reduce_sum(tf.square(BIAS_3)))

## Variable learning rate
LEARNING_RATE = tf.train.exponential_decay(LEARNING_RATE, 0, 5, 0.85, staircase=True)
## Adam optimzer for finding the right weight
OPTIMIZER = tf.train.AdamOptimizer(LEARNING_RATE).minimize(LOSS,var_list=[WEIGHTS_3,WEIGHTS_3,WEIGHTS_3,
                                                                         BIAS_3,BIAS_3,BIAS_3])

## Metrics definition
CORRECT_PRED = tf.equal(tf.argmax(y_train,1), tf.argmax(OUTPUT_LAYER,1))
ACCURACY = tf.reduce_mean(tf.cast(CORRECT_PRED, tf.float32))

## Training parameters
BATCH_SIZE = 128
HM_EPOCHS=15
DROPOUT_PROB = 0.6
TRAINING_ACCURACY = []
TRAINING_LOSS = []
TESTING_ACCURACY = []

s.run(tf.global_variables_initializer())
for epoch in range(HM_EPOCHS):    
    ARR = np.arange(X_train.shape[0])
    np.random.shuffle(ARR)
    for index in range(0,X_train.shape[0],BATCH_SIZE):
        s.run(OPTIMIZER, {input_X: X_train[ARR[index:index+BATCH_SIZE]],
                          input_y: y_train[ARR[index:index+BATCH_SIZE]],
                        KEEP_PROB:DROPOUT_PROB})
    TRAINING_ACCURACY.append(s.run(ACCURACY, feed_dict= {input_X:X_train, 
                                                         input_y: y_train,KEEP_PROB:1}))
    TRAINING_LOSS.append(s.run(LOSS, {input_X: X_train, 
                                      input_y: y_train,KEEP_PROB:1}))
    
    ## Evaluation of model
    TESTING_ACCURACY.append(accuracy_score(y_test.argmax(1), 
                            s.run(OUTPUT_LAYER, {input_X: X_test,KEEP_PROB:1}).argmax(1)))
    print("Epoch:{0}, Train LOSS: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                    TRAINING_LOSS[epoch],
                                                                    TRAINING_ACCURACY[epoch],
                                                                   TESTING_ACCURACY[epoch]))