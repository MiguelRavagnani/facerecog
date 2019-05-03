import tensorflow as tf
import pickle
import numpy as np

pickle_in = open("X.pickle","rb")
x_train = pickle.load(pickle_in)

pickle_in = open("Y.pickle","rb")
y_train_raw = pickle.load(pickle_in)

y_train = np.zeros((len(y_train_raw), y_train_raw.max()+1))
y_train[np.arange(len(y_train_raw)), y_train_raw] = 1

print(x_train)
print(y_train)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = y_train_raw.max()+1
batch_size = 100

total_batches = int(2307/batch_size)

hm_epochs = 15

x = tf.placeholder('float', [None, len(x_train[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(x_train[0]), n_nodes_hl1])),
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

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        tf_log = 'tf.log'

        for epoch in range(hm_epochs):

            try:
                epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
                print('STARTING:',epoch)
            except:
                epoch = 1

            if epoch != 1:
                saver.restore(sess,"model.ckpt")
            epoch_loss = 1

            i = 0
            while i < len(x_train):
                start = i
                end = i+batch_size

                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
                
            saver.save(sess, "model.ckpt")
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch +=1

    
train_neural_network(x)
