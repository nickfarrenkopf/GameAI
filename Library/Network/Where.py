import os
import time
import numpy as np
import tensorflow as tf


### LAYERS ###

def weight(shape):
    """ weights tensor with shape """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias(shape, value=0.1):
    """ bias tensor with shape """
    return tf.Variable(tf.constant(value, shape=shape))


### PARAMS ###

def create(network_path, name, size, hidden, n_classes, e=1e-8, batch_size=4):
    """ create where classification network """

    with tf.device('/gpu:0'):

        # shapes
        n_layers = len(hidden) 
        input_shape = (None, size)
        output_shape = (batch_size, n_classes)

        # name and save path
        where_name = 'WHERE_{}_{}_{}_{}'.format(name, size, n_layers,
                                                n_classes)
        save_path = os.path.join(network_path, where_name)
        print(save_path)
        
        # start
        start = time.time()

        # set variable scope    
        with tf.variable_scope(where_name):

            # placeholders
            inputs = tf.placeholder(tf.float32, input_shape, name='inputs')
            outputs = tf.placeholder(tf.float32, output_shape, name='outputs')
            alpha = tf.placeholder(tf.float32, name='alpha')
            
            # hidden layers
            current = inputs
            print(current.shape)
            for i in range(len(hidden)):
                # weight and bias
                W = weight([int(current.shape[1]), hidden[i]])
                b = bias([hidden[i]])

                # output
                output = tf.nn.relu(tf.add(tf.matmul(current, W), b))
                current = output
                print(current.shape)

            # final layer
            W_f = weight([int(current.shape[1]), n_classes])
            b_f = bias([n_classes])
            logits = tf.add(tf.matmul(current, W_f), b_f, name='logits')
            print(logits.shape)

            # REG metrics
            preds = tf.square(logits - outputs, name='preds')
            probs  = tf.square(logits - outputs, name='probs')
            correct = tf.square(logits - outputs, name='correct')
            accs = tf.square(logits - outputs, name='accs')

            # CLASS metrics
            #preds = tf.argmax(logits, axis=1, name='preds')
            #probs = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
            #                                                labels=outputs,
            #                                                name='probs')
            #correct = tf.equal(preds, tf.argmax(outputs, axis=1),
            #                   name='correct')
            #accs = tf.reduce_mean(tf.cast(correct, tf.float32), name='accs')

            # training
            loss = tf.reduce_mean(probs, name='cost')
            optimizer = tf.train.AdamOptimizer(alpha, epsilon=e).minimize(loss)
            tf.add_to_collection('{}_train'.format(where_name), optimizer)

            # load session
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True))
            sess.run(tf.global_variables_initializer())

            # save network
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print('Created {} network in {}'.format(where_name,
                                                    time.time() - start))


