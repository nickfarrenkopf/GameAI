import os
import time
import numpy as np
import tensorflow as tf


### LAYERS ###

def weight(shape):
    """ creates weights tensor of desired shape """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias(shape, value=0.1):
    """ creates bias tensor of desired shape """
    return tf.Variable(tf.constant(value, shape=shape))


### CREATE ###

def create(network_path, name, size, hidden, n_classes, e=1e-8):
    """ create multi class classification network """

    with tf.device('/gpu:0'):

        # shapes
        n_layers = len(hidden) 
        input_shape = (None, size)
        output_shape = (None, n_classes)

        # name and save path
        classify_name = 'CLASS_{}_{}_{}_{}'.format(name, size, n_layers,
                                                   n_classes)
        save_path = os.path.join(network_path, classify_name)
        print(save_path)
        
        # start
        start = time.time()

        # set variable scope    
        with tf.variable_scope(classify_name):

            # placeholders
            inputs = tf.placeholder(tf.float32, input_shape, name='inputs')
            outputs = tf.placeholder(tf.float32, output_shape, name='outputs')
            alpha = tf.placeholder(tf.float32, name='alpha')
            
            # hidden layers
            current = inputs
            for i in range(len(hidden)):
                # weight and bias
                W = weight([int(current.shape[1]), hidden[i]])
                b = bias([hidden[i]])

                # output
                output = tf.nn.relu(tf.add(tf.matmul(current, W), b))
                current = output

            # final layer
            W_f = weight([int(current.shape[1]), n_classes])
            b_f = bias([n_classes])
            logits = tf.add(tf.matmul(current, W_f), b_f, name='logits')

            # metrics
            preds = tf.argmax(logits, axis=1, name='preds')
            probs = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                            labels=outputs,
                                                            name='probs')
            correct = tf.equal(preds, tf.argmax(outputs, axis=1),
                               name='correct')
            accs = tf.reduce_mean(tf.cast(correct, tf.float32), name='accs')

            # training
            loss = tf.reduce_mean(probs, name='cost')
            optimizer = tf.train.AdamOptimizer(alpha, epsilon=e).minimize(loss)
            tf.add_to_collection('{}_train'.format(classify_name), optimizer)

            # load session
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True))
            sess.run(tf.global_variables_initializer())

            # save network
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print('Created {} network in {}'.format(classify_name,
                                                    time.time() - start))


