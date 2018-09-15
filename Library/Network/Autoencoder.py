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

def conv2d(x, W):
    """ 2D convolutional operation tensor """
    return tf.nn.conv2d(x, W, strides=c_strides, padding='SAME')

def deconv2d(x, W, output_shape):
    """ 2D unconvolutional operation tensor """
    return tf.nn.conv2d_transpose(x, W, output_shape, c_strides, padding='SAME')

def maxpool(x):
    """ maxpool operation with shape and strides """
    return tf.nn.max_pool(x, ksize=p_ksize, strides=p_strides, padding='SAME')

def maxpool_argmax(x):
    """ maxpool with argmax """
    _, mask = tf.nn.max_pool_with_argmax(x, ksize=p_ksize, strides=p_strides,
                                         padding='SAME')
    mask = tf.stop_gradient(mask)
    net = maxpool(x)
    return net, mask

def unpooled(net, mask, input_shape, output_shape, batch_size=4):
    """ unpool with reverse argmax through indicies """
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(batch_size, dtype=tf.int64),
                             shape=[batch_size, 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices and reshape update values to one dimension
    updates_size = tf.size(net)
    idxs = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(idxs, values, output_shape)
    return ret


### PARAMS ###

batch_size = 1
patch = 3
stride = 2
c_strides = [1, 1, 1, 1]
p_strides = [1, stride, stride, 1]
p_ksize = [1, stride, stride, 1]


### CREATE ###

def create(network_path, auto_name, h, w, hidden, length=3, e=1e-8):
    """ create convolutional autoencoder network """

    with tf.device('/gpu:0'):

        # shapes
        n_layers = len(hidden) 
        input_shape = (batch_size, h, w, length)
        flat_size = int(h * w / (stride * stride) ** n_layers * hidden[-1])
        flat_shape = (batch_size, flat_size)

        # name and save path
        auto_name = 'AUTO_{}_{}_{}_{}_{}'.format(auto_name, h, w, n_layers,
                                                 flat_size) ###
        save_path = os.path.join(network_path, auto_name)
        print(save_path)

        # set variable scope    
        with tf.variable_scope(auto_name):

            # start
            print('Creating network...')
            start = time.time()
            encoder = []
            shapes = []
            
            # placeholders
            inputs = tf.placeholder(tf.float32, input_shape, name='inputs')
            alpha = tf.placeholder(tf.float32, name='alpha')

            # encoder
            current = inputs
            for i in range(n_layers):
                
                # conv 1
                W1 = weight([patch, patch, int(current.shape[3]), hidden[i]])
                b1 = bias([hidden[i]])
                first = tf.nn.relu(tf.add(conv2d(current, W1), b1))
                
                # conv 2
                W2 = weight([patch, patch, int(first.shape[3]), hidden[i]])
                b2 = bias([hidden[i]])
                second = tf.nn.relu(tf.add(conv2d(first, W2), b2))
                
                # pool
                pool, mask = maxpool_argmax(second)

                # save values
                encoder.append((W1, W2, mask))
                shapes.append((first.shape, second.shape, pool.shape))
                current = pool
                print('Encode: {}'.format(current.shape))
                
            # flat layer
            mid = tf.reshape(current, flat_shape)
            #mid = tf.reshape(current, flat_shape, name='flat')

            # binary layer
            mean, _ = tf.metrics.mean(mid)
            binary = tf.cast(tf.greater(mid, mean), tf.float32, name='flat')

            # reverse
            current = binary
            encoder.reverse()
            shapes.reverse()
            print('Flat: {}'.format(current.shape))

            # decoder
            for i, sets in enumerate(shapes):
                
                # unpool
                pool = unpooled(current, encoder[i][2], current.shape, sets[-2],
                                batch_size=batch_size)
                    
                # conv 1
                W1 = encoder[i][1]
                # W1 = weight(encoder[i][1].shape)
                b1 = bias([W1.shape[3]])
                first = tf.nn.relu(tf.add(deconv2d(pool, W1, sets[-2]), b1))
                
                # conv 2
                nxtshp = input_shape if i + 1 == len(shapes) else shapes[i+1][-1]
                W2 = encoder[i][0]
                # W2 = weight(encoder[i][0].shape)
                b2 = bias([nxtshp[3]])
                second = tf.nn.relu(tf.add(deconv2d(first, W2, nxtshp), b2))

                # save values
                current = second
                print('Decode: {}'.format(current.shape))
            
            # metrics
            outputs = tf.identity(current, name='outputs')
            cost = tf.reduce_mean(tf.square(inputs - outputs), name='cost')
            optimizer = tf.train.AdamOptimizer(alpha, epsilon=e).minimize(cost)
            tf.add_to_collection('{}_train'.format(auto_name), optimizer)

            # load session
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True))
            sess.run(tf.global_variables_initializer())

            # save network
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print('Created {} network in {}'.format(auto_name,
                                                    time.time() - start))


