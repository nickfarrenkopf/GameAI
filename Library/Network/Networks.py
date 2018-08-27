import os
import time
import itertools
import numpy as np
import tensorflow as tf


class Network(object):
    """ base network """

    def __init__(self, save_path, name):
        """ params - name, path, session, and graph """
        self.name = name
        self.save_path = os.path.join(save_path, name)
        self.graph = tf.Graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True),
                               graph=self.graph)
        self.load_main()

    ### FILE ###

    def load_main(self):
        """ load main parts of network """
        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(self.save_path + '.meta')
                saver.restore(self.sess, self.save_path)
                get_tensor = self.graph.get_tensor_by_name
                self.inputs = get_tensor('{}/inputs:0'.format(self.name))
                self.outputs = get_tensor('{}/outputs:0'.format(self.name))
                self.alpha = get_tensor('{}/alpha:0'.format(self.name))
                self.cost = get_tensor('{}/cost:0'.format(self.name))
                self.train = tf.get_collection('{}_train'.format(self.name))[0]

    def save_network(self, step=0):
        """ saves network to save path """
        print('Saving network {}'.format(self.name))
        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.Saver()
                if step == 0:
                    saver.save(self.sess, self.save_path)
                else:
                    saver.save(self.sess, self.save_path, global_step=step)


class NetworkAuto(Network):
    """ Convolutional Autoencoder Network """

    def __init__(self, save_path, name):
        """ initializes network and load extras """
        start = time.time()
        Network.__init__(self, save_path, name)
        self.load_extras()
        #print('{} loaded in {} '.format(name, time.time() - start))

    ### FILE ###

    def load_extras(self):
        """ """
        with self.sess.as_default():
            with self.graph.as_default():
                get_tensor = self.graph.get_tensor_by_name
                self.flat = get_tensor('{}/flat:0'.format(self.name))

    def batch_me(self, data, func, alpha=0, batch=1):
        """ """
        datas = []
        for i in range(data.shape[0] // batch + 1):
            subdata = data[batch * i : batch * (i + 1)]
            if subdata.shape[0] < batch:
                subdata = np.array(list(subdata) +
                                   list(data[:batch - subdata.shape[0]]))
            feed = {self.inputs: subdata, self.alpha : alpha}
            res = self.sess.run(func, feed)
            if res is None:
                res = [0]
            if isinstance(res, np.float32):
                res = np.array([res])
            datas.append(res)
        r = np.array(list(itertools.chain.from_iterable(datas)))
        return r[:data.shape[0]]
            

    ### TENSORS ###

    def get_flat(self, input_data):
        """ return middle flat layer of network given feed """
        #print('Flat shape {}'.format(input_data.shape))
        return self.batch_me(input_data, self.flat)

    def get_outputs(self, input_data):
        """ return outputs of network given feed """
        return self.batch_me(input_data, self.outputs)

    def get_cost(self, input_data):
        """ return cost of network given feed """
        stuff = self.batch_me(input_data, self.cost)
        return np.mean(stuff)

    def train_network(self, input_data, alpha):
        """ train network given feed """
        #print(input_data.shape)
        stuff = self.batch_me(input_data, self.train, alpha=alpha)


class NetworkClass(Network):
    """ Binary Classification Network """
    
    def __init__(self, save_path, name, auto_network, n_classes):
        """ initializes network given session and save path """
        self.auto_network = auto_network
        self.n_classes = int(n_classes)
        
        start = time.time()
        Network.__init__(self, save_path, name)
        self.load_extras()
        #print('{} loaded in {} '.format(name, time.time() - start))

    ### FILE ###

    def load_extras(self):
        """ creates saver, loads meta graph, gets params """
        with self.sess.as_default():
            with self.graph.as_default():
                # d = [n.name for n in tf.get_default_graph().as_graph_def().node]
                get_tensor = self.graph.get_tensor_by_name
                self.logits = get_tensor('{}/logits:0'.format(self.name))
                self.probs = get_tensor('{}/probs:0'.format(self.name))
                self.preds = get_tensor('{}/preds:0'.format(self.name))
                self.correct = get_tensor('{}/correct:0'.format(self.name))
                self.accs = get_tensor('{}/accs:0'.format(self.name))
                

    ### AUTO NETWORK ###

    def get_base_outputs(self, input_data):
        """ """
        #print('Input shape: {}'.format(input_data.shape))
        node_net_in = self.auto_network.get_flat(input_data)
        node_net_in = np.reshape(node_net_in, (node_net_in.shape[0], -1))
        return node_net_in


    ### TENSORS ###

    def get_logits(self, input_data):
        """ return logits of network given feed """
        feed = {self.inputs: self.get_base_outputs(input_data)}
        return self.sess.run(self.logits, feed)
    
    def get_probs(self, input_data):
        """ return probs of network given feed """
        feed = {self.inputs: self.get_base_outputs(input_data)}
        return self.sess.run(self.probs, feed)

    def get_preds(self, input_data):
        """ return preds of network given feed """
        feed = {self.inputs: self.get_base_outputs(input_data)}
        return self.sess.run(self.preds, feed)

    def get_correct(self, input_data, output_data):
        """ return probs of network given feed """
        feed = {self.inputs: self.get_base_outputs(input_data),
                self.outputs: np.reshape(output_data, (-1, self.n_classes))}
        return self.sess.run(self.correct, feed)

    def get_accuracy(self, input_data, output_data):
        """ return probs of network given feed """
        feed = {self.inputs: self.get_base_outputs(input_data),
                self.outputs: np.reshape(output_data, (-1, self.n_classes))}
        return self.sess.run(self.accs, feed)

    def get_cost(self, input_data, output_data):
        """ return cost of network given feed """
        feed = {self.inputs: input_data}
        feed = {self.inputs: self.get_base_outputs(input_data),
                self.outputs: np.reshape(output_data, (-1, self.n_classes))}
        return self.sess.run(self.cost, feed)

    def train_network(self, input_data, output_data, alpha):
        """ train network given feed """
        feed = {self.inputs: self.get_base_outputs(input_data),
                self.outputs: np.reshape(output_data, (-1, self.n_classes)),
                self.alpha: alpha}
        self.sess.run(self.train, feed)

