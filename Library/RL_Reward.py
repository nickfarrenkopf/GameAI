from keras.models import Sequential
from keras.layers import Dense

from Library.General import DataThings as DT
from Library import RL_Component


class Reward(RL_Component):
    """ """

    def __init__(self, game, agent):
        """ """
        
        # RL components
        self.name = 'reward'
        self.game = game
        self.agent = agent

        # load network
        RL_Component.__init__(self, agent.reward_network_path)
        self.load_network()


    ### NETWORK ###

    def create_network(self, n_hidden=64, n_layers=2):
        """ create ANN reward nertwork with keras """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.game.state_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(self.game.n_rewards, activation='softmax'))
        network.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
        self.network = network


    ### RUN TIME ###

    def get_reward(self):
        """ get prediction given environment gamestate """
        return self.network.predict(self.game.get_gamestate(), verbose=0)


    ### TRAIN - OFFLINE ###

    def train_network_offline(self, epochs=3000, n_loop=10, save_me=False):
        """ train keras network with saved gamestate data """
        # get data
        pre_data, _, labels = self.gamedata_files_to_reward_network_inputs()
        data = self.game.auto_network.get_flat(pre_data)
        # loop and fit
        for _ in range(n_loop):
            self.network.fit(data, labels, epochs=epochs//n_loop, verbose=0)
            metrics = self.network.evaluate(data, labels, verbose=0)
            print('Metrics: {}'.format(metrics))
        self.save_network(save_me)

    def test_network_offline(self):
        """ test network against all data """
        # group by zeros and non zero labels
        pass

    def gamedata_files_to_reward_network_inputs(self, shuffle_me=True):
        """ load gamestate images based on labeled class data """
        # find gamestate indexes to load
        idxs, labels = self.agent.load_reward_indexes_and_labels()
        _, labels, files = self.add_zero_reward_labels(idxs, labels)
        files, labels = self.shuffle_mes(files, labels, shuffle_me)
        # load data and format to network
        data = DT.load_images(files)
        one_hot = DT.to_one_hot_labels(labels)
        return data, labels, one_hot

    def add_zero_reward_labels(self, idxs, labels):
        """ add zero reward labels given pre-labeled indexes """
        files = self.game.get_all_gamedata_paths()
        files_to_load = [files[i] for i in idxs]
        for i in range(0, len(files), len(files) // len(labels) - 1):
            if i not in idxs:
                files_to_load.append(files[i])
                labels.append(0)
        return idxs, labels, files_to_load


    ### TRAIN - ONLINE ###  

    def train_network_online(self, epochs=10):
        """ ??? """
        pass

    def test_network_online(self):
        """ """
        pass


