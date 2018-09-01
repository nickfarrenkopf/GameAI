



class Environments(object):
    """ """


    def __init__(self, game_path, game_name):
        """ """
        self.game_path = game_path
        self.game_name = game_name
    


        self.window = 0




    ### WINDOW ###

    def set_window():
        """ """
        pass

    def get_window():
        """ """
        pass
    


    ### GAMESTATE ###

    def get_gamestate():
        """ """
        return self.get_window():


    ### NETWORK ###

    def train_model_network(self, state_data, action_data):
        """ """
        n_actions = len(action_data) - 1
        for i in range(n_actions):
            data = state_data[i] + action_data[i]
            label = state_data[i + 1]
            network.train(data, labels) ####


    






