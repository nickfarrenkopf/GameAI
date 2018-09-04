


# GBA: ruld, ab, ltrt, stse => 10
# PAC: ruld => 4
# AW2: GBA => 10

# NETWORKS
# net1 - value network, St + At -> Qt
# net2 - action network, St + At -> 0,1





class Action(object):
    """ """


    def __init__(self):
        """ """
        pass



    def train_value_network(self, state_data, action_data):
        """ """
        n_actions = len(action_data) - 1
        for i in range(n_actions):
            old_value = 0
            new_value = 0
            data = state_data[i] + action_data[i]
            network.train(data, new_value)
