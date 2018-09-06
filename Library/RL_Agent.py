import os
from os.path import join
from keras.models import load_model


class Agent(object):
    """

    Agent is pacman player 1
    Agent is pacman player 2
    Agent is digdug player
    Agent is aw2 player

    """

    def __init__(self, game_path):
        """ """

        # environment
        self.base_path, self.game_name = os.path.split(game_path)
        
        # file location
        self.info_path = os.path.join(games_path, 'agents.txt')
        self.network_path = os.path.join(games_path, 'value_network.h5')
        self.network = load_model(self.network_path)

        self.reward_network_path = join(games_path, 'reward_network.h5')

        # agent info
        self.agent_id, self.agent_name, self.actions = self.load_text()


    ### FILE ###

    def load_text(self):
        """ FIX ME """
        with open(self.info_path, 'r') as file:
            data = file.read().split('\n')
        for row in data:
            if ':' not in row:
                agent_id = row
            else:
                key, value = row.split(':')
                if key == 'name':
                    name = value
                if key == 'actions':
                    actions = value
        return agent_id, name, actions.split(',')

    def create_text(self, name, actions):
        """ FIX ME """
        index = 1
        new_path = self.info_path.replace('.', '._1')
        data = '\n'.join([index, name, ','.join(actions)])
        with open(new_path, 'w') as file:
            file.write(data)
        print('Created new agent at {}'.format(new_path))


    ### ACTION ###

    def choose_action(self, gamestate):
        """ """
        pass

    def take_action(self, action):
        """ """
        if action not in self.actions:
            print('Action not in list: {}'.format(self.actions))
            return False
        Screen.send_keys(a)
        

### PROGRAM ###

if __name__ == '__main__':

    path = 'C:\\Users\\Nick\\Desktop\\Ava\\Programs\\GameAI\\games\\pacman'
    ag = Agent(path)


