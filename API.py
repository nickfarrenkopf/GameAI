import os
import time

import data_things as dt
import paths
from Library import Screen
from Library import RL_Agent
from Library import RL_Environment
from Library import RL_Reward


from Library import NetworkAPI as NETS

### API ###

def run_program(train_me=False, save_me=True):
    """ """

    # clean previous game data
    clear_gamestate_data(game)
    count = 0

    # loop until done
    done = False
    while not done:

        # ENVIRONMENT
        GS = env.get_gamestate()

        # ACTION
        A = player.choose_action(GS)
        player.take_action(A)

        # REWARD
        R = env.get_reward(GS)

        # TRAIN
        if train_me:
            train_networks(GS, A, R)
            
        # END OF LOOP
        if save_me:
            save_path = os.path.join(THING, '{}.png'.format(count))
            Screen.save_image(GS, save_path)
            count += 1
        time.sleep(pause_time)


### TRAIN ####

def train_networks(GS, A, R):
    """ """
    # what_input = input('I want some number here I think')
    # load specified data
    gamestate_data = 0
    action_data = 0
    # train model
    env.train_model_network(gamestate_data, action_data)
    # train value
    agent.train_value_network(gamestate_data, action_data)
    # train binary action
    # pass for now
    # train reward network
    if R != 0:
        # train reward network
        pass



def train_reward_network():
    """ """
    neg_nums = [0] + list(range(233, 247)) + list(range(328, 341))
    pos_nums = list(range(385, 393))
    #[file if for file in os.listdir(gamedata_path) if file in list_]


### HELPER ###

def clear_gamestate_data():
    """ clears all images from filepath """
    path = os.path.join(game_path, 'gamedata')
    print(path)
    _ = [os.remove(os.path.join(path, file)) for file in os.listdir(path)
         if '.png' in file]
    


def record_data():
    """ """
    pass




def listen_game_data():
    """ """
    done = False
    while not done:
        pred = reward.get_reward()[0]
        print(list(pred).index(pred.max()))
        time.sleep(0.25)



def record_game_data():
    """ """
    # params
    count = 0
    print('Start count: {}'.format(count))
    done = False
    while not done:
        c = str(count)
        if count < 10:
            text = '00' + c
        elif count < 100:
            text = '0' + c
        else:
            text = c
        path = 'image_{}'.format(text)
        env.screencap_window(name=path)
        time.sleep(0.1)
        print(count)
        count += 1
        done = count > 500



### PARAMS ###



pause_time = 1

game_name = 'pacman'
game_path = os.path.join(paths.games_path, game_name)
gamedata_path = os.path.join(paths.games_path, 'gamedata')



auto_network = NETS.load_auto(paths.network_path, 'AUTO_test_512_512_6_256')

# RL
env = RL_Environment.Environment(game_path, game_name, auto_network)
player = RL_Agent.Agent(game_path, game_name)

reward = RL_Reward.Reward(env, player)
#ds, ls, idxs = reward.files_to_labeled()
reward.train_network()
#listen_game_data()

#-10:0-9 204-217 492-500
#1:305-312

# REWARD is agent and environment specific
# ACTIONS is agent (?) and environment specific


# SET ENV INDO



### PROGRAM ###

#run_program(train_me=False, save_me=False)

#clear_gamestate_data()
#record_game_data()

#


