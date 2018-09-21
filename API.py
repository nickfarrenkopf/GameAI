import os
import time
from pynput.keyboard import Key

import paths

from Library.General import Screen
from Library.General import DataThings as dt
from Library.RL import Game as GG
from Library.RL import Agent as AA
from Library.RL import Environment as EE
from Library.RL import Reward as RR


### API ###

def run_program(train_me=False, save_me=True):
    """ """

    # clean previous game data
    count = 0

    # loop until done
    done = False
    while not done:

        # RL EVENTS
        GS = env.get_gamestate()
        A = player.choose_action(GS)
        player.take_action(A)
        R = env.get_reward(GS)

        # TRAIN
        if train_me:
            train_networks(GS, A, R)
            
        # END OF LOOP
        if save_me:
            save_path = os.path.join(THING, '{}.png'.format(count))
            Screen.save_image(GS, save_path)
            count += 1
        time.sleep(0.5)


### TRAIN ####

def train_networks(GS, A, R):
    """ """
    print('Training? ')
    gamestate_data = 0
    action_data = 0
    if R != 0:
        pass


### HELPER ###

def listen_game_data():
    """ """
    done = False
    while not done:
        pred = reward.get_reward()[0]
        print(list(pred).index(pred.max()))
        time.sleep(0.25)

def record_game_data():
    """ """
    # starting file counter
    count = 0
    if len(os.listdir(game.state_path)) > 0:
        count = int(os.listdir(game.state_path)[-1].split('_')[1]) + 1
    print('Start count: {}'.format(count))
    # wait for key
    done = False
    while not done:
        key = Screen.get_key()
        print(key)
        # if in whitelist, take screencap
        if key in game.action_keys or key in game.label_keys:
            text = '0' * (4 - len(str(count))) + str(count)
            names = ['image_{}_{}_0'.format(text, values[keys.index(key)])]
            windows = [game.get_window()]
            if key in game.action_keys:         
                for i in range(1, 3):
                    text = '0' * (4 - len(str(count))) + str(count)
                    names.append('image_{}_{}_{}'.format(text,
                                                         values[keys.index(key)],
                                                         i))
                    windows.append(game.get_window())
                    time.sleep(0.1)
            for name, window in zip(names, windows):
                path = os.path.join(game.state_path, name + '.png')
                Screen.save_image(window, path)
            count += 1
        done = key == 'p'


### PARAMS ###

# location
game_path = paths.pacman_path

# network
auto_network = dt.load_auto(paths.network_path, 'AUTO_test_512_512_6_256')

# key mappings
keys = [Key.up,Key.right,Key.down,Key.left,'z','x','a','s','q','r','w',
        Key.enter,Key.space]
values = ['up','right','down','left','z','x','a','s','q','r','w',
          'enter','space']


### PROGRAM ###

if True:
    
    game = GG.Game(game_path, auto_network)
    env = EE.Environment(game)
    agent = AA.Agent(game, env)
    reward = RR.Reward(game, agent)


# train
#reward.train_network_offline()


#idxs, labels = game.load_labels('left')

#idxs, labels = game.find_game_data(game.reward_labels)

#record_game_data()
#listen_game_data()


