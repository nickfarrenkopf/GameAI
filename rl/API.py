import os
import time

import paths
#import Screen


import agent
import environment


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



### HELPER ###

def clear_gamestate_data(game):
    """ clears all images from filepath """
    filepaths = os.path.join(game_path, 'gamedata')
    _ = [os.remove(file) for file in filepaths if '.png' in file]




### PARAMS ###



pause_time = 1

game_name = 'pacman'
game_path = os.path.join(paths.games_path, game_name)

# RL
env = environment.Environments(game_path, game_name)
player = agent.Agent(game_path, game_name)



# REWARD is agent and environment specific
# ACTIONS is agent (?) and environment specific


# SET ENV INDO



### PROGRAM ###

#run_program(train_me=False, save_me=False)


