import time

import paths
from learning import EmulatorEnvironment as EE
from learning import EmulatorAgent as EAG
from learning import EmulatorAction as EAC
from learning import EmulatorReward as ER

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS


### DATA GENERATION ###

def test_class_network():
    """ """
    last_pred = -1
    for i in range(2000):
        data = auto_network.get_latent(env.get_window(network=True))
        preds = class_network.get_preds(data)[0]
        # print if different
        if last_pred != preds:
            last_pred = preds
            print('Iter {} pred {}'.format(i, preds))
        #time.sleep(0.2)

def test_pred_action():
    """ """
    for i in range(2000):
        data = env.get_state()
        act1 = agent.method.actor_network.get_preds(data)
        act2 = agent.method.actor_TARGET.get_preds(data)
        action = agent.actions[act2[0]]
        print('Iter {} action {}'.format(i, action))
        if ',' in action:
            EAc.take_actions(action.split(','))
        else:
            EAc.take_action(action)
        #time.sleep(0.2)
        

### PROGRAM ###

if __name__ == '__main__':

    """ GAME """
    name = 'pacman'
    paths.set_base(name)

    """ NETWORKS """
    if 1: # AUTO
        auto_network = AUTO.load(name, paths.load_json())
    if 1: # CLASS
        class_network = CLASS.load(name, paths.load_json())

    """ RL """
    
    if 1: # ENV
        env = EE.Emulator(paths, name)
        #env.save_window()
    
    if 1: # AGENT
        agent = EAG.EmulatorAgent(paths, env, 0, EAC.sets['emulator_2'], True)
    
    if 1: # REWARD
        reward = ER.EmulatorReward(paths, env, 'hello')
        reward.load_rewards()
        if 0:
            for i in range(200):
                reward.get_reward()
                time.sleep(0.2)
    

    #test_class_network()
    #test_pred_action()


