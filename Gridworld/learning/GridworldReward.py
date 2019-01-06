import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Learning import RewardUtils as RU


### GRIDWORLD REWARD ###

class GridworldReward(RU.Reward):
    """ """

    def __init__(self, name, value):
        """ """
        RU.Reward.__init__(self, name, value)


### HELPER ###

def load(names):
    """ """
    return RU.RewardList([GridworldReward(n, rewardDict[n]) for n in names])


### REWARDS ###

# reward dictionaries
rewardDict = dict({'win': 1, 'lose': -10, 'megalose': -100, 'pass': -1,
                   'none': 0})
rewardDictRev = {v: k for k, v in keyDict.items()}

# common sets
rewardset_1 = load(['win', 'pass'])
rewardset_2 = load(['megawin', 'win', 'pass'])
rewardset_3 = load(['win', 'pass', 'megalose'])


