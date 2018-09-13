import os
from os.path import join


# top level
base_path = os.path.dirname(os.path.realpath(__file__))
data_path = join(base_path, 'data')

# data types
games_path = join(data_path, 'games')
images_path = join(data_path, 'images')
network_path = join(data_path, 'networks')

# games
pacman_path = join(games_path, 'pacman')
pacman_gamedata_path = join(pacman_path, 'gamedata')

# images

# networks
hs_data_path = join(data_path, 'cards')
