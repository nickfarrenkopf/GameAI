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
aw2_path = join(games_path, 'advanced_wars_2')
mario_path = join(games_path, 'super_mario_advance_4')
centipede_path = join(games_path, 'centipede')
kirby_path = join(games_path, 'kirby')
mariokart_path = join(games_path, 'mariokart')

# images
all_paths = [pacman_path]#, aw2_path, mario_path, centipede_path, kirby_path, mariokart_path]


