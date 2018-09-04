import os
from os.path import join


# top
base_path = os.path.dirname(os.path.realpath(__file__))
data_path = join(base_path, 'data')
games_path = join(base_path, 'games')
network_path = join(data_path, 'networks')

# images
images_path = join(data_path, 'images')
#cards_path = join(data_path, 'cards')
network_path = join(data_path, 'networks')

labels_path = join(images_path, 'labels')

# text
#card_info = join(data_path, 'card_info.txt')
#command_info = join(data_path, 'command_info.txt')
#node_info = join(data_path, 'node_info.txt')

# network

#auto_path = join(network_path, 'auto')
#binary_path = join(network_path, 'binary')
#class_path = join(network_path, 'classify')
#where_path = join(network_path, 'where')

#@games_path = join()

adwars2_labels = os.path.join(labels_path, 'adwars2')
pacman_labels = os.path.join(games_path, 'pacman', 'gamedata')

# GBA
#gba_path = join(game_path, 'GBA')
#gba_games_path = join(gba_path, 'games')
#gba_emulator_path = join(gba_Path, 'mGBA', 'mGBA.exe')

