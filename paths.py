import os
from os.path import join


# top
base_path = os.path.dirname(os.path.realpath(__file__))
data_path = join(base_path, 'data')

# images
image_path = join(data_path, 'images')
#cards_path = join(data_path, 'cards')
network_path = join(data_path, 'networks')

# text
#card_info = join(data_path, 'card_info.txt')
#command_info = join(data_path, 'command_info.txt')
#node_info = join(data_path, 'node_info.txt')

# network

#auto_path = join(network_path, 'auto')
#binary_path = join(network_path, 'binary')
#class_path = join(network_path, 'classify')
#where_path = join(network_path, 'where')


