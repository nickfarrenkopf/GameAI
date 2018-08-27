import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from PIL import Image

import paths

from Library import Screen
from Library import DataThings as DT
import data_classes as DC


from Library.DataThings import *


### TEXT FILE ###

def load_data():
    """ """
    path = paths.adwars2_labels
    images = np.array([np.array(Image.open(os.path.join(path, f))) / 255
              for f in os.listdir(path)])
    #print(images.shape)
    return images

def load_card_info():
    """ load card text info """
    text_data = DT.read_file_csv(paths.card_info)
    cards = sorted([row[0] for row in text_data])
    card_dict = dict((c, DC.Card(*r)) for c, r in zip(cards, text_data))
    return card_dict, cards

def load_command_info():
    """ load command text info """
    text_data = DT.read_file_split_empty(paths.command_info)
    commands = [DC.Command(lines) for lines in text_data]
    command_dict = dict((cmd.text, cmd) for cmd in commands)
    return command_dict, list(command_dict.keys())

def load_node_info(auto_network):
    """ """
    text_data = DT.read_file_split_empty(paths.node_info)
    nodes = [DC.DataNode(auto_network, lines) for lines in text_data]
    node_dict = dict((node.name, node) for node in nodes)
    return node_dict, list(sorted(node_dict.keys()))


### NETWORK FILE ###

def load_auto(auto_name):
    """ """
    return NETS.load_auto(paths.auto_path, auto_name)

ss2 = 256
def load_class(auto_net, name):
    """ """
    # check no duplicate networks
    nets = [f for f in os.listdir(paths.class_path) if '_{}_'.format(name) in f
            and '.meta' in f]
    if len(nets) != 1:
        print('issue?')
    # check if full name or partial name
    if name.startswith('CLASS'):
        n_classes = int(name[name.rindex('_'):])
        class_name = name
    else:
        n_classes = nets[0][:nets[0].index('.')].split('_')[-1]
        class_name = 'CLASS_{}_{}_2_{}'.format(name, ss2, n_classes)
    return NETS.load_class(paths.class_path, class_name, auto_net, n_classes)

def load_binary(auto_net, name):
    """ """
    formatted_name = 'CLASS_{}_{}_1_2'.format(name, ss2)
    binary_name = name if name.startswith('CLASS') else formatted_name
    return NETS.load_class(paths.binary_path, binary_name, auto_net, 2)


### IMAGE FILE ###

a = 395
b = 286
c = 3

def load_images(vocab):
    """ """
    return [Image.open(join(paths.cards_path, v + '.png')) for v in vocab]

#def load_data(vocab):
#    """ """
#    data = [np.array(img)[:a, :b, :c] / 255 for img in load_images(vocab)]
#    labels = DT.new_labels(len(vocab))
#    return np.array(data), labels

def load_scaled_data(vocab, ratio=[1.0]):
    """ """
    sizes = [(int(b * r), int(a * r)) for r in ratio]
    data = [np.array([np.array(img.resize(s, Image.ANTIALIAS))[:, :, :3] / 255
                      for img in load_images(vocab)]) for s in sizes]
    labels = [np.array([new_label(i, len(vocab)) for i, _ in enumerate(vocab)])
              for s in sizes]
    data = np.array(data[0] if len(data) == 1 else data)
    return data, np.array(labels)


### PROCESS DATA ###

def filter_data(data_dict, vocab, c_type=None, c_class=None, cost=None,
                attack=None, health=None):
    """ """
    new_vocab = [v for v in vocab
                 if  (not c_type or data_dict[v].type_ in c_type)
                 and (not c_class or data_dict[v].class_ in c_class)
                 and (not cost or data_dict[v].cost in cost)
                 and (not attack or data_dict[v].attack in attack)
                 and (not health  or data_dict[v].health in health)]
    return new_vocab

def to_class_data(card_dict, vocab, classes, ratio=[1.0]):
    """ """
    data, labels = [], []
    for i, c in enumerate(classes):
        sub_vocab = filter_data(card_dict, vocab, c_class=c)
        sub_data, _ = load_scaled_data(sub_vocab, ratio=ratio)
        sub_labels = [new_label(i, len(classes)) for _ in range(len(sub_data))]
        data += list(sub_data)
        labels += sub_labels
    return np.array(data), np.array(labels)

def to_type_data(card_dict, vocab, types, ratio=[1.0]):
    """ """
    data, labels = [], []
    for i, t in enumerate(types):
        sub_vocab = filter_data(card_dict, vocab, c_type=t)
        sub_data, _ = load_scaled_data(sub_vocab, ratio=ratio)
        sub_labels = [new_label(i, len(types)) for _ in range(len(sub_data))]
        data += list(sub_data)
        labels += sub_labels
    return np.array(data), np.array(labels)

def to_ocr_data(card_dict, vocab, numbers, ratio=[1.0], s=128):
    """ """
    combos = ((1, 0, 0, 40, 50), (0, 1, 0, 45, 362), (0, 0, 1, 247, 362))
    data, labels = [], []
    last_size = 0
    for i in numbers:
        for c, a, h, x, y in combos:
            _, sub_vocab = filter_data(card_dict, vocab,
                                       cost=i if c else None,
                                       attack=i if a else None,
                                       health=i if h else None,
                                       c_type='minionweapon' if a or h else None)
            sub_imgs, _ = load_data(sub_vocab)
            sub_data = [Screen.subimage(img, x, y, s, s) for img in sub_imgs]
            sub_labels = [new_label(int(i), len(numbers)) for _ in sub_data]
            data += list(sub_data)
            labels += sub_labels
        print('OCR: {}  Number: {}'.format(i, len(labels) - last_size))
        last_size = len(labels)
    return data, np.array(labels)



### HELPER ###

def record_click():
    """ """
    print('Waiting for click...')
    x, y = Screen.get_click()
    text = input('Image save name:')
    subimage = Screen.subimage(Screen.get_data(), x, y, 128, 128)
    save_path = os.path.join(paths.image_path, '{}_{}.png'.format(text, 128))
    Screen.save_image(subimage, save_path)
    subimage = Screen.subimage(Screen.get_data(), x, y, 256, 256)
    save_path = os.path.join(paths.image_path, '{}_{}.png'.format(text, 256))
    Screen.save_image(subimage, save_path)

