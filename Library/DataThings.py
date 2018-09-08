import numpy as np
import matplotlib.pyplot as plt


### FILE ###

def read_file_csv(filename):
    """ """
    with open(filename, 'r') as file:
        return [row.split(',') for row in file.read().split('\n')]

def read_file(filename):
    """ """
    with open(filename, 'r') as file:
        return file.read().split('\n')

def to_one_hot_labels(labels):
    """ """
    label_set = list(sorted(set(labels)))
    one_hot = np.array([new_label(label_set.index(label), len(label_set))
                            for label in labels])
    return one_hot
    



def read_file_split_empty(filename):
    """ """
    with open(filename, 'r') as file:
        lines = [''] + file.read().split('\n')
    idxs = [i for i, l in enumerate(lines) if l == ''] + [len(lines)]
    data = [lines[idx + 1:idxs[i+1]] for i, idx in enumerate(idxs[:-1])]
    return data


### DATA ###

def pad_me(data, pad1, pad2):
    """ """
    return np.pad(data, ((0, 0), (pad1, pad1), (pad2, pad2), (0, 0)),
                  mode='constant', constant_values=0)

def subdata(data, height, width):
    """ """
    i = np.random.randint(data.shape[1] + 1 - height)
    j = np.random.randint(data.shape[2] + 1 - width)
    return data[:, i:i + height, j:j + width, :]

def subdata_xy(data, height, width, x, y):
    """ """
    return data[:, x-height//2:x+height//2, y-width//2:y+width//2, :]


### LABELS ###

def to_one_hot(old_labels, n_classes=0):
    """ """
    label_set = list(sorted(set(labels)))
    n_classes = len(label_set) if n_classes == 0 else n_classes
    new_labels = [new_label(label_set.index(label), n_classes)
                  for label in old_labels]
    return new_labels


def new_label(i, n_classes):
    """ """
    label = np.zeros(n_classes)
    label[i] = 1
    return label

def new_labels(n_classes):
    """ """
    return np.array([new_label(i, n_classes) for i in range(n_classes)])




### BINARY ###

plus = pos = np.array(((0, 1),))
minus = np.array(((1, 0),))

def to_pos_neg_data(data, labels):
    """ turns regular data set into pbinary data set """
    pos_data = np.array([data[i] for i, la in enumerate(labels) if la[1] == 1])
    pos_labels = np.array([plus for _ in range(len(pos_data))])
    neg_data = np.array([data[i] for i, la in enumerate(labels) if la[0] == 1])
    neg_labels = np.array([minus for _ in range(len(neg_data))])
    return pos_data, pos_labels, neg_data, neg_labels

def to_binary_labels(word, vocab):
    """ returns binary labels given positive label """
    return np.reshape([plus if w == word else minus for w in vocab], (-1,2))


### PLOT ###

def plot_data_multiple(data, labels=None, n_x=4, n_y=6, figure_size=(16, 8),
                       save_path=False):
    """ """
    fig = plt.figure(figsize=figure_size)
    # subplots
    for i in range(min(n_x * n_y, len(data))):
        ax = fig.add_subplot(n_x, n_y, i + 1)
        ax.imshow(data[i])
        ax.set_aspect('equal')
        if labels is not None:
            ax.set_title(labels[i])
        ax.axis('off')
    # other
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    fig.set_tight_layout(False)
    _ = [fig.savefig(save_path) if save_path else fig.show() for i in range(1)]
    return fig


