import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


### TEXT FILE ###

def read_file(filename):
    """ return lines from text file """
    with open(filename, 'r') as file:
        return file.read().split('\n')

def read_file_csv(filename, split_value=','):
    """ return lines from text file splut by comma """
    return [row.split(split_value) for row in read_file(filename)]

def read_file_split_empty(filename):
    """ """
    lines = [''] + read_file(fielname)
    idxs = [i for i, l in enumerate(lines) if l == ''] + [len(lines)]
    data = [lines[idx + 1:idxs[i+1]] for i, idx in enumerate(idxs[:-1])]
    return data


### IMAGE FILE ###

def load_image(file):
    """ returns normalized data for image file """
    return np.array(Image.open(file)) / 255

def load_images(files):
    """ returns normalized data for images files """
    return np.array([load_image(file) for file in files])


### NETWORK FILE ###

def load_keras_network(filename):
    """ """
    return load_model(filename)


### DATA ###

def pad_me(data, pad1, pad2):
    """ """
    return np.pad(data, ((0, 0), (pad1, pad1), (pad2, pad2), (0, 0)),
                  mode='constant', constant_values=0)

def subdata(data, height, width):
    """ """
    i = np.random.randint(data.shape[1] - height)
    j = np.random.randint(data.shape[2] - width)
    return data[:, i:i + height, j:j + width, :]

def subdata_xy(data, height, width, x, y):
    """ """
    return data[:, x-height//2:x+height//2, y-width//2:y+width//2, :]


### LABELS ###

def new_label(idx, n_classes):
    """ """
    label = np.zeros(n_classes)
    label[idx] = 1
    return label

def to_one_hot(labels, n_classes=0):
    """ """
    label_set = list(sorted(set(labels)))
    n_classes = len(label_set) if n_classes == 0 else n_classes
    one_hot = [new_label(label_set.index(lab), n_classes) for lab in labels]
    return one_hot


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


