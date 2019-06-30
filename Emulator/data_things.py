import os
import random
from random import shuffle
from itertools import chain

import paths

def get_filepaths_for_labels(wanted_labels, shuffle_me=True, with_other=False):
    """ """
    filepaths, labels = [], []
    other_files, other_labels = [], []
    filenames = os.listdir(paths.image_path)
    label_set = set(chain.from_iterable([get_labels(f) for f in filenames]))
    # loop over files
    for f in filenames:
        ls = get_labels(f)
        if len(list(label_set & set(ls))) > 0 and any([l in wanted_labels for l in ls]):
            filepaths.append(os.path.join(paths.image_path, f))
            idx = [l in wanted_labels for l in ls]
            labels.append(ls[idx.index(True)])
        else:
            other_files.append(os.path.join(paths.image_path, f))
            other_labels.append('_NA')
    # add files not in class list as other
    if with_other:
        idxs = random.sample(range(len(other_files)), len(filepaths) * 2)
        #idxs = list(range(len(other_files)))
        filepaths += [other_files[i] for i in idxs]
        labels += [other_labels[i] for i in idxs]
    # shuffle order
    if shuffle_me:
        idxs = list(range(len(filepaths)))
        shuffle(idxs)
        filepaths = [filepaths[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    return filepaths, labels

def get_labels(file):
    """ """
    return file.split('.')[0].split('_')[-1].split(',')

def prepend_zeros_idx():
    """ """
    count = 0
    filenames = os.listdir(paths.image_path)
    episodes = [int(f.split('_')[1]) for f in filenames]
    for i in range(len(set(episodes))):
        files = [filenames[j] for j, e in enumerate(episodes) if e == i]
        max_len = len(str(len(files)))
        print('Episode: {}   n_files: {}'.format(i, len(files)))
        for file in files:
            params = file.split('_')
            middle = '0' * (max_len - len(params[2])) + params[2]
            str_name = '_'.join(params[:2] + [middle] + params[3:])
            if str_name != file:
                os.rename(os.path.join(paths.image_path, file),
                          os.path.join(paths.image_path, str_name))
                count += 1
    print('Files renamed {}'.format(count))

