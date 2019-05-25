import os
import time
import itertools
import numpy as np
from os.path import join, split, basename
from scipy import spatial

import paths
import HearthstoneConstants as HC
import HearthstoneElement as HE

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.Computer import Screen
from Library.Computer import Mouse
from Library.Computer import Keyboard


### LABELER ###

def load_data_labels(name, non_neg=True, neg=False):
    """ """
    ds = FT.load_images(files[:n])
    labs = [os.path.basename(f) for f in files[:n]]
    fp = os.path.join(paths.labels_path, '{}_labels.txt'.format(name))
    txt_labs = FT.read_file_csv(fp)
    ls = [[int(la[1]), int(la[2])] if len(la) == 3 else [-2,-2]
           for la in txt_labs]
    # load file labels
    #
    if non_neg and not neg:
        ds = np.array([d for i, d in enumerate(ds) if -1 not in ls[i]])
        ls = np.array([la for i, la in enumerate(ls) if -1 not in la])
    if neg:
        idxs = [i for i, la in enumerate(ls) if -2 in la]
        ds = np.array([d for i, d in enumerate(ds) if -2 in ls[i]])
        ls = np.array([la for i, la in enumerate(ls) if -2 in la])
        return ds, ls, idxs
    return ds, ls

import time
def label_things(name):
    """ """
    ds, ls, idxs = load_data_labels(name, neg=True)
    new_ls = []
    #labs = [os.path.basename(f) for f in files[:n]]
    for i in range(300):
        FT.save_image_to_file(ds[i], 'label.png')
        time.sleep(0.3)
        k1, _ = Keyboard.get_key_single()
        if k1 == '.':
            k1 = -2
            k2 = -2
        else:
            time.sleep(0.3)
            k2, _ = Keyboard.get_key_single()
            if k1 == 'q':
                k1 = 10
            if k2 == 'q':
                k2 = 10
        new_ls.append([int(k1), int(k2)])
        print('{} {}'.format(os.path.basename(files[idxs[i]]), new_ls[-1]))
    # old labels
    fp = os.path.join(paths.images_path, '{}_labels.txt'.format(name))
    txt_labs = FT.read_file_csv(fp)
    old_ls = [[int(la[1]), int(la[2])] if len(la) == 3 else [-2,-2]
           for la in txt_labs]
    # change labs
    for i, nl in enumerate(new_ls):
        old_ls[idxs[i]] = nl
    # save new
    save_data = ['{},{},{}'.format(la[0], old_ls[i][0], old_ls[i][1])
                 for i, la in enumerate(txt_labs)]
    save_data = '\n'.join(save_data)
    #
    with open(fp, 'w') as f:
        f.write(save_data)



def get_similar(vecs, vectors, top_n=20):
    """ """
    sims = [(i, np.sum([1-spatial.distance.cosine(v1, v2) if
                        1-spatial.distance.cosine(v1, v2) > 0 else 0
                        for v1 in vecs]),
             basename(files[i]))
            for i, v2 in enumerate(vectors)]
    sims.sort(key=lambda x: x[1])
    returns = sims[-top_n:]
    returns = list(reversed(returns))
    return sims[-top_n:]

def load_some_labels(idx=1):
    """ """
    labels = [lab for lab in paths.get_labels(name)
                  if '-2' not in lab[1:] and '-1' not in lab[1:]]
    ds_small = FT.load_images([os.path.join(paths.images_path, name, 
                                    '{}'.format(lab[0]))
                               for lab in labels])
    all_labels = []
    for i in range(11):
        # find random label with correct label
        ind = 0
        for j in range(10):
            while labels[ind][idx] != str(i):
                ind += 1
            # all_labels
            all_labels.append(labels[ind])
            ind += 1
    return all_labels

def norm(ds):
    """ """
    print(ds.max(), ds.min(), np.mean(ds))
    return ds
    return (ds - ds.min()) / (ds.max() - ds.min())

def compare(auto_network, vecs, vectors):
    """ """
    pass

def data_analysis(name, label='2'):
    """ """
    # load small dataset
    labels = [lab for lab in paths.get_labels(name) if label in lab[1:]]
    ds_small = FT.load_images([join(paths.images_path, name, 
                                    '{}'.format(lab[0]))
                               for lab in labels])
    ds = ds_small
    ds_small = ds_small[:3]
    print('Len labels {}: {}'.format(label, len(ds_small)))
    DT.plot_data_multiple(ds_small, save_path='label_{}.png'.format(label))
    # generate mu, sd, and latent for SMALL data set
    ds_mu = np.reshape(norm(auto_network.get_mu(ds_small)), (-1, 2, 2))
    ds_sd = np.reshape(norm(auto_network.get_sigma(ds_small)), (-1, 2, 2))
    ds_la = np.reshape(norm(auto_network.get_latent(ds_small)), (-1, 2, 2))
    print(ds_mu.max(), ds_mu.min(), np.mean(ds_mu))
    print(ds_sd.max(), ds_sd.min(), np.mean(ds_sd))
    print(ds_la.max(), ds_la.min(), np.mean(ds_la))
    print('MM {}\n MS{}\n'.format(np.mean(ds_mu, axis=0), np.std(ds_mu, axis=0)))
    print('SM {}\n SS{}\n'.format(np.mean(ds_sd, axis=0), np.std(ds_sd, axis=0)))
    print('LM {}\n LS{}\n'.format(np.mean(ds_la, axis=0), np.std(ds_la, axis=0))) 
    plot_data = [[ds_small[i],ds_mu[i],ds_sd[i],ds_la[i]] for i in range(len(ds_mu))]
    plot_data = list(itertools.chain.from_iterable(plot_data))
    DT.plot_data_multiple(plot_data, n_x=8, n_y=8, norm=True,
                          save_path='auto_data_{}.png'.format(label))
    data_mu = auto_network.get_mu(ds)
    data_sd = auto_network.get_sigma(ds)
    data_la = auto_network.get_latent(ds)
    # generate auto data
    #auto_data_x = func(ds_small)
    #auto_data_y = func(ds)
    # cluster data
    top_similar = get_similar(np.reshape(ds_mu, (-1, 4)), data_mu, top_n=20)
    for p in top_similar:
        print(p[1:])
    #DT.plot_data_multiple(np.array([p[0]for p in top_similar]),
    #                      save_path='sim_data_{}.png'.format(label))
    a = 1 / 0
    #mam = [ds[i] for i, _, _ in top_similar]
    ots = [[x, y] for x, y in zip(ddads, ress)]
    ots = list(itertools.chain.from_iterable(ots))
    # DT.plot_data_multiple(ots)
    mam_auto = np.reshape(auto_network.get_mu(mam), (-1, 8, 8))
    
    asd = [[x, y] for x, y in zip(mam, mam_auto)]
    asd = list(itertools.chain.from_iterable(asd))
    all_data.append()
    # data to plot
    #plot_data for row in all_data
        
    DT.plot_data_multiple(asd)
    DT.plot_data_multiple(plot_data)





### DATA GENERATION ###

def record_data(idx, n_record=99999, sleep_time=1):
    """ """
    print('Recording Heathstone Element screen data')
    for i in range(n_record):
        for e in elements:
            print(e.name)
            if i % e.every_n == 0:
                e.save_window(idx, i)
        time.sleep(sleep_time)

def delete_similar_images(image_path, n_future, min_diff=0.001):
    """ """
    # loop over files in image path
    files = FT.get_filepaths(image_path)
    for i, f in enumerate(files):
        if i % 200 == 0:
            print('Iteration {} {}'.format(i, split(f)[1]))
        # load data past this file
        subfiles = FT.get_filepaths(image_path)
        idx = subfiles.index(f)
        subfiles = subfiles[idx + 1: idx + n_future]
        # loop over data
        for j, ds in enumerate(FT.load_images(subfiles)):
            diff = np.mean(np.square(np.abs(FT.load_images([file]) - ds)))
            # if same images and different files
            if diff < min_diff and f != subfiles[j]:
                print(split(f)[1], split(subfiles[j])[1])
                os.remove(f)
                break
import cv2
from PIL import Image

def watch_videos():
    """ """
    ffs = ['Hearthstone 5_11_2019 11_28_12 AM.mp4',
           'Hearthstone 5_11_2019 11_34_39 AM.mp4',
           'Hearthstone 5_11_2019 11_47_15 AM.mp4',
           'Hearthstone 5_11_2019 12_19_55 PM.mp4',
           'Hearthstone 5_11_2019 1_14_05 PM.mp4',
           'Hearthstone 5_11_2019 2_12_42 PM']
    for idx, ff in enumerate(ffs):
        file = 'D://Storage//The_Arc//HearthstoneVideos//' + ff
        vidcap = cv2.VideoCapture(file)
        success,image = vidcap.read()
        count = 0
        diff = 0
        overall_count = 0
        last_saved = [[] for _ in elements]
        while success:
            saved_new = False
            for j, e in enumerate(elements):
                if count % (15 * e.every_n) == 0:
                    name = '{}_v{}_{}.jpg'.format(e.name, idx, overall_count)
                    img = image[e.p1[0]:e.p2[0], e.p1[1]:e.p2[1]]
                    img = cv2.resize(img, (e.h_n, e.w_n))
                    if len(last_saved[j]) > 0:
                        diff = np.abs(np.mean(last_saved[j]) - np.mean(img))
                    if len(last_saved[j]) == 0 or diff > 1:
                        cv2.imwrite(name, img)
                        saved_new = True
                        last_saved[j] = np.array(img)
            if saved_new:
                overall_count += 1
            success,image = vidcap.read()
            count += 1

### PROGRAM ###

# Hearthstone Elements
elements = HE.load_elements(HC.elements, networks=False)
#print('Loaded elements {}'.format([e.name for e in elements]))
    
# base folder
name = 'hero'
names = [name + 'S', name + 'O']
filepath = paths.get_image_path(name)
files = FT.get_filepaths(filepath)
n = 5000

if __name__ == '__main__':
    

    """ SCREEN WATCHERS """
    # file data
    
    

    """ DATA """    

    if 0: # IMAGE DATA
        ds, ls = load_data_labels(name, neg=False)
        print('Data shape: {} {} {}'.format(ds.shape, ds.max(), ds.min()))
        print('N labels: {}'.format(len(ls)))

    """ DATA PROCESSING """

    # online data recording
    if 0:
        record_data(8, n_record=999999)

    # clean images
    if 0:
        all_paths = [paths.get_image_path(e) for e in HC.elements]
        for p in all_paths:
            delete_similar_images(p, n_future=20)


    if 1: # LOAD - NETWORK
        elements = HE.load_elements(names, networks=False)
        e = elements[0]
        #auto_network = e.auto_network
        #auto_network.print_info()
    #label_things(name)


    #data_analysis(name)



    watch_videos()

