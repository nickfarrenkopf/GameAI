import os
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
from pynput import mouse, keyboard

import Library.LibraryGlobals as LG


### PARAMS ###

# global listeners
mx, my = 0, 0
mouseAPI = mouse.Controller()
keyAPI = keyboard.Controller()


### SCREEN ###

def get_data():
    """ takes screenshot and returns normalized data """
    return np.array(ImageGrab.grab()) / 255

def get_data_resized(width, height):
    """ """
    img = ImageGrab.grab()
    out = np.array(img.resize((height, width), Image.ANTIALIAS)) / 255
    save_image(out, 'full_screen.png')
    return out

def get_fullscreen():
    """ """
    sd = get_data_resized(270, 480)
    sd = np.pad(sd, ((25,25), (0,0), (0,0)), mode='constant',
                constant_values=0)
    return sd

def subimage(data, x, y, x_size, y_size, pad_value=0):
    """ """
    width, height = data.shape[1], data.shape[0]
    x_min, x_max, x_left, x_right = stats_pad(x, width, x_size)
    y_min, y_max, y_left, y_right = stats_pad(y, height, y_size)
    data_sub = data[y_min:y_max, x_min:x_max, :]
    datas = np.pad(data_sub, ((y_left, y_right), (x_left, x_right), (0, 0)),
                   mode='constant', constant_values=pad_value)
    return datas

def save_image(data, save_path):
    """ saves data as image to path, un-normalizing if necessary """
    data = data * 255 if data.max() <= 1.0 else data
    image = Image.fromarray(data.astype('uint8'))
    image.save(save_path)


### INTERACTION ###

def move_to(x, y):
    """ """
    mouseAPI.position = (x, y)

def click(x, y, n_click=1):
    """ """
    move_to(x, y)
    mouseAPI.click(mouse.Button.left, n_click)

def send_keys(text):
    """ """
    keyAPI.type(text)


### MOUSE ###

def on_click(x, y, button, pressed):
    """ pynput mouse on-click function """
    global mx, my
    mx, my = x, y
    if not pressed:
        return False

def on_release(key):
    """ pynput keyboard on-release function """
    global mx, my
    if key == keyboard.Key.space:
        mx, my = mouseAPI.position
        return False

def get_click(use_mouse=True):
    """ start mouse listener and return location of next click """
    global mx, my
    if use_mouse:
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()
    else:
        with keyboard.Listener(on_release=on_release) as listener:
            listener.join()
    return mx, my


### HELPER ###

def stats_pad(z, z_max, size, z_min=0):
    """ stats for numbers given padding desired FIX ME """
    zz_min = max(z - size // 2, z_min)
    zz_max = min(z + size // 2, z_max)
    pad_left = max(size // 2 - z, 0)
    pad_right = max(size // 2 - (z_max - z), 0)
    return zz_min, zz_max, pad_left, pad_right



### LISTENER ###

def listen(c_max=100):
    """ """
    global data_i, datas
    count = 0
    while count < c_max:
        # data
        #data_new = get_data()
        #diff = np.mean(np.abs(data_new - data_i))
        #diffs = np.mean([np.abs(data_new - d) for d in datas])
        #new_y = np.ones(x1.shape) * diff
        #new_v = np.ones(x1.shape) * diffs
        # 
        #update_plot(new_y, new_v)
        #print('{:.5f} {:.5f}'.format(diff, diffs))
        # increment
        #data_i = data_new
        #datas.append(data_new)
        #datas = datas if len(datas) < 8 else datas[1:]
        count += 1

def plot_data():
    """ """
    pass
    

def update_plot(new_y, new_v):
    """ """
    plt.cla()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot(x1, new_y)
    ax.plot(x1, new_v)
    plt.pause(0.05)
    plt.show()

def update_img_plot():
    """ """
    plt.clear()
    plt.imshow(mid_data)
    plt.show()


def shade_image(im_path, n_listen=200):
    """ """
    # set up window
    LG.img = np.array(Image.open(im_path))
    LG.counter, LG.drawing = 0, False
    cv2.namedWindow(im_path)
    cv2.setMouseCallback(im_path, draw_circle)
    # wait for image draw
    while LG.counter < n_listen:
        cv2.imshow(im_path, LG.img)
        k = cv2.waitKey(1)
        LG.counter += 1
    cv2.destroyAllWindows()
    save_image(LG.img, im_path.replace('.', '_shaded.'))

def draw_circle(event, x, y, flags, param):
    """ """
    # start or stop drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        LG.drawing = True
    if event == cv2.EVENT_LBUTTONUP:
        LG.drawing = False
    # shade image
    if event == cv2.EVENT_MOUSEMOVE:
        if LG.drawing == True:
            LG.counter = 0
            cv2.circle(LG.img, (x, y), shade_radius, shade_color, -1)


shade_radius = 5
shade_color = (0, 0, 255)

if __name__ == '__main__':

    # location
    image_path = 'C:\\Users\\Nick\\Desktop\\Ava\\Programs\\HearthstoneAI\\data\\images'

    # draw params

    shade_radius = 5
    shade_color = (0, 0, 255)

    # inital plot
    #plt.ion()
    #fig, ax = plt.subplots()
    #ax.set_xlim([0, 1])
    #ax.set_ylim([0, 1])

    # initial data
    #x1 = np.arange(0, 1, 1 / 100)
    #data_i = get_data()
    #datas = [data_i]

    shade_image('2.png')
    #img1 = np.array(Image.open(os.path.join(image_path, '2.png')))
    #img2 = np.array(Image.open(os.path.join(image_path, '2_label.png')))


