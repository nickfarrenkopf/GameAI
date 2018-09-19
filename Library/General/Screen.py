import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
from pynput import mouse, keyboard
from pynput.keyboard import Key as K


### PARAMS ###

# global listeners
mx, my = 0, 0
mouseAPI = mouse.Controller()

mkey = ''
keyAPI = keyboard.Controller()
key_dict = {'left': K.left, 'right': K.right, 'up': K.up, 'down': K.down,
            'enter': K.enter, 'space': K.space}


### SCREEN ###

def get_data():
    """ returns a normalized screenshot """
    return np.array(ImageGrab.grab()) / 255

def get_data_box(x1, y1, x2, y2):
    """ returns a normalized subsection of screen given box """
    return get_data()[x1: x2, y1: y2, :]
    
def get_data_coord(x, y, width, height):
    """ returns a normalized subsection of screen given coordinates """
    return get_data_box(x-width//2, y-height//2, x+width//2, y+height//2)

def get_data_resized(width, height):
    """ returns a normalized screenshot of specificed size """
    img = ImageGrab.grab()
    return np.array(img.resize((height, width), Image.ANTIALIAS)) / 255

def save_image(data, save_path):
    """ saves data as image to path, un-normalizing if necessary """
    data = data * 255 if data.max() <= 1.0 else data
    image = Image.fromarray(data.astype('uint8'))
    image.save(save_path)


### MOUSE ACTION ###

def move_to(x, y):
    """ move key to coordinates """
    mouseAPI.position = (x, y)

def click(x, y, n_click=1):
    """ move key to coordinates and left click """
    move_to(x, y)
    mouseAPI.click(mouse.Button.left, n_click)

def right_click(x, y, n_click=1):
    """ move key to coordinates and right click """
    move_to(x, y)
    mouseAPI.click(mouse.Button.right, n_click)


### KEY ACTION ###

def send_keys(pressed):
    """ keyboard click keys depending on sent keys """
    if pressed in key_dict:
        keyAPI.press(key_dict[pressed])
        keyAPI.release(key_dict[pressed])
    else:
        keyAPI.type(pressed)


### MOUSE LISTEN ###

def on_click(x, y, button, pressed):
    """ pynput mouse on-click function """
    global mx, my
    mx, my = x, y
    if not pressed:
        return False

def get_click(message=None):
    """ start mouse listener and return location of next click """
    global mx, my
    if message:
        print(message)
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    return mx, my


### KEY LISTEN ###

def on_key(key):
    """ pynput keyboard on-release function """
    global mkey
    try:
        mkey = key.char
    except AttributeError:
        mkey = key
    return False

def get_key(message=None):
    """ start mouse listener and return next typed key """
    global mkey
    if message:
        print(message)
    with keyboard.Listener(on_press=on_key) as listener:
        listener.join()
    return mkey


