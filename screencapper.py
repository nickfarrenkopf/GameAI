import os
import time
import numpy as np
from PIL import Image, ImageGrab


### HELPER ###

def get_data_resized_xy(x1, y1, x2, y2, width, height):
    """ returns a normalized screenshot of specificed size """
    img = ImageGrab.grab()
    img = img.crop((y1, x1, y2, x2))
    return np.array(img.resize((height, width), Image.ANTIALIAS)) / 255

def save_image(data, save_path):
    """ saves data as image to path, un-normalizing if necessary """
    data = data * 255 if data.max() <= 1.0 else data
    image = Image.fromarray(data.astype('uint8'))
    image.save(save_path)


### PARAMS @@@

base_path = os.path.dirname(os.path.realpath(__file__))
images_path = os.path.join(base_path, 'images')
max_h = 1080
max_w = 1920
sleep_time = 0.5


### PROGRAM ###

for i in range(10):
    save_path = os.path.join(images_path, 'auto_data_{}.png'.format(i))
    data = get_data_resized_xy(0, 0, max_w // 2, max_h, 1024, 1024)
    save_image(data, save_path)
    time.sleep(sleep_time)
