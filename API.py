
from PIL import Image



from Library import Screen





### ###

def find_region():
    """ """
    global rect
    # click region of deisred input
    x1, y1 = Screen.get_click('Click top left')
    #print(x1, y1)
    x2, y2 = Screen.get_click('Click bottom right')
    #print(x2, y2)
    # parse image
    dats = Screen.get_data()[y1:y2, x1:x2, :]
    Screen.save_image(dats, 'test.png')    
    rect = ((x1, y1), (x2, y2))
    #
    print('x coords: {} {}'.format(x1, x2))
    print('y coords: {} {}'.format(y1, y2))
    print('x y size: {} {}'.format(x2 - x1, y2 - y1))
    



### PARAMS ###

rect = ((), ())



### PROGRAM ###

dats = find_region()




