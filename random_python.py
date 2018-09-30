
from pynput import mouse
from pynput.keyboard import Listener





def on_function(key):
    """ """
    listener.stop()
    print(key)
    


key_listener = keyboard.Listener(on_release=on_function)
key_listener.start()


def on_click(x, y, button, pressed):
    """ """
    if button == mouse.Button.left:
        print ('Left')
    if button == mouse.Button.right:
        key_listener.stop()
        print ('right')
    if button == mouse.Button.middle:
        print ('middle')

with mouse.Listener(on_click=on_click) as listener:
    try:
        listener.join()
    except MyException as e:
        print('Done'.format(e.args[0]))




