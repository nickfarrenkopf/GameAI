import paths

from learning import EmulatorEnvironment as EE


import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Computer import Keyboard

from threading import Thread

#t = Thread(target=Keyboard.start_constant_listener, args=())
#t.start()



#env = EE.Emulator(paths, 'test')
#env.save_state_image()

import time


key = Keyboard.get_key()

#for i in range(10):
#    print(Keyboard.get_pressed())
#    time.sleep(0.1)
