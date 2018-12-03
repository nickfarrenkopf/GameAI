import pygame

import GW1
import GW2
import GW3
import GW4
import GW5

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors


### API ###

def run(margin=20, height=100, width=100):
    """ initiate Gridworld with lerning """
    initialize_pygame(width, height, margin)
    done = False
    while not done:
        done = check_pygame_events()
        gridworld.run_learning(listening_to_keys)
        draw_screen(width, height, margin)
        end_pygame_loop()
    exit_pygame()

def initialize_pygame(width, height, margin):
    """ run pygame.init(), build screen, start clock """
    global screen, clock
    pygame.init()
    pygame.display.set_caption('Gridworld')
    h = height * gridworld.height + margin * (gridworld.height + 1)
    w = width * gridworld.width + margin * (gridworld.width + 1)
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

def check_pygame_events():
    """ loop over pygame events, exiting or execute actions """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if listening_to_keys:
            check_manual_events(event)
        else:
            check_automated_events(event)
    return False

def end_pygame_loop():
    """ tick counter and draw screen """
    clock.tick(game_speed)
    pygame.display.flip()

def exit_pygame():
    """ close display """
    pygame.display.quit()
    pygame.quit()


### ACTIONS ###
    
def check_manual_events(event, sarsa):
    """ move with arrow keys ? fix how I do this """
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT: # left arrow -> move left
            gridworld.take_action((0,-1))
        if event.key == pygame.K_RIGHT: # right arrow -> move right
            gridworld.take_action((0,1))
        if event.key == pygame.K_UP: # up arrow -> move up
            gridworld.take_action((-1,0))
        if event.key == pygame.K_DOWN: # down arrow -> move down
            gridworld.take_action((1,0))

def check_automated_events(event):
    """ speed up or slow down actions """
    global game_speed
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP: # up arrow -> speed up
            game_speed += 1 if game_speed < 60 else 0
        if event.key == pygame.K_DOWN: # down arrow -> speed down
            game_speed -= 1 if game_speed > 2 else  0


### HELPER ###

def set_gridworld(num):
    """ define gridworld by version number """
    global gridworld
    if num == 1:
        gridworld = GW1.Gridworld_1(5)
    elif num == 2:
        gridworld = GW2.Gridworld_2(5)
    elif num == 3:
        gridworld = GW3.Gridworld_3(4, 12)
    elif num == 4:
        gridworld = GW4.Gridworld_4(7, 10)
    elif num == 5:
        gridworld = GW5.Gridworld_5(5, 5)

def draw_screen(width, height, margin):
    """ draw gridworld on screen """
    screen.fill(Colors.BLACK)
    for i in range(gridworld.height):
        for j in range(gridworld.width):
            color = gridworld.color_grid[i * gridworld.width + j]
            x = (margin + width) * j + margin
            y = (margin + height) * i + margin
            pygame.draw.rect(screen, color, [x, y, width, width])


### PARAMS ###

# pygame
gridworld = ''
screen = ''
clock = ''
game_speed = 30
listening_to_keys = False


### PROGRAM ###

# stat GridworldAPI
set_gridworld(5)
run()


