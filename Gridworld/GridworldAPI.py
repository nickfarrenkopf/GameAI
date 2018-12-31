import pygame

import paths
import GW1
import GW2
import GW3
import GW4
import GW5


### PYGAME ###

def run():
    """ initiate Gridworld with reinforcement lerning """
    initialize_pygame()
    done = False
    while not done:
        done = check_pygame_events()
        if not manual_mode:
            gridworld.iterate()
        draw_screen()
        end_pygame_loop()
    exit_pygame()

def initialize_pygame():
    """ run pygame.init(), build screen, start clock """
    global screen, clock
    pygame.init()
    pygame.display.set_caption(gridworld.name)
    h = HEIGHT * gridworld.height + MARGIN * (gridworld.height + 1)
    w = WIDTH * gridworld.width + MARGIN * (gridworld.width + 1)
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

def check_pygame_events():
    """ loop over pygame events, exiting or execute actions """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        check_general(event)
        _ = [check_manual(event) if manual_mode else check_automated(event)]
    return False

def draw_screen():
    """ draw gridworld on screen """
    screen.fill((0,0,0))
    for i in range(gridworld.height):
        for j in range(gridworld.width):
            color = gridworld.color_grid[i * gridworld.width + j]
            x = (MARGIN + WIDTH) * j + MARGIN
            y = (MARGIN + HEIGHT) * i + MARGIN
            pygame.draw.rect(screen, color, [x, y, WIDTH, WIDTH])

def end_pygame_loop():
    """ tick counter and draw screen """
    clock.tick(game_speed)
    pygame.display.flip()

def exit_pygame():
    """ close display """
    pygame.display.quit()
    pygame.quit()
    if save_tabular:
        gridworld.end_learning()


### EVENTS ###

def check_general(event):
    """ swicth between manual and automated, switch gridworld """
    global grdiworld, manual_mode
    if event.type == pygame.KEYDOWN:
        # switch between manual and automation modes
        if event.key == pygame.K_a:
            manual_mode = False
        if event.key == pygame.K_m:
            manual_mode = True
        # change gridworld with number keys
        if event.key == pygame.K_1:
            set_gridworld(1)
        if event.key == pygame.K_2:
            set_gridworld(2)
        if event.key == pygame.K_3:
            set_gridworld(3)
        if event.key == pygame.K_4:
            set_gridworld(4)
        if event.key == pygame.K_5:
            set_gridworld(5)
    
def check_manual(event):
    """ move agent in gridworld using action """
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT: # left arrow -> move left
            gridworld.iterate((0,-1))
        if event.key == pygame.K_RIGHT: # right arrow -> move right
            gridworld.iterate((0,1))
        if event.key == pygame.K_UP: # up arrow -> move up
            gridworld.iterate((-1,0))
        if event.key == pygame.K_DOWN: # down arrow -> move down
            gridworld.iterate((1,0))

def check_automated(event):
    """ speed up or slow down actions """
    global game_speed
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP: # up arrow -> speed up
            game_speed += 1 if game_speed < 60 else 0
        if event.key == pygame.K_DOWN: # down arrow -> speed down
            game_speed -= 1 if game_speed > 2 else  0


### GRIDWORLD ###

def set_gridworld(num):
    """ define gridworld by version number """
    global gridworld
    # if gridworld already exists
    redisplay = False
    if gridworld:
        redisplay = True
        exit_pygame()
    # reset gridworld
    gridworld = get_gridworld(num)
    gridworld.set_training_params(run_train, run_pred, train_start, with_decay,
                                  with_random)
    gridworld.initialize()
    if load_initial:
        gridworld.load_value_data()
    # reinitialize pygame on gridworld previously existing
    if redisplay:
        initialize_pygame()

def get_gridworld(num):
    """ """
    if num == 1:
        gw = GW1.Gridworld_1(paths)
    elif num == 2:
        gw = GW2.Gridworld_2(paths)
    elif num == 3:
        gw = GW3.Gridworld_3(paths)
    elif num == 4:
        gw = GW4.Gridworld_4(paths) 
    elif num == 5:
        gw = GW5.Gridworld_5(paths)
    return gw


### PARAMS ###

# pygame
gridworld = None
screen = None
clock = None
game_speed = 3

# screen
HEIGHT = 100
WIDTH = 100
MARGIN = 20

# sarsa tabular
with_random = 1
with_decay = 1
load_initial = 0
save_tabular = 0

# sarsa network
train_start = 0
run_train = 0
run_pred = 0
save_network = 0

# gridworld
manual_mode = 0


### PROGRAM ###

"""

TEST CASES

(Method) SarsaTabular
 - fresh start
 - with Q data
 
(Method) SarsaNetwork
 - saved network data with Q data
 - saved network data without Q data
 - training converges (opt)

"""

if __name__ == '__main__':
    

    # initialize gridworld
    set_gridworld(1)
    if manual_mode:
        gridworld.iterate()

    # run previous episodes
    if 0:
        n_episodes = 100
        print('Running {} episodes...'.format(n_episodes))
        gridworld.iterate_episodes(n_episodes=n_episodes)

    # run gridworld
    if 1:
        print('Running program...')
        run()

    # save trained network
    if run_train and save_network:
        gridworld.method.sarsa_network.save_network()


