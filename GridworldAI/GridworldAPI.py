import pygame

import colors

# define which gridworld
from gridworlds import gridworld_1 as GD1
from gridworlds import gridworld_2 as GD2
from gridworlds import gridworld_3 as GD3
from gridworlds import gridworld_4 as GD4
from gridworlds import gridworld_6 as GD6


num = 6

if num == 1:
    env = GD1.Gridworld_1(5)
elif num == 2:
    env = GD2.Gridworld_2(5)
elif num == 3:
    env = GD3.Gridworld_3(4, 12)
elif num == 4:
    env = GD4.Gridworld_4(7, 10)
elif num == 6:
    env = GD6.Gridworld_6(5, 5)


### PARAMS ###

# display size params
MARGIN = 20
HEIGHT = 100
WIDTH = 100
SCREEN_HEIGHT = HEIGHT * env.height + MARGIN * (env.height + 1)
SCREEN_WIDTH = WIDTH * env.width + MARGIN * (env.width + 1)

# other params
GAME_SPEED_MIN = 2
GAME_SPEED_MAX = 60
N_BEFORE_EPISODES = 0

# variable
game_speed = 30
key_check = False


### PROGRAM ###

# initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Gridworld')

# run episodes prior to gridworld
if env.sarsa.episode_counter < N_BEFORE_EPISODES:
    for _ in range(N_BEFORE_EPISODES):
        env.sarsa.first_time_step()
        while not env.in_terminal_state():
             env.sarsa.next_time_step()

# loop until done
done = False
clock = pygame.time.Clock()
while not done:
    
    # --- main pygame event
    for event in pygame.event.get():

        # exit loop of pygame closure
        if event.type == pygame.QUIT:
            done = True
            
        # speed up or slow down auto world
        if event.type == pygame.KEYDOWN and not key_check:
            if event.key == pygame.K_UP:
                game_speed += 1 if game_speed < GAME_SPEED_MAX else 0
            if event.key == pygame.K_DOWN:
                game_speed -= 1 if game_speed > GAME_SPEED_MIN else  0
                
        # user selected action for manual world
        if event.type == pygame.KEYDOWN and key_check:
            if event.key == pygame.K_LEFT:
                sarsa.environment.take_action((0,-1))
            if event.key == pygame.K_RIGHT:
                sarsa.environment.take_action((0,1))
            if event.key == pygame.K_UP:
                sarsa.environment.take_action((-1,0))
            if event.key == pygame.K_DOWN:
                sarsa.environment.take_action((1,0))

    # --- take next SARSA step
    if env.sarsa.end_of_episode:
        env.sarsa.first_time_step()
    elif not key_check:
        env.sarsa.next_time_step()  
    
    # --- update visual
    screen.fill(colors.BLACK)
    for row in range(env.height):
        for col in range(env.width):
            
            # color grid depending on value and state
            color = env.color_grid[row * env.width + col]
            if env.grid[row][col] == 1:
                color = colors.GREEN if env.in_terminal_state() else colors.RED
            
            # draw rectangle at location
            x = (MARGIN + WIDTH) * col + MARGIN
            y = (MARGIN + HEIGHT) * row + MARGIN
            pygame.draw.rect(screen, color, [x, y, WIDTH, WIDTH])

    # --- end pygames loop
    clock.tick(game_speed)
    pygame.display.flip()

pygame.display.quit()
pygame.quit()


