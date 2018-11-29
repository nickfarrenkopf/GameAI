import colors
from gridworlds import Gridworld
from sarsa_tabular import Sarsa_Tabular


class Piece(object):
    """ object found in gridworld """

    def __init__(self, name, index, number, color):
        """ creates piece with specified name, index, number, and color """
        self.name = name
        self.index = index
        self.number = number
        self.color = color


class Gridworld_5(Gridworld.Gridworld):
    """ gridworld example 5 """

    damage = 0
    current_score = 0
    score_max = 100
    number_monster = 4
    number_repair = 1
    number_prize = 1


    def __init__(self, height, width):
        """ creates base gridworld and defines actions """
        Gridworld.__init__(self, height, width)
        self.starting_place = self.size // 2
        self.terminal_states = self.corners()
        #self.pieces = self.define_pieces()
        self.color_grid = self.define_color_grid()
        # self.state = list(self.board) + list(self.damage)


    ### METHODS ###

    def define_pieces(self):
        """ defines and returns pieces found in gridworld """
        p0 = Piece('Player', index = 1, number = 1, color = self.RED)
        p1 = Piece('Monster', index = 2, number = 4, color = self.BLUE)
        p2 = Piece('Repair', index = 3, number = 1, color = self.MAGENTA)
        p3 = Piece('Prize', index = 4, number = 1, color = self.GOLD)
        return [p0, p1, p2, p3]


    ### OVERRIDE METHODS ###

    def define_method(self):
        """ defines and returns RL solution method """
        sarsa = Sarsa_Tabular(self)
        sarsa.set_parameters()
        return sarsa

    def define_color_grid(self):
        """ everything is white """
        color_grid = [self.WHITE for index in self.board]
        return color_grid

    def initialize_state(self):
        """ places player/prize at locations, randomizes monster/repair """
        self.board[self.starting_place] = 1
        for _ in range(self.number_monster):
            self.random_nonterminal_state(2)
        for _ in range(self.number_repair):
            self.random_nonterminal_state(3)
        for _ in range(self.number_prize):
            self.random_terminal_state(4)  
        self.damage = 0
        self.current_score = 0
        self.color_grid = self.define_color_grid()

    def take_action(self, action):
        """ return next state and reward given action """
        # find where moving
        grid_coord = self.location_of(1, grid=True)
        next_grid_coord = np.array(grid_coord) + np.array(action)
        if not self.on_grid(next_grid_coord):
            return self.board, -1
        next_state = self.to_board(next_grid_coord)
        reward = -1
        # if monster
        if self.board[next_state] == 2:
            self.damage += 1
            self.move_piece_direction(1, action)
            self.random_nonterminal_state(2)
            print(2)
        # if repair
        elif self.board[next_state] == 3:
            self.damage -= 1 if self.damage > 0 else 0
            self.move_piece_direction(1, action)
            self.random_nonterminal_state(3)
            print(3)
        # if prize
        elif self.board[next_state] == 4:
            self.current_score += 10
            self.reward = 10
            self.move_piece_direction(1, action)
            self.random_terminal_state(4)
            print(4)
        # else
        elif self.on_grid(next_grid_coord):
            self.move_piece_direction(1, action)
        # reward
        reward = -100 if self.damage > 1 else reward
        self.color_grid = self.define_color_grid()
        return self.board, reward

    def in_terminal_state(self):
        """ returns true if grid in terminal state """
        return self.current_score >= self.score_max or self.damage > 1


