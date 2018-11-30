import sys
import numpy as np


### Q VALUE ###

class Q_value(object):
    """ class for holding state action parir value and trace """
    def __init__(self, value, trace):
        self.value = value
        self.trace = trace


### SARSA CLASS AND METHODS ###

class Sarsa_Tabular(object):
    """ simulates the SARSA tabular reinforcement learning method """

    def __init__(self, environment):
        """ intiializes environment, variables, and constants of sarsa """
        self.environment = environment
        self.Q = dict()
        self.S, self.S_next = None, None
        self.A, self.A_next = None, None
        self.A_all = environment.get_action_profile()
        self.set_parameters()
        self.action_counter = 0
        self.episode_counter = 0
        self.end_of_episode = True


    ### ACCESS ENVIRONMENT ###
 
    def initial_state(self):
        """ resets environment and returns initial state """
        self.environment.reset_state()
        return tuple(self.environment.state)

    def take_action(self, action):
        """ returns next state and reward given action """
        S_next, reward = self.environment.take_action(action)
        return tuple(S_next), reward


    ### OUTSIDE ACCESS ###

    def set_parameters(self, initial_value=1, alpha=0.5, gamma=0.5,
                       lambdas=0.9, epsilon=0.1, epsilon_decay=1,
                       decay_start=50, trace_type='dutch'):
        """ sets parameters for sarsa method """
        self.initial_value = initial_value
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.LAMBDA = lambdas
        self.EPSILON = epsilon
        self.EPSILON_DECAY = epsilon_decay
        self.DECAY_START = decay_start
        self.TRACE_TYPE = trace_type

    def first_time_step(self):
        """ sets beginning of episode by reseting all variables """
        self.S = self.initial_state()
        self.check_dict(self.S)
        self.A = self.choose_action(self.S)
        for key in self.Q.keys():
            self.Q[key].trace = 0
        self.action_counter = 0
        self.episode_counter += 1
        self.end_of_episode = False

    def next_time_step(self):
        """ takes next time step, updating state, action, and values """
        # take action, observe next state and reward, choose next action
        self.S_next, R = self.take_action(self.A)
        self.check_dict(self.S_next)
        self.A_next = self.choose_action(self.S_next)
        # backup values
        self.backup_action(R)
        # end of time step
        self.S, self.A = self.S_next, self.A_next
        if self.episode_counter > self.DECAY_START:
            self.EPSILON *= self.EPSILON_DECAY
        self.action_counter += 1
        self.end_of_episode = self.environment.in_terminal_state()


     ### SARSA CALC ###

    def backup_action(self, R):
        """ backs up state action pair value and trace """
        # state action pairs
        SA = self.pair(self.S, self.A)
        SA_next = self.pair(self.S_next, self.A_next)
        # backup values
        delta = R + self.GAMMA * self.Q[SA_next].value - self.Q[SA].value
        self.Q[SA].trace = self.calc_trace(self.Q[SA].trace)
        # update state action pair values
        all_pairs = [key for key in self.Q.keys() if len(key) == len(SA)]
        for sa in all_pairs:
            self.Q[sa].value += self.ALPHA * delta * self.Q[sa].trace
            self.Q[sa].trace *= self.GAMMA * self.LAMBDA

    def check_dict(self, S):
        """ adds state and state-action pairs to dictionary if needed """
        if tuple(S) not in self.Q.keys():
            keys = [tuple(S)] + [self.pair(S, A) for A in self.A_all]
            values = [Q_value(self.initial_value, 0) for _ in keys]
            self.Q.update(zip(keys, values))

    def choose_action(self, S):
        """ choose action following e-greedy policy """
        # greedy action
        values = [self.Q[SA].value for SA in self.state_action_pairs(S)]
        max_idx = [i for i, v in enumerate(values) if v == max(values)]
        action = self.A_all[np.random.choice(max_idx)]
        # non greedy action
        if np.random.rand() < self.EPSILON:
            other_idx = [i for i, v in enumerate(values) if v != max(values)]
            other_idx = max_idx if not other_idx else other_idx
            action = self.A_all[np.random.choice(other_idx)]
        return action

    def calc_trace(self, E):
        """ calculates new trace value given type """
        E_next = 1 if self.TRACE_TYPE == 'replacing' else E
        E_next = E + 1 if self.TRACE_TYPE == 'accumulating' else E
        E_next = (1 - self.ALPHA) * E + 1 if self.TRACE_TYPE == 'dutch' else E
        return E_next


    ### HELPER ###

    def pair(self, S, A):
        """ joins state and action as tuple """
        return tuple(list(S) + list(A))

    def state_action_pairs(self, S):
        """ returns all state-action pairs for given state """
        return [self.pair(S, A) for A in self.A_all]


