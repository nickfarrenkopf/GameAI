


class Action(object):
    """ action player can take during turn """

    def __init__(self, name, values):
        """ defines action by name and range of possible values """
        self.name = name
        self.values = values



class Environment(object):
    """ """

    def __init__(self):
        """ """

        # state params
        self.state = None
        self.state_size = None
        self.terminal_states = None

        # action params
        self.actions = None


    ### STATE ###

    def set_initial_state(self):
        """ """
        pass

    def set_terminal_states(self):
        """ """
        pass

    def in_terminal_state(self):
        """ """
        pass


    ### ACTION ###

    def define_actions(self):
        """ """
        pass

    def take_action(self):
        """ """
        pass
    

    ### LEARNING ###

    def define_method(self):
        """ """
        pass


    ### HELPER ###

    def get_action_profile(self):
        """ returns a list consisting of all possible action combos """
        action_values = [action.values for action in self.actions]
        return list(itertools.product(*action_values))  

