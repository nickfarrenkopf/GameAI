import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import AgentUtils
from Library.Learning.Methods import Tabular as TL


class Agent(AgentUtils.Agent):
    """ """

    def __init__(self, environment, key, actions, start_idx=0, learning=True):
        """ """
        AgentUtils.Agent.__init__(self, environment, key, actions)

    
        # learning
        self.learning = learning
        self.set_method()



    ### GRIDWORLD ###

    def learn(self, action):
        """ """
        if self.learning:
            print(self.get_reward())
            self.method.next_time_step(action, self.get_reward())
        

    def get_state(self):
        """ """
        return tuple(self.environment.state)

    def get_color(self):
        """ """
        if self.in_terminal_state():
            return self.C_WIN if self.get_reward() > 0 else self.C_LOSE
        else:
            return self.COLOR


    def interpret_action(self, action=None):
        """ """
        if type(action) is str:
            action = self.actions.find_by_name(action)
        # choose action if none given
        if not action:
            action = self.choose_action()
        return action
    

    ### ACTION ###

    def in_terminal_state(self):
        """ """
        return self.state_idx in self.env.terminal_states


    ### LEARNING ###

    def set_method(self):
        """ """
        print('Using SARSA tabular')
        self.method = TL.Tabular(self)

    def load_value_data(self):
        """ """
        data = self.paths.load_json()['learning']
        if self.name in data:
            self.agent.method.load_q_data_json(data[self.name])

    def save_value_data(self):
        """ """
        data = self.paths.load_json()
        data['learning'].update({self.name: {}})
        for k in self.method.Q.keys():
            SA = ','.join([str(i) for i in list(k)])
            data['learning'][self.name][SA] = self.method.Q[k].value
        self.paths.write_json(data)






