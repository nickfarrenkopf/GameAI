import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import AgentUtils
from Library.Learning.Methods import Tabular as TL


class GridworldAgent(AgentUtils.Agent):
    """ """

    def __init__(self, environment, key, start_idx=0, learning=True,
                 color=Colors.RED, c_win=Colors.GREEN, c_lose=Colors.RED):
        """ """
        AgentUtils.Agent.__init__(self, environment, key)

        # environment
        self.state_idx = None

        # constants
        self.START_IDX = start_idx
        self.COLOR = color
        self.C_WIN = c_win
        self.C_LOSE = c_lose
    
        # learning
        self.learning = learning
        self.set_method()



    ### GRIDWORLD ###

    def learn(self, action):
        """ """
        if self.learning:
            self.method.next_time_step(action, self.get_reward())
        

    def get_state(self):
        """ """
        return tuple(self.environment.state)

    def get_color(self):
        """ """
        if self.in_terminal_state():
            return self.C_WIN if self.env.get_reward() > 0 else self.C_LOSE
        else:
            return self.COLOR


    ### ACTION ###

    def in_terminal_state(self):
        """ """
        return self.state_idx in self.env.terminal_states


    def take_Q_action():
        """ """
        pass

    def agent_press_key(self, action):
        """ """
        if action in keyDictRev:
            key = keyDictRev[action]
            Keyboard.press_key(key)


    def find_by_dict_key(self, string_action):
        """ """
        return key_dict[string_action]






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






