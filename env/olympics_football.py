import random
import os
import sys
from pathlib import Path

from olympics_engine.generator import create_scenario
from olympics_engine.scenario.football import *

from utils.box import Box
from env.simulators.game import Game

import numpy as np

class OlympicsFootball(Game):
    def __init__(self, conf, seed=None):
        super(OlympicsFootball, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                         conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.seed = seed
        self.set_seed()

        Gamemap = create_scenario("football")
        self.env_core = football(Gamemap)
        self.max_step = int(conf['max_step'])
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space

        map_element = []
        for i in self.env_core.map['objects']:
            if i.type == 'arc':
                map_element.append([i.type, i.init_pos, i.start_radian*180/math.pi, i.end_radian*180/math.pi, i.color])
            else:
                map_element.append([i.type, i.init_pos, i.color])
        self.init_info = {"scenario": "football", "map_objects": map_element}

        self.step_cnt = 0
        self.won = {}
        self.n_return = [0] * self.n_player

        _ = self.reset()

        self.board_width = self.env_core.view_setting['width']+2*self.env_core.view_setting['edge']
        self.board_height = self.env_core.view_setting['height']+2*self.env_core.view_setting['edge']

        self.init_info["board_height"] = self.board_height
        self.init_info["board_width"] = self.board_width

    @staticmethod
    def create_seed():
        seed = random.randrange(1000)
        return seed

    def set_seed(self, seed=None):
        if not seed:        #use previous seed when no new seed input
            seed = self.seed
        else:               #update env global seed
            self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        init_obs = self.env_core.reset()
        self.ball_pos_init()
        self.step_cnt = 0
        self.done = False
        self.won = {}
        self.n_return = [0]*self.n_player

        self.init_info["agent_position"] = self.env_core.agent_pos
        self.init_info["agent_direction"] = [self.env_core.agent_theta[i][0] for i in range(len(self.env_core.agent_list))] #copy.deepcopy(self.env_core.agent_theta)
        self.init_info["agent_color"] = [self.env_core.agent_list[i].color for i in range(len(self.env_core.agent_list))]
        self.init_info["agent_r"] = [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))]
        self.init_info["agent_energy"] = [self.env_core.agent_list[i].energy for i in range(len(self.env_core.agent_list))]
        self.init_info["agent_vis"] = [self.env_core.agent_list[i].visibility for i in range(len(self.env_core.agent_list))]
        self.init_info["agent_vis_clear"] = [self.env_core.agent_list[i].visibility_clear for i in range(len(self.env_core.agent_list))]

        self.current_state = init_obs
        self.all_observes = self.get_all_observes()
        self.ball_end_pos=None

        return self.all_observes

    def ball_pos_init(self):
        y_min, y_max = 300, 500
        for index, item in enumerate(self.env_core.agent_list):
            if item.type == 'ball':
                random_y = random.uniform(y_min, y_max)
                self.env_core.agent_init_pos[index][1] = random_y




    def step(self, joint_action):
        self.is_valid_action(joint_action)
        joint_action_decode = self.decode(joint_action)
        info_before = {"actions": [i for i in joint_action_decode]}

        all_observations, reward, done, info_after = self.env_core.step(joint_action_decode)
        info_after = self.step_after_info()
        self.current_state = all_observations
        self.all_observes = self.get_all_observes()

        self.step_cnt += 1
        self.done = done
        if self.done:
            self.ball_position()
            self.set_n_return()

        return self.all_observes, reward, self.done, info_before, info_after



    def is_valid_action(self, joint_action):
        if len(joint_action) != self.n_player:          #check number of player
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

    def step_after_info(self, info=''):
        info = {"agent_position":self.env_core.agent_pos, "agent_direction":[self.env_core.agent_theta[i][0] for i in range(len(self.env_core.agent_list))],
                "agent_energy":[self.env_core.agent_list[i].energy for i in range(len(self.env_core.agent_list))]}
        return info

    def decode(self, joint_action):
        joint_action_decode = []
        for act_id, nested_action in enumerate(joint_action):
            temp_action = [0, 0]
            temp_action[0] = nested_action[0][0]
            temp_action[1] = nested_action[1][0]
            joint_action_decode.append(temp_action)

        return joint_action_decode

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each = {"obs": self.current_state[i], "controlled_player_index": i}
            all_observes.append(each)

        return all_observes

    def set_action_space(self):
        return [[Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))] for _ in range(self.n_player)]

    def get_reward(self, reward):
        return [reward]

    def is_terminal(self):
        return self.env_core.is_terminal()

    def ball_position(self):
        self.ball_end_pos = None
        for agent_idx in range(self.env_core.agent_num):
            agent = self.env_core.agent_list[agent_idx]
            if agent.type == 'ball' and agent.finished:
                self.ball_end_pos = self.env_core.agent_pos[agent_idx]

    def set_n_return(self):


        if self.ball_end_pos is None:
            self.n_return = [0,0]
        else:
            if self.ball_end_pos[0]<400:
                # if self.env_core.agent_pos[0][0]<400:
                #     return [0,1]
                # else:
                #     return [1,0]
                self.n_return = [0,1]
            elif self.ball_end_pos[0]>400:
                # if self.env_core.agent_pos[0][0]<400:
                #     return [1,0]
                # else:
                #     return [0,1]
                self.n_return = [1,0]
            else:
                raise NotImplementedError



    def check_win(self):
        if self.ball_end_pos is None:
            return '-1'
        else:
            if self.ball_end_pos[0]<400:
                # if self.env_core.agent_pos[0][0]<400:
                #     return '1'
                # else:
                #     return '0'
                return '1'
            elif self.ball_end_pos[0]>400:
                # if self.env_core.agent_pos[0][0]<400:
                #     return '0'
                # else:
                #     return '1'
                return '0'
            else:
                raise NotImplementedError


    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]




