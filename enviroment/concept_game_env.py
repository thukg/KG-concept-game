import gym
from gym.spaces import Discrete, Box
import config
import enviroment.concept_game_api
import pickle
import numpy as np
import os
import itertools

class ConceptGameEnv(gym.Env):
    def __init__(self, env_config):
        self.name, parser = env_config['name'], env_config['parser']
        input_file = config.graph_data+self.name.lower()
        if not os.path.exists(input_file):
            raise Exception('no {} graph'.format(self.name))
        self.names, self.edges, self.graph_mat = parser(input_file)
        self.n, self.m = len(self.names), len(self.edges)
        self.get_graph_dis()
        self.action_space = Discrete(self.n)
        self.observation_space = [Discrete(3), Box(0, 2, shape=(self.n, ), dtype=np.int32), Box(0, 1, shape=(self.n, self.n), dtype=np.float32)]
    
    def get_graph_dis(self):
        mat = self.graph_mat
        n = self.n
        dis = np.zeros((n, n))
        for i, j in itertools.product(range(n), range(n)):
            if i != j:
                dis[i][j] = config.max_d if mat[i][j] == 0 else mat[i][j]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dis[i][j] = min(dis[i][j], dis[i][k]+dis[k][j])
        self.graph_dis = dis
    
    def reset(self):
        self.labels = [0] * self.n
        self.refine_history = []
        self.history = []
        self.player = 1
        self.done = False
        obs = [self.player, self.labels]
        return obs
    
    def step(self, a):
        assert not self.done and a in range(self.n)
        round = len(self.history)
        obs = [self.player, self.labels]
        if self.labels[a] != 0:
            self.done = True
            return obs, -1, True, {'message': 'illegal action, repeated occupation'}
        if round >= 2:
            p = sum([self.labels[i] == self.player for i in range(self.n) if self.graph_mat[a, i] > 0])
            if p == 0:
                self.done = True
                return obs, -1, True, {'message': 'illegal action, no adjacent occupation'}
        self.labels[a] = self.player
        self.refine_history.append(a)
        self.history.append({'round': round, 'player': self.player, 'action': a})
        self.player = 3 - self.player
        obs = [self.player, self.labels]
        return obs, 0, False, {'message': 'action accepted'}
    
    def to_nn_input(self, obs):
        player, labels = obs
        input = []
        for l in labels:
            if l == 0:
                input.append([1, 0, 0])
            if l == player:
                input.append([0, 1, 0])
            if l == 3 - player:
                input.append([0, 0, 1])
        return np.array(input)
    
    def render(self):
        pass
    
    def close(self):
        pass
    
    def seed(self):
        pass
    
    def save_replay(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pickle.dump({'n': self.n, 'm': self.m, 'edges': self.edges, 'history': self.history})