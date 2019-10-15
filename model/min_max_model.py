import random
import numpy as np
from model.base_model import BaseModel
import model.model_api as model_api
import config

class MinMaxModel(BaseModel):
    def __init__(self, env, policy, max_depth=config.max_depth):
        BaseModel.__init__(self, env)
        assert policy in [0, 1, 2]
        self.get_value = model_api.value_funcs[policy]
        self.max_depth = max_depth
    
    def min_max_search(self, depth, player, labels):
        candidates = model_api.get_candidates(player, labels, self.graph_mat)
        if not candidates:
            return -config.inf, None
        max_score = -config.inf
        policy = None
        for c in candidates:
            labels[c] = player
            if depth < self.max_depth:
                score, _ = self.min_max_search(depth+1, 3-player, labels)
                score = -score
            else:
                score = self.get_value(player, labels, self.graph_mat, self.graph_dis)
            labels[c] = 0
            if score > max_score:
                max_score = score
                policy = c
        return max_score, policy
    
    def forward(self, obs):
        player, labels = obs
        score, policy = self.min_max_search(0, player, labels)
        if policy:
            return policy
        else:
            return random.randint(0, self.n-1)