import random
import numpy as np
from model.base_model import BaseModel
import model.model_api as model_api

class GreedyModel(BaseModel):
    def __init__(self, env, policy):
        BaseModel.__init__(self, env)
        assert policy in [0, 1, 2]
        self.get_value = model_api.value_funcs[policy]
    
    def forward(self, obs):
        player, labels = obs
        candidates = model_api.get_candidates(player, labels, self.graph_mat)
        if candidates:
            for c in candidates:
                labels[c] = player
                values = [self.get_value(player, labels, self.graph_mat, self.graph_dis) for c in candidates]
                labels[c] = 0
            return candidates[np.argmax(values)]
        else:
            return random.randint(0, self.n-1)