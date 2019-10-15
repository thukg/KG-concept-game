import random
import numpy as np
from model.base_model import BaseModel
import model.model_api as model_api

class SpragueGrundyModel(BaseModel):
    def __init__(self, env):
        BaseModel.__init__(self, env)
        self.sg = {}
        labels = [0] * self.n
        self.dp(1, labels)
    
    def state_to_id(self, labels):
        id = 0
        for l in labels:
            id = id*3+l
        return id
    
    def mex(self, sg_set):
        for i in range(len(sg_set)+1):
            if i not in sg_set:
                return i
    
    def dp(self, player, labels):
        id = self.state_to_id(labels)
        if not id in self.sg:
            candidates = model_api.get_candidates(player, labels, self.graph_mat)
            sg_set = set()
            if candidates:
                for c in candidates:
                    labels[c] = player
                    sg_set.add(self.dp(3-player, labels))
                    labels[c] = 0
            self.sg[id] = self.mex(sg_set)
        return self.sg[id]
    
    def forward(self, obs):
        player, labels = obs
        candidates = model_api.get_candidates(player, labels, self.graph_mat)
        #print(player, labels, candidates)
        if candidates:
            for c in candidates:
                labels[c] = player
                if self.sg[self.state_to_id(labels)] == 0:
                    labels[c] = 0
                    return c
                labels[c] = 0
            return random.sample(candidates, 1)[0]
        else:
            return random.randint(0, self.n-1)