import random
from model.base_model import BaseModel

class RandomModel(BaseModel):
    def __init__(self, env):
        BaseModel.__init__(self, env)
    
    def forward(self, obs):
        player, labels, graph_mat = obs
        s = sum([player == l for l in labels])
        candidates = []
        for i in range(self.n):
            if labels[i] == 0 and (s == 0 or sum([labels[j] == player for j in range(self.n) if graph_mat[i, j] > 0])):
                candidates.append(i)
        if candidates:
            return random.sample(candidates, 1)[0]
        else:
            return random.randint(0, self.n-1)