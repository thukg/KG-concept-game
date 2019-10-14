from abc import ABC, abstractmethod

class BaseModel:
    def __init__(self, env):
        self.n = env.n
        self.trainable = False
        self.nn_model = False
        self.graph_mat = env.graph_mat
        self.graph_dis = env.graph_dis
    
    @abstractmethod
    def forward(self, obs):
        pass