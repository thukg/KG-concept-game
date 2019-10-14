from model.random_model import RandomModel
from model.greedy_model import GreedyModel
from model.alpha_beta_model import AlphaBetaModel
from model.fully_connected_network_model import FCNModel
import config

def name_to_model(s, env):
    s = s.lower()
    assert s in config.model_list
    model = None
    if s == 'random':
        model = RandomModel(env)
    if s == 'greedy-p0':
        model = GreedyModel(env, 0)
    if s == 'greedy-p1':
        model = GreedyModel(env, 1)
    if s == 'greedy-p2':
        model = GreedyModel(env, 2)
    if s == 'ab-p0':
        model = AlphaBetaModel(env, 0)
    if s == 'ab-p1':
        model = AlphaBetaModel(env, 1)
    if s == 'ab-p2':
        model = AlphaBetaModel(env, 2)
    if s == 'fcn':
        model = FCNModel(env)
    model.name = s
    return model

def get_candidates(player, labels, graph_mat):
    n = len(labels)
    candidates = []
    s = sum([player == l for l in labels])
    for i in range(n):
        if labels[i] == 0 and (s == 0 or sum([labels[j] == player for j in range(n) if graph_mat[i, j] > 0])):
            candidates.append(i)
    return candidates

def bfs(f, mat):
    n = len(f)
    g = [i for i in range(n) if f[i] == 0]
    while g:
        t = g[0]
        for i in range(n):
            if mat[t, i] > 0:
                if f[i] > f[t] + 1:
                    f[i] = f[t] + 1
                    g.append(i)
        g = g[1:]
    return f

def value_func0(player, labels, graph_mat, graph_dis):
    n = len(labels)
    g = [[], [], []]
    for i in range(n):
        g[labels[i]].append(i)
    s = 0
    for l0 in g[0]:
        min1 = min([n]+[graph_dis[i, l0] for i in g[1]])
        min2 = min([n]+[graph_dis[i, l0] for i in g[2]])
        if min1 < min2:
            s += (player == 1)
        if min1 > min2:
            s += (player == 2)
    return s
    
def value_func1(player, labels, graph_mat, graph_dis):
    n = len(labels)
    self_f, oppo_f = [n]*n, [n]*n
    for i in range(n):
        if labels[i] == player:
            self_f[i] = 0
            oppo_f[i] = -1
        if labels[i] == 3-player:
            self_f[i] = -1
            oppo_f[i] = 0
    self_f = bfs(self_f, graph_mat)
    oppo_f = bfs(oppo_f, graph_mat)
    s = 0
    for i in range(n):
        if labels[i] == 0:
            if self_f[i] < oppo_f[i]:
                s += 1
            if self_f[i] > oppo_f[i]:
                s -= 1
    return s

def value_func2(player, labels, graph_mat, graph_dis):
    n = len(labels)
    self_f, oppo_f = [n]*n, [n]*n
    for i in range(n):
        if labels[i] == player:
            self_f[i] = 0
            oppo_f[i] = -1
        if labels[i] == 3-player:
            self_f[i] = -1
            oppo_f[i] = 0
    self_f = bfs(self_f, graph_mat)
    oppo_f = bfs(oppo_f, graph_mat)
    s = sum([-self_f[i]+oppo_f[i] for i in range(n) if labels[i] == 0])
    return s

value_funcs = [value_func0, value_func1, value_func2]