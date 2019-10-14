import numpy as np

class PlayInfo:
    def __init__(self, winner, play_first, history):
        self.winner = winner
        self.play_first = play_first
        self.history = history
        self.step = len(self.history)

def play_info_stat(plays):
    stat = {}
    for p in plays:
        s = 'winner: {}, play_first: {}'.format(p.winner, p.play_first)
        if s not in stat:
            stat[s] = []
        stat[s].append(p.step)
    for key in stat:
        print('{} times: {} mean_step: {}'.format(key, len(stat[key]), np.mean(stat[key])))