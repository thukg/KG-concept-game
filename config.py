import os

data = 'data/'
enviroment = 'enviroment/'
model = 'model/'
replay = 'replay/'

def init():
    if not os.path.exists(data):
        os.mkdirs(data)
    if not os.path.exists(enviroment):
        os.mkdirs(enviroment)
    if not os.path.exists(model):
        os.mkdirs(model)
    if not os.path.exists(replay):
        os.mkdirs(replay)
init()

non_neural_model_list = ['random', 'greedy-p0', 'greedy-p1', 'greedy-p2', 'minmax-p0', 'minmax-p1', 'minmax-p2', 'ab-p0', 'ab-p1', 'ab-p2', 'sg']
neural_model_list = ['fcn']
model_list = non_neural_model_list + neural_model_list
max_d = 1e8
inf = 1e16
max_depth = 3

graph_data = data+'graph_data/'
n_train_process = 3
learning_rate = 0.002
print_interval = 20
gamma = 0.98
test_episode = 1000
test_sleep_time = 4
#train_episode = int(test_episode * test_sleep_time / print_interval * 40)
self_play_copy_episode = 1000