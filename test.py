from enviroment.concept_game_env import ConceptGameEnv
import enviroment.concept_game_api as concept_game_api
import model.model_api as model_api
import torch
import config
import play_info

def create_graph(random_config):
    concept_game_api.random_generator(random_config['n'], random_config['m'], config.graph_data+random_config['name'])

def prepare(env_config, model1_name, model2_name, model1_path, model2_path):
    env = ConceptGameEnv(env_config)
    model1 = model_api.name_to_model(model1_name, env)
    model2 = model_api.name_to_model(model2_name, env)
    if model1.trainable and model1_path:
        model1.load_state_dict(torch.load(model1_path))
    if model2.trainable and model2_path:
        model2.load_state_dict(torch.load(model2_path))
    return env, model1, model2

def test(env, model1, model2, episode):
    done = False
    s = env.reset()
    step = 0
    while not done:
        step_model = model1 if step % 2 == episode % 2 else model2
        if step_model.nn_model:
            a = step_model.forward(torch.from_numpy(env.to_nn_input(s)).float())
        else:
            a = step_model.forward(s)
        s, r, done, info = env.step(a)
        step += 1
    res = play_info.PlayInfo(1 if step_model == model2 else 2, episode % 2 + 1, env.history)
    print('episode {}, winner {}'.format(episode, res.winner))
    return res

def test_main(env_config, model1_name, model2_name, rounds=1, random_config=None, model1_path=None, model2_path=None):
    assert not (random_config and (model1_name in config.neural_model_list or model2_name in config.neural_model_list))
    if not random_config:
        env, model1, model2 = prepare(env_config, model1_name, model2_name, model1_path, model2_path)
    res = []
    for r in range(rounds):
        if random_config:
            create_graph(random_config)
            env, model1, model2 = prepare(env_config, model1_name, model2_name, model1_path, model2_path)
        res.append(test(env, model1, model2, r))
    play_info.play_info_stat(res)

if __name__ == '__main__':
    graph_name = 'test.txt'
    env_config = {'name': graph_name, 'parser': concept_game_api.sample_parser}
    random_config={'n': 20, 'm': 26, 'name': graph_name}
    test_main(env_config, 'greedy-p0', 'greedy-p0', rounds=10, random_config=random_config)
    #create_graph(50, 1000, 'test.txt')