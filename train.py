from enviroment.concept_game_env import ConceptGameEnv
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributions import Categorical
import enviroment.concept_game_api as concept_game_api
import model.model_api as model_api
import config
import time

def train_sub(id, env, model1, model2):
    local_model = model_api.name_to_model(model1.name, env)
    local_model.load_state_dict(model1.state_dict())
    if model2 is None:
        self_play_model = model_api.name_to_model(model1.name, env)
        self_play_model.load_state_dict(model1.state_dict())
    optimizer = optim.Adam(model1.parameters(), lr=config.learning_rate)
    for episode in range(config.train_episode):
        done = False
        s = env.reset()
        s_lst, a_lst, r_lst = [], [], []
        step = 0
        while not done:
            if step % 2 == episode % 2:
                prob = local_model.pi(torch.from_numpy(env.to_nn_input(s)).float())
                a = Categorical(prob).sample().item()
                s_lst.append(env.to_nn_input(s))
                s, r, done, info = env.step(a)
                a_lst.append([a])
                r_lst.append(r)
            else:
                if model2 is None:
                    prob = self_play_model.pi(torch.from_numpy(env.to_nn_input(s)).float())
                    a = Categorical(prob).sample().item()
                else:
                    a = model2.forward(s)
                s, r, done, info = env.step(a)
            step += 1
        if r_lst[-1] == 0:
            R = 1.0
            r_lst[-1] = 1.0
        else:
            R = -1.0
        td_target_lst = []
        for reward in r_lst[::-1]:
            R = config.gamma * R + reward
            td_target_lst.append([R])
        td_target_lst.reverse()
        s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(td_target_lst)
        advantage = td_target - local_model.v(s_batch)
        pi = local_model.pi(s_batch)
        pi_a = pi.gather(1, a_batch)
        loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())
        optimizer.zero_grad()
        loss.mean().backward()
        for model1_param, local_param in zip(model1.parameters(), local_model.parameters()):
            model1_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(model1.state_dict())
        if model2 is None:
            if episode % config.self_play_copy_episode == config.self_play_copy_episode-1:
                self_play_model.load_state_dict(model1.state_dict())
    print("Training process {} reached maximum episode.".format(id))

def test_sub(env, model1, model2):
    score = 0.0
    tot_step = 0
    for episode in range(config.test_episode):
        done = False
        s = env.reset()
        step = 0
        r_last = 0
        while not done:
            if step % 2 == episode % 2:
                prob = model1.pi(torch.from_numpy(env.to_nn_input(s)).float())
                a = Categorical(prob).sample().item()
                s, r, done, info = env.step(a)
                score += r
                r_last = r
            else:
                if model2 is None:
                    prob = model1.pi(torch.from_numpy(env.to_nn_input(s)).float())
                    a = Categorical(prob).sample().item()
                else:
                    a = model2.forward(s)
                s, r, done, info = env.step(a)
            step += 1
        if r_last == 0:
            score += 1.0
        tot_step += step
        if episode % config.print_interval == config.print_interval-1:
            print("# of episode :{}, avg score : {:.1f}, avg step: {:.1f}".format(episode+1, score / config.print_interval, tot_step / config.print_interval))
            score = 0.0
            tot_step = 0
            time.sleep(config.test_sleep_time)

def train_main(env_config, model1, model2=None): # None for self-play
    model1.share_memory()
    processes = []
    for id in range(config.n_train_process+1):
        env = ConceptGameEnv(env_config)
        if id == 0:
            p = mp.Process(target=test_sub, args=(env, model1, model2))
        else:
            p = mp.Process(target=train_sub, args=(id, env, model1, model2))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    torch.save(model1.state_dict(), config.data+env.name+'.'+model1.name+'.modeldata')
    env.close()

def main(env_config, model1_name, model2_name=None):
    env = ConceptGameEnv(env_config)
    assert model1_name in config.neural_model_list and (model2_name is None or model2_name in config.non_neural_model_list)
    model1 = model_api.name_to_model(model1_name, env)
    if not model2_name:
        train_main(env_config, model1)
    else:
        model2 = model_api.name_to_model(model2_name, env)
        train_main(env_config, model1, model2)

if __name__ == '__main__':
    env_config = {'name': 'test.txt', 'parser': concept_game_api.sample_parser}
    main(env_config, 'fcn', 'greedy-p0')