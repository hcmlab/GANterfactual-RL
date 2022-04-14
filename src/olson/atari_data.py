import sys
import numpy as np
import gym
from scipy.misc import imresize
from concurrent import futures


#prepro = lambda img: imresize(img[35:195], (80,80)).astype(np.float32).reshape(3,80,80)/255.
#prepro_rgb = lambda img: imresize(img[35:195], (80,80)).astype(np.float32)/255.
prepro_rgb = lambda img: img[35:195].astype(np.float32)/255.
prepro_bw = lambda img: imresize(img[35:195], (80,80)).astype(np.float32).mean(2).reshape(1,80,80)/255.

#returns rgb, bw
'''def prepro(img):
    if img.shape[0] > 80: 
        img = img[35:195]
        img2 = imresize(img, (80,80)).astype(np.float32)
        
    return img.astype(np.float32)/255. , img2.mean(2).reshape(1,80,80)/255.
'''
def prepro(img):
    return prepro_rgb(img), prepro_bw(img)


def ablate_screen(orig_img, section):
    img = orig_img
    if section == "none":
        return orig_img
    #height first, then width
    if section == "bottom":
        img[:,:,40:] = 0
    elif section == "top":
        img[:,:,:40] = 0
    elif section == "barrier":
        img[:,:,60:70] = 0
    elif section == "agent":
        img[:,:,70:] = 0
    elif section == "left":
        img[:,:,:,:40] = 0
    elif section == "right":
        img[:,:,:,40:] = 0
    elif section == "center_column":
        img[:,:,:,20:60] = 0
    elif section == "center_row":
        img[:,:,20:60,:] = 0
    elif section == "stripe_column":
        img[:,:,:,5:10 ] = 0
        img[:,:,:,15:20] = 0
        img[:,:,:,25:30] = 0
        img[:,:,:,35:40] = 0
        img[:,:,:,45:50] = 0
        img[:,:,:,55:60] = 0
        img[:,:,:,65:70] = 0
        img[:,:,:,75:80] = 0
    elif section == "stripe_row":
        img[:,:,5:10 ,:] = 0
        img[:,:,15:20,:] = 0
        img[:,:,25:30,:] = 0
        img[:,:,35:40,:] = 0
        img[:,:,45:50,:] = 0
        img[:,:,55:60,:] = 0
        img[:,:,65:70,:] = 0
        img[:,:,75:80,:] = 0
    else:
        raise "hey, you tried to ablate a screen that didnt exist"

    return img

def map_fn(fn, *iterables):
    with futures.ThreadPoolExecutor(max_workers=16) as executor:
        result_iterator = executor.map(fn, *iterables)
    return [i for i in result_iterator]


class MultiEnvironment():
    def __init__(self, name, batch_size, fskip = 0):
        self.batch_size = batch_size
        self.envs = [] #map(lambda idx: gym.make(name), range(batch_size))
        self.name = name
        for i in range(batch_size):
           
            env = gym.make(name) 
            if fskip > 0: env.unwrapped.frameskip = fskip

            env.seed(i)
            self.envs.append(env)
        self.action_meanings = self.envs[0].unwrapped.get_action_meanings()
        self.saved_state = None

    def seed(self, seed):
        for i in range(self.batch_size):
            self.envs[i].seed(seed + i)

    def reset(self):
        bws = []
        rgbs = []
        for env in self.envs:
            rgb, bw = prepro(env.reset())
            rgbs.append(rgb)
            bws.append(bw)
        return np.array(rgbs), np.array(bws)

    def get_action_size(self, env_name = None):
        return self.envs[0].action_space.n

    def only_one_env(self):
        self.envs = [self.envs[0]]

    def clone_full_state(self):
        return [env.unwrapped.clone_full_state() for env in self.envs]

    def restore_full_state(self, states):
        for env, state in zip(self.envs, states):
            env.unwrapped.restore_full_state(state)

    def get_action_meanings(self):
        return self.action_meanings

    def step(self, actions):
        assert len(actions) == len(self.envs)

        def run_one_step(env, action):
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            rgb, bw = prepro(state)
            return rgb, bw, reward, done, info

        results = map_fn(run_one_step, self.envs, actions)
        
        state_rgb, state_bw, rewards, dones, infos = zip(*results)
        return np.array(state_rgb),np.array(state_bw), rewards, dones, infos


if __name__ == '__main__':
    batch_size = 64
    env = MultiEnvironment('Pong-v0', batch_size)
    for i in range(10):
        actions = np.random.randint(0, 4, size=batch_size)
        states, rewards, dones, infos = env.step(actions)

'''
import random
import sys
import numpy as np
import gym
import torch
from torch.autograd import Variable
from scipy.misc import imresize
from copy import deepcopy
from multiprocessing import Pool
import model

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

class AtariDataloader():
    def __init__(self, name, batch_size, agent_file, game_length=1000, frameskip=-1):
        self.environments = []
        self.batch_size = batch_size
        self.game_length = game_length

        temp_env = gym.make(name)
        actions = temp_env.action_space.n

        for i in range(batch_size):
            env = gym.make(name)
            if frameskip >= 0:
                env.unwrapped.frameskip = 3
            env.seed(i)
            env.reset()

            agent = model.Agent(actions).share_memory() #no cuda 
            agent.load_state_dict(torch.load(agent_file))

            self.environments.append((env, agent))

    def get_action_size(self, env_name = None):
        return self.environments[0][0].action_space.n
        

    def _get_playthrough(self, env):
        (env, model) = env
        ret = []
        episode_length = 0
        done = False
        total_length = self.game_length + self.ending
        cx = Variable(torch.zeros(1, 256))# lstm memory vector
        hx = Variable(torch.zeros(1, 256)) # lstm activation vector
        state = torch.Tensor(prepro(env.reset()))

        states = []
        values = []
        logps = []

        while episode_length <= (total_length):
            episode_length += 1

            value, logit, (hx, cx) = model((Variable( state.view(1,1,80,80)), (hx, cx)))
            logp = torch.nn.functional.log_softmax(logit, dim=1)

            action = logp.max(1)[1].data 
            state, reward, done, _ = env.step(action.numpy()[0])
            state = torch.Tensor(prepro(state))
            
            #ret.append((state, value.data, logp))
            states.append(state.numpy())
            values.append(value.data.numpy()[0])
            logps.append(logp.data.numpy()[0])

        return ((states), (values), (logps))

    def _convert_to_torch(self, pooled_games):
        states = []
        values = []
        logps = []
        for g in pooled_games:
            states.append(g[0])
            values.append(g[1])
            logps.append(g[2])

        states = Variable(torch.Tensor(np.array(states))).cuda()
        values = Variable(torch.Tensor(np.array(values))).cuda()
        logps  = Variable(torch.Tensor(np.array(logps))).cuda()

        return torch.transpose(states, 0,1), torch.transpose(values, 0,1), torch.transpose(logps, 0,1)

    def get_game_runs(self):
        self.ending = random.randint(0, 20)

        #ret = self._get_playthrough(self.environments[0])
        #import pdb; pdb.set_trace()

        #return ret

        #return the obs
        pool = Pool(processes = self.batch_size)
        games = pool.map(self._get_playthrough, (self.environments))
        pool.close()

        return self._convert_to_torch(games)
'''