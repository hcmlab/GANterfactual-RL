from concurrent import futures

import gym
import numpy as np
from PIL import Image, ImageOps

from src.atari_wrapper import AtariWrapper


def prepro_rgb(img, pacman=True):
    if pacman and len(img) != 176:
        img = img[0:173, :, :]
        img = np.array(ImageOps.expand(Image.fromarray(img), (8, 1, 8, 2)))
    elif not pacman and len(img) != 160:
        img = img[35:195]
    img = np.array(img).astype(np.float32) / 255.
    return img


# Setting ablate_agent to default False because the code of olson et al. is already applying ablation elsewhere.
def prepro_bw(img, pacman=True, ablate_agent=False):
    if pacman:
        img = AtariWrapper.preprocess_frame(img)
        img = np.squeeze(img, axis=-1)
    else:
        img = AtariWrapper.preprocess_space_invaders_frame(img, ablate_agent=ablate_agent)
    img = np.expand_dims(img, axis=0)
    return img


def prepro(img, pacman=True):
    return prepro_rgb(img, pacman=pacman), prepro_bw(img, pacman=pacman)


def prepro_dataset_batch(img_batch, pacman=True):
    rgb_batch = []
    bw_batch = []
    for i in range(len(img_batch)):
        if pacman:
            rgb = img_batch[i]
            bw = img_batch[i][1:-2, 8:-8, :]
        else:
            rgb = bw = img_batch[i]
        rgb_batch.append(prepro_rgb(rgb, pacman=pacman))
        bw_batch.append(prepro_bw(bw, pacman=pacman))

    return np.array(rgb_batch), np.array(bw_batch)


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
        print("hey, you tried to ablate a screen that didnt exist")

    return img

def map_fn(fn, *iterables):
    with futures.ThreadPoolExecutor(max_workers=16) as executor:
        result_iterator = executor.map(fn, *iterables)
    return [i for i in result_iterator]


class MultiEnvironment():
    def __init__(self, name, batch_size, fskip=4, power_pill_objective=False):
        self.batch_size = batch_size
        self.envs = [] #map(lambda idx: gym.make(name), range(batch_size))
        self.name = name
        if name.startswith("SpaceInvaders"):
            self.space_invaders = True
        else:
            self.space_invaders = False
        for i in range(batch_size):
           
            env = gym.make(name) 
            # if fskip > 0: env.unwrapped.frameskip = fskip

            env.seed(i)
            self.envs.append(env)
        self.fskip = fskip
        self.action_meanings = self.envs[0].unwrapped.get_action_meanings()
        self.saved_state = None
        self.power_pill_objective = power_pill_objective
        self.power_pills_left = np.full(len(self.envs), 4)
        self.noop_action = 0

    def seed(self, seed):
        for i in range(self.batch_size):
            self.envs[i].seed(seed + i)

    def reset(self, noop_min=0, noop_max=27):
        bws = []
        rgbs = []

        for i, env in enumerate(self.envs):
            self.power_pills_left[i] = 4
            env.reset()
            for _ in range(250):
                obs, _, done, _ = env.step(self.noop_action)
                if done:
                    obs = env.reset()
            noops = np.random.randint(noop_min + 1, noop_max + 1)
            for _ in range(noops):
                obs, _, done, _ = env.step(self.noop_action)
                if done:
                    obs = env.reset()

            rgb, bw = prepro(obs, pacman=not self.space_invaders)
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
            state, stacked_state, reward, done, info = self.repeat_frames(env, action)
            if done:
                state = env.reset()
            rgb, bw = prepro(state, pacman=not self.space_invaders)
            return rgb, bw, reward, done, info

        results = map_fn(run_one_step, self.envs, actions)
        
        state_rgb, state_bw, rewards, dones, infos = zip(*results)
        return np.array(state_rgb), np.array(state_bw), rewards, dones, infos

    def repeat_frames(self, env, action):
        ''' skip frames to be inline with baselines DQN. stops when the current game is done
        :param action: the choosen action which will be repeated
        :param skip_frames: the number of frames to skip
        :return max frame: the frame used by the agent
        :return stacked_observations: all skipped observations '''
        stacked_observations = []
        total_reward = 0
        done = False
        info = None
        for i in range(self.fskip):
            observation, reward, done, info = env.step(action)
            total_reward += reward
            stacked_observations.append(observation)

            if self.ate_power_pill(reward):
                self.power_pills_left -= 1

        return observation, stacked_observations, total_reward, done, info

    @staticmethod
    def ate_power_pill(reward):
        return reward == 50


if __name__ == '__main__':
    batch_size = 64
    env = MultiEnvironment('Pong-v0', batch_size)
    for i in range(10):
        actions = np.random.randint(0, 4, size=batch_size)
        states, rewards, dones, infos = env.step(actions)
