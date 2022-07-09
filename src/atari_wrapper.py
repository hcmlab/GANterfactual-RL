import PIL
import cv2
import gym
import numpy as np
from PIL import Image


PAC_MAN_SIZE = (84, 84)
SPACE_INVADERS_SIZE = (80, 80)


class AtariWrapper:
    ''' simple implementation of openai's atari_wrappers for our purposes'''

    def __init__(self, env_name, power_pill_objective=False, deepq_preprocessing=True, ablate_agent=False):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.reset()
        if env_name.startswith("SpaceInvaders"):
            self.space_invaders = True
        else:
            self.space_invaders = False
            # needed since our deepQ and ACER models use different preprocessing
            self.deepq_preprocessing = deepq_preprocessing
        if self.space_invaders:
            self.width, self.height = SPACE_INVADERS_SIZE
        else:
            self.width, self.height = PAC_MAN_SIZE
        self.original_size = (210, 160, 3)
        self.zeros = np.zeros((self.width, self.height))
        self.stacked_frame = np.stack((self.zeros, self.zeros, self.zeros, self.zeros), axis=-1)
        self.original_zeros = np.zeros(self.original_size)
        self.original_stacked_frame = np.stack((self.original_zeros, self.original_zeros, self.original_zeros,
                                                self.original_zeros), axis=-1)
        self.noop_action = 0
        self.power_pill_objective = power_pill_objective
        self.power_pills_left = 4
        self.ablate_agent = ablate_agent

    @staticmethod
    def preprocess_frame(frame):
        ''' preprocessing according to openai's atari_wrappers.WrapFrame
            also applys scaling between 0 and 1 which is done in tensorflow in baselines
        :param frame: the input frame
        :return: rescaled and greyscaled frame
        '''
        if len(frame) == 210:
            frame = frame[0:173, :, :]
        elif len(frame) == 176:
            frame = np.array(Image.fromarray(frame).crop((8, 1, 168, 174)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, PAC_MAN_SIZE, interpolation=cv2.INTER_AREA)
        frame = frame / 255.
        return frame[:, :, None]

    @staticmethod
    def preprocess_frame_ACER(frame):
        ''' preprocessing according to openai's atari_wrappers.WrapFrame
            Does NOT apply scaling between 0 and 1 since ACER does not use it
        :param frame: the input frame
        :return: rescaled and greyscaled frame
        '''
        if len(frame) == 210:
            frame = frame[0:173, :, :]
        elif len(frame) == 176:
            frame = np.array(Image.fromarray(frame).crop((8, 1, 168, 174)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, PAC_MAN_SIZE, interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

    @staticmethod
    def preprocess_space_invaders_frame(frame, ablate_agent):
        if len(frame) == 210:
            frame = frame[35:195]
        frame = Image.fromarray(frame).resize(SPACE_INVADERS_SIZE, PIL.Image.BICUBIC)
        frame = np.array(frame).astype(np.float32).mean(2)
        frame = frame / 255.
        if ablate_agent:
            frame[70:] = 0
        return frame

    @staticmethod
    def preprocess_original_frame(frame):
        frame = frame / 255.
        return frame[:, :, None]

    def update_stacked_frame(self, new_frame):
        ''' adds the new_frame to the stack of 4 frames, while shifting each other frame one to the left
        :param new_frame:
        :return: the new stacked frame
        '''
        for i in range(3):
            self.stacked_frame[:, :, i] = self.stacked_frame[:, :, i + 1]
        if self.space_invaders:
            new_frame = self.preprocess_space_invaders_frame(new_frame, self.ablate_agent)
        elif self.deepq_preprocessing:
            new_frame = self.preprocess_frame(new_frame)
        else:
            new_frame = self.preprocess_frame_ACER(new_frame)
        new_frame = np.squeeze(new_frame)
        self.stacked_frame[:, :, 3] = new_frame
        return self.stacked_frame

    def update_original_stacked_frame(self, new_frame):
        for i in range(3):
            self.original_stacked_frame[:, :, :, i] = self.original_stacked_frame[:, :, :, i + 1]
        new_frame = self.preprocess_original_frame(new_frame)
        new_frame = np.squeeze(new_frame)
        self.original_stacked_frame[:, :, :, 3] = new_frame
        return self.original_stacked_frame

    @staticmethod
    def to_channels_first(stacked_frames):
        stacked_frames = np.array(stacked_frames)
        channels_first = np.empty((4, stacked_frames.shape[0], stacked_frames.shape[1]))
        for j in range(4):
            channels_first[j] = stacked_frames[:, :, j]
        return channels_first

    def step(self, action, skip_frames=4):
        max_frame, stacked_observations, reward, done, info = self.repeat_frames(action, skip_frames=skip_frames)
        stacked_frames = self.update_stacked_frame(max_frame)
        self.update_original_stacked_frame(max_frame)

        if self.power_pill_objective and self.power_pills_left == 0:
            done = True

        # reset the environment if the game ended
        if done:
            self.reset()

        if self.space_invaders:
            return self.to_channels_first(stacked_frames), stacked_observations, reward, done, info
        else:
            return stacked_frames, stacked_observations, reward, done, info

    def repeat_frames(self, action, skip_frames=4):
        ''' skip frames to be inline with baselines DQN. stops when the current game is done
        :param action: the choosen action which will be repeated
        :param skip_frames: the number of frames to skip
        :return max frame: the frame used by the agent
        :return stacked_observations: all skipped observations '''
        stacked_observations = []
        # TODO dirty numbers
        obs_buffer = np.zeros((2, self.original_size[0], self.original_size[1], 3), dtype='uint8')
        total_reward = 0
        for i in range(skip_frames):
            observation, reward, done, info = self.env.step(action)
            stacked_observations.append(observation)
            if i == skip_frames - 2: obs_buffer[0] = observation
            if i == skip_frames - 1: obs_buffer[1] = observation
            if done:
                break

            if self.power_pill_objective:
                if self.ate_power_pill(reward):
                    self.power_pills_left -= 1
                    reward = 50
                else:
                    reward = 0

            total_reward += reward

        max_frame = obs_buffer.max(axis=0)
        return max_frame, stacked_observations, total_reward, done, info

    def reset(self, noop_min=0, noop_max=30):
        """ Do no-op action for a number of steps in [1, noop_max], to achieve random game starts.
        We also do no-op for 250 steps because Pacman cant do anything at the beginning of the game (number found empirically)
        """
        self.power_pills_left = 4
        self.env.reset()
        if not self.space_invaders:
            for _ in range(250):
                obs, _, done, _ = self.env.step(self.noop_action)
                if done:
                    obs = self.env.reset()
        noops = np.random.randint(noop_min + 1, noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs

    def fixed_reset(self, step_number, action):
        '''
        Create a fixed starting position for the environment by doing *action* for *step_number* steps
        :param step_number: number of steps to be done at the beginning of the game
        :param action: action to be done at the start of the game
        :return: obs at the end of the starting sequence
        '''
        self.env.reset()
        for _ in range(step_number):
            obs, _, done, _ = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()
        return obs

    @staticmethod
    def ate_power_pill(reward):
        return reward == 50


