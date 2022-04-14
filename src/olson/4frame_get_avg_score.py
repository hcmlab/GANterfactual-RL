# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imsave
from scipy.misc import imresize

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'
from collections import deque

def immsave(file, pixels, size=400):
    imsave(file, imresize(pixels,size))

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='SpaceInvaders-v0', type=str, help='gym environment')
    parser.add_argument('--missing', default='none', type=str, help='gym environment')
    parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=0, type=int, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='test mode sets lr=0, chooses most likely actions')
    parser.add_argument('--lstm_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--gpu', default=7, type=int, help='')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--game_runs', default=30, type=int, help='# of games to play')
    parser.add_argument('--fskip', default=8, type=int)
    parser.add_argument('--latent', default=256, type=int)
    parser.add_argument('--gamma', default=0.99, type=float, help='discount for gamma-discounted rewards')
    parser.add_argument('--tau', default=1.0, type=float, help='discount for generalized advantage estimation')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--agent_file', default="", type=str, help='file to test')
    parser.add_argument('--img_dir', type=str, default='4frame_vid')
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: imresize(img[35:195], (80,80)).astype(np.float32).mean(2).reshape(1,80,80)/255.



def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()
        
class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, num_actions, latent_size):
        super(NNPolicy, self).__init__()
        channels = 4
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 5 * 5, latent_size)
        self.critic_linear, self.actor_linear = nn.Linear(latent_size, 1), nn.Linear(latent_size, num_actions)


    def forward(self, inputs):
        x = F.elu(self.conv1(inputs)) ; x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x)) ; x = F.elu(self.conv4(x))
        x = self.linear(x.view(-1, 32 * 5 * 5))
        return self.critic_linear(x), self.actor_linear(x)

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar')
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts)
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

def ablate_screen(orig_img, section):
    img = orig_img
    #height first, then width
    if section == "none":
        return orig_img
    if section == "bottom":
        img[:,40:] = 0
    elif section == "top":
        img[:,:40] = 0
    elif section == "barrier":
        img[:,60:70] = 0
    elif section == "agent":
        img[:,70:] = 0
    elif section == "left":
        img[:,:,:40] = 0
    elif section == "right":
        img[:,:,40:] = 0
    elif section == "center_column":
        img[:,:,20:60] = 0
    elif section == "center_row":
        img[:,20:60,:] = 0
    elif section == "stripe_column":
        img[:,:,5:10 ] = 0
        img[:,:,15:20] = 0
        img[:,:,25:30] = 0
        img[:,:,35:40] = 0
        img[:,:,45:50] = 0
        img[:,:,55:60] = 0
        img[:,:,65:70] = 0
        img[:,:,75:80] = 0
    elif section == "stripe_row":
        img[:,5:10 ,:] = 0
        img[:,15:20,:] = 0
        img[:,25:30,:] = 0
        img[:,35:40,:] = 0
        img[:,45:50,:] = 0
        img[:,55:60,:] = 0
        img[:,65:70,:] = 0
        img[:,75:80,:] = 0
    else:
        raise "hey, you tried to ablate a screen that didnt exist"

    return img

def train(model, args):
    env = gym.make(args.env) # make a local (unshared) environment
    env.unwrapped.frameskip = args.fskip
    env.seed(args.seed )  
    torch.manual_seed(args.seed)
    if args.render == 1: os.makedirs(args.img_dir, exist_ok=True)
    
    bs = 1

    img = prepro(env.reset())
    img = ablate_screen(img, args.missing)
    state = Variable(torch.Tensor(img).view(1,1,80,80)).cuda()
    state_history = deque([state, state.clone(), state.clone(),state.clone()], maxlen=4)

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, False # bookkeeping
    np.set_printoptions(precision=4)

    epr_list = []
    clipped_reward = []
    cur_clipped_reward = 0
    while len(epr_list) < args.game_runs: # openai baselines uses 40M frames...we'll use 80M
        state = torch.cat(list(state_history), dim=1)

        episode_length += 1
        value, logit = model(state)
        logp = F.log_softmax(logit, dim=1)

        action = logp.max(1)[1].data
        #if episode_length == 432 or episode_length == 433:
        #    action = 5
        new_frame, reward, done, _ = env.step(action)

        #if args.render == 1: immsave(os.path.join(args.img_dir, "{:05d}.png".format(episode_length)),new_frame)
        img = ablate_screen(prepro(new_frame), args.missing)
        state_history.append(Variable(torch.Tensor(img).view(1,1,80,80)).cuda())

        epr += reward
        reward = np.clip(reward, -1, 1) # reward
        if reward < 0: print("????")
        cur_clipped_reward += reward
        done = done or episode_length >= 1e4 # keep agent from playing one episode too long
        if env.unwrapped.ale.lives() < 3: done = True
        if done: # update shared data. maybe print info.
            if args.render == 1: 
                print("{}, {}, {}, {}".format(args.seed, epr, cur_clipped_reward, episode_length))
                exit()
            epr_list.append(epr)
            clipped_reward.append(cur_clipped_reward)
            episode_length, epr, eploss = 0, 0, 0
            cur_clipped_reward = 0
            img = ablate_screen(prepro(env.reset()), args.missing)
            state = Variable(torch.Tensor(img).view(bs,1,80,80).cuda())
            state_history = deque([state, state.clone(), state.clone(),state.clone()], maxlen=4)

    #print("latent size: {}".format(args.latent))
    output = np.array(epr_list)
    output2 = np.array(clipped_reward)
    print("{},{},{},{},{}".format(args.latent,np.mean(output),np.std(output),np.mean(output2),np.std(output2)))
    #print("scores: mean: {}, std: {}".format(np.mean(output),np.std(output)))
    #print("rewards: mean: {}, std: {}".format(np.mean(output2),np.std(output2)))

if __name__ == "__main__":
    args = get_args()
    args.num_actions = gym.make(args.env).action_space.n # get the action space of this game
    args.img_dir += os.path.basename(args.agent_file)

    if args.fskip % 2 ==0 and args.env == 'SpaceInvaders-v0':
        print("SpaceInvaders needs odd frameskip due to bullet alternations")
        args.fskip = args.fskip - 1
    
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    model = NNPolicy(args.num_actions, args.latent).cuda() 
    model.load_state_dict(torch.load(args.agent_file))

    train(model, args)
   
