# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='SpaceInvaders-v0', type=str, help='gym environment')
    parser.add_argument('--missing', default='bottom', type=str, help='which part of the screen to ablate')
    parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--latent_size', default=32, type=int, help='size of the last fc layer')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='test mode sets lr=0, chooses most likely actions')
    parser.add_argument('--lstm_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--frameskip', default=7, type=int)
    parser.add_argument('--gamma', default=0.99, type=float, help='discount for gamma-discounted rewards')
    parser.add_argument('--tau', default=1.0, type=float, help='discount for generalized advantage estimation')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: imresize(img[35:195], (80,80)).astype(np.float32).mean(2).reshape(1,80,80)/255.

def ablate_screen(orig_img, section):
    img = orig_img
    #height first, then width
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

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()
        
class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels, num_actions, latent_size = 256):
        super(NNPolicy, self).__init__()
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
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # there's a "step += 1" later
            super.step(closure)

def train(shared_model, shared_optimizer, rank, args, info):
    env = gym.make(args.env) # make a local (unshared) environment
    env.unwrapped.frameskip = args.frameskip
    env.seed(args.seed + rank) ; torch.manual_seed(args.seed + rank) # seed everything
    model = NNPolicy(channels=4, num_actions=args.num_actions, latent_size=args.latent_size) # init a local (unshared) model
    img = ablate_screen(prepro(env.reset()), args.missing)
    state = Variable(torch.Tensor(img).view(1,1,80,80))
    state_history = deque([state, state.clone(), state.clone(),state.clone()], maxlen=4)

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    while info['frames'][0] <= 1.6e8 or args.test: # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss
        
        for step in range(args.lstm_steps):
            episode_length += 1
            state = torch.cat(list(state_history), dim=1)
            
            value, logit = model(state)
            logp = F.log_softmax(logit, dim=1)

            action = logp.max(1)[1] if args.test else torch.exp(logp).multinomial(num_samples = 1)
            new_frame, reward, done, _ = env.step(action.item())
            img = ablate_screen(prepro(new_frame), args.missing)
            state_history.append(Variable(torch.Tensor(img).view(1,1,80,80)))

            if args.render: save_frame(env, args.save_dir + 'imgs/', episode_length)#env.render()


            epr += reward
            reward = np.clip(reward, -1, 1) # reward
            done = done or episode_length >= 1e4 # keep agent from playing one episode too long
            
            info['frames'] += 1 ; num_frames = int(info['frames'][0])
            if num_frames % 2e6 == 0: # save every 2M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

            if done: # update shared data. maybe print info.
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

                if rank ==0 and time.time() - last_disp_time > 60: # print info ~ every minute
                    elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                    printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, run epr {:.2f}, run loss {:.2f}'
                        .format(elapsed, info['episodes'][0], num_frames/1e6, info['run_epr'][0], info['run_loss'][0]))
                    last_disp_time = time.time()

                episode_length, epr, eploss = 0, 0, 0
                img = ablate_screen(prepro(env.reset()), args.missing)
                state = Variable(torch.Tensor(img).view(1,1,80,80))
                state_history = deque([state, state.clone(), state.clone(),state.clone()], maxlen=4)
                if args.test: exit()

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)

        
        next_value = Variable(torch.zeros(1,1)) if done else model(torch.cat(list(state_history), dim=1))[0]
        values.append(Variable(next_value.data))

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad() ; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        shared_optimizer.step()

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    gae = discount(delta_t, args.gamma * args.tau)
    logpys = logps.gather(1, Variable(actions).view(-1,1))
    policy_loss = -(logpys.view(-1) * Variable(torch.Tensor(gae.copy()))).sum()
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = Variable(torch.Tensor(discounted_r.copy()))
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum() # encourage lower entropy
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
    
def test(args):
    from scipy.misc import imsave
    env = gym.make(args.env)
    env.reset()
    for i in range(30):
        new_frame, reward, done, _ = env.step(0)
    img = prepro(new_frame)
    import pdb; pdb.set_trace()
    img = ablate_screen(img, args.missing)
    imsave("test.png", img[0] * 255)
    state = Variable(torch.Tensor(img).view(1,1,80,80))

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        #mp.set_start_method("spawn") #this must not be in global scope
        mp_context = mp.get_context("spawn")
    elif sys.platform == 'linux' or sys.platform == 'linux2': 
        raise "Must be using Python 3 with linux!" #or else you get a deadlock in conv2d
    
    args = get_args()
    if args.frameskip % 2 ==0 and args.env == 'SpaceInvaders-v0':
        print("SpaceInvaders needs odd frameskip due to bullet alternations")
        args.frameskip = args.frameskip - 1

    #test(args)

    args.save_dir = '{}-{}-{}fskip_latent{}/'.format(args.env.lower(), args.missing, args.frameskip, args.latent_size) # keep the directory structure simple
    if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode w one process
    if args.test:  args.lr = 0 # don't train in render mode
    args.num_actions = gym.make(args.env).action_space.n # get the action space of this game
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.
    
    torch.manual_seed(args.seed)
    shared_model = NNPolicy(channels=4, num_actions=args.num_actions, latent_size=args.latent_size).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    
    info = {k : torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir)*1e6
    if int(info['frames'][0]) == 0: printlog(args,'', end='', mode='w') # clear log file

    processes = []
    for rank in range(args.processes):
        p = mp_context.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start() ; processes.append(p)
    for p in processes:
        p.join()
