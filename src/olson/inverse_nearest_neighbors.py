import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch import autograd
from torch.autograd import Variable
import model
from time import sleep

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.stats import entropy
import gym
from atari_data import MultiEnvironment, ablate_screen, prepro

from collections import deque
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

from collections import defaultdict




from scipy.ndimage.filters import gaussian_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--img_dir', type=str, default=None)

    parser.add_argument('--latent', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--wae_latent', type=int, default=128)
    parser.add_argument('--agent_latent', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--env', type=str, default='SpaceInvaders-v0')
    parser.add_argument('--enc_file', type=str, default=None)
    parser.add_argument('--gen_file', type=str, default=None)
    parser.add_argument('--Q', type=str, default="Q")
    parser.add_argument('--P', type=str, default="P")
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--agent_file', type=str, default="")
    parser.add_argument('--missing', type=str, default="")
    parser.add_argument('--frame_skip', type=int, default=7)
    parser.add_argument('--speed', type=float, default=.01)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--frames_to_cf', type=int, default=50)
    parser.add_argument('--cf_all_actions', type=int, default=0)
    parser.add_argument('--salient_intensity', type=int, default=500)
    parser.add_argument('--last_frame_diff', type=int, default=10)
    
    args = parser.parse_args()
    return args


#original: unchanged frame
#cf: the new counterfactual frame
#delta: change in a pixel required to make a notice
def get_changed_pixels(original, cf, delta=0.0001):
    diff = cf - original

    diff = np.abs(diff)
    diff[diff < delta] = 0
    diff = np.sum(diff, axis =2)
    max_diff = np.max(diff)
    #diff[diff > (max_diff/2)] = max_diff

    if max_diff > delta:
        diff = diff / max_diff 
    #added = np.sum(np.max(diff, delta), dim=2)
    #removed = np.sum(np.min(diff, delta), dim=2)

    #normalized_added = added / np.max(added)
    #normalized_removed = removed / np.min(removed)

    return diff




def saliency_on_atari_frame(saliency, atari, fudge_factor=330, channel=2, sigma=.75):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    #S = imresize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    S = saliency.astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min() 
    S = fudge_factor*pmax * S / S.max()
    I = atari.astype('uint16')
    I[35:195,:,channel] += S.astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I

def generate_saliency(atari, original, cf, salient_intensity):
    
    d_pixels = get_changed_pixels(original, cf)
    return saliency_on_atari_frame(d_pixels, atari, salient_intensity)

FONT_FILE = '/usr/local/eecsapps/cuda/cuda-10.0/jre/lib/fonts/LucidaSansRegular.ttf'
            
def immsave(file, pixels, text_to_add = "", size=200):
    np_img = imresize(pixels,size, interp = 'nearest')

    if text_to_add == "":
        imsave(file, np_img)
        return

    height_to_add = np.uint8(np_img.shape[0] / 8)
    width_to_add = np_img.shape[1]
    padding = np.zeros((height_to_add, width_to_add, 3))
    np_img = np.vstack([padding, np_img])

    img = Image.fromarray(np.uint8( np_img))
    d = ImageDraw.Draw(img)
    if os.path.isfile(FONT_FILE):
        fnt = ImageFont.truetype(FONT_FILE, np.uint8(height_to_add/3))
        d.text((0,0), text_to_add, font = fnt, fill=(255,255,255))
    else:
        d.text((0,0), text_to_add, fill=(255,255,255))

    img.save(file)



def printlog(s, img_dir, fname='log.txt', end='\n', mode='a'):
    print(s, end=end) 
    f=open(os.path.join(img_dir,fname),mode) 
    f.write(s+'\n') 
    f.close()

def get_low_entropy_states(agent, frames_to_cf, cur_envs, new_frame_bw, missing , end_frame):

    done = False
    i = 0
    entropies = []
    rewards=0

    env = gym.make("SpaceInvaders-v0") # make a local (unshared) environment
    env.unwrapped.frameskip = 7
    env.seed(13 )
    torch.manual_seed(13)
    img = ablate_screen(prepro(env.reset())[1], missing)
    state = Variable(torch.Tensor(img).view(1,1,80,80)).cuda()
    state_history = deque([state, state.clone(), state.clone(),state.clone()], maxlen=4)


    all_game_actions = defaultdict(int)    
    while done == False:
        i+=1
        state = torch.cat(list(state_history), dim=1)

        logit = agent.pi(agent(state))
        p = F.softmax(logit, dim=1)

        actions = p.max(1)[1].data.cpu().numpy()
        new_frame, reward, done, _ = env.step(actions)
        rewards += np.clip(reward, -1, 1)

        immsave(os.path.join("temp", "{:05d}.png".format(i)),new_frame)


        if env.unwrapped.ale.lives() < 3: done = True
        all_game_actions[actions[0]] +=1

        img = ablate_screen(prepro(new_frame)[1], missing)
        state_history.append(Variable(torch.Tensor(img).view(1,1,80,80)).cuda())

        probabilty_array = p.data.cpu().numpy()[0]
        cur_entropy = entropy(probabilty_array)
        entropies.append(cur_entropy)
        #print("{}, {}".format(i, cur_entropy))
    #exit()
    sorted_entropies = sorted(entropies[20:end_frame])

    for i in range(len(p[0])):
        all_game_actions[i] +=0
    return sorted_entropies[min(frames_to_cf, len(entropies))-1], all_game_actions

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def calculate_rank(ddict_ranks, a):
    sorted_values = sorted(ddict_ranks.values())
    ranking = sorted_values.index(ddict_ranks[a])
    #import pdb; pdb.set_trace()
    return ranking

def find_nearest_neighbor(z, a, nodes_list):

    ret_pic = None
    min_dist = 99999999999
    z_numpy = z[0].cpu().data.numpy()
    for node, pic, action in nodes_list:
        if action != a:
            continue

        cur_dist = np.linalg.norm(z_numpy-node)
        if cur_dist < min_dist:
            min_dist = cur_dist
            ret_pic = pic

    return ret_pic

def run_game(agent, frames, envs, seed, img_dir, salient_intensity, missing):
    states = []
    
    envs.seed(seed )  
    torch.manual_seed(seed)

    new_frame_rgb, new_frame_bw = envs.reset()


    action_description = envs.get_action_meanings()


    if seed == 13:
        min_entropy = 1.05
        end_frame = 396
    elif seed == 45:
        min_entropy = .815 
        end_frame = 446
    else:
        exit("missing correct seeds for user study explanations")


    #saves = envs.clone_full_state()
    #_, ddict_ranks =  get_low_entropy_states(agent, frames_to_cf, envs, new_frame_bw, missing, end_frame)
    #envs.restore_full_state(saves)
    #torch.manual_seed(seed)

    '''env = gym.make("SpaceInvaders-v0") # make a local (unshared) environment
    env.unwrapped.frameskip = 7
    env.seed(13 )
    torch.manual_seed(13)

    envs.envs[0] = env
    new_frame_rgb, new_frame_bw = envs.reset()
    import pdb; pdb.set_trace()'''
 
    agent_state = Variable(torch.Tensor(ablate_screen(new_frame_bw, missing)).cuda())
    agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)
    np.set_printoptions(precision=4)

    done = [False]
    i = 0
    cf_count = 0
    last_frame = -100
    last_frame_diff = 10
    ret = []
    while cf_count < len(frames):
        i+=1
        agent_state = torch.cat(list(agent_state_history), dim=1)#torch.cat(list(agent_state_history), dim=1)

        z_a = agent(agent_state)
       
        actions = F.softmax(agent.pi(z_a), dim=1).max(1)[1].data.cpu().numpy()
        atari_frame = envs.envs[0].render(mode='rgb_array')

        if i in frames:
            ret.append( (z_a[0].cpu().data.numpy(), new_frame_rgb[0], actions[0],atari_frame) )
            cf_count += 1

        new_frame_rgb, new_frame_bw, _, done, _ = envs.step(actions)

        agent_state_history.append(Variable(torch.Tensor(ablate_screen(new_frame_bw, missing)).cuda()))
    
    return ret

'''out_state = np.hstack(state[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())

immsave(img_dir + "/state_rgb{}.png".format(i), out_state)

        

for a in range(envs.get_action_size()):
    if a == actions[0]: continue

    if a <= 0: continue
    print("performing nn {} on action {}".format(cf_count, a))

    out_nn = find_nearest_neighbor(z_a, a, nodes_list)

    out_state = np.hstack(state[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())


    saliency_img = generate_saliency(atari_frame, out_state[:,480:640,:], out_nn, salient_intensity) /255

    #original input, saliency, CF
    demo_img = np.hstack([out_state[:,480:640,:], saliency_img[35:195,:], out_nn]) * 255
    text_to_add = "Original action a:                                                                 Saliency, Time Step:                                                      Counterfactual action a': "
    text_to_add2 = "\n{} {: <9}                                                                                 {:04d}                                                                             {} {}".format(actions[0], action_description[actions[0]], i, a, action_description[a])

    file_details = '{:04d}_action{}_cf{}{}.png'.format(i, action_description[actions[0]],  a,  action_description[a])
    file = img_dir + '/demo' + file_details #/demo_{:04d}_action{}r{}_cf{}r{}{}.png'.format(i, actions[0], calculate_rank(ddict_ranks, actions[0]), a,  calculate_rank(ddict_ranks, a), action_description[a])
    immsave(file, demo_img, text_to_add + text_to_add2)'''




def build_node_dict(agent, envs, seed, nodes_list, iters, missing):

    new_frame_rgb, new_frame_bw = envs.reset()

    agent_state = Variable(torch.Tensor(ablate_screen(new_frame_bw, missing)).cuda())
    agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)
   
    ret = []

    #hardcoded frame indices where the agent takes that action
    if seed == 13: #ablation
        fire_nodes =  {1} #1
        right_nodes = set()      #2
        left_nodes = {0} #3
        rightfire_nodes = {2,3,4,5} #4
        leftfire_nodes = {6,7,8,9} #5
    else: #original agent
        fire_nodes =  {2}
        right_nodes = {6,7}
        left_nodes = set()
        rightfire_nodes = {0,1,3,4,5,8,9}
        leftfire_nodes = set()

    #logic to ensure we try nn for every frame where agent doesnt take action "a"
    t = set(range(10))
    node_map = {
        0 : list(t - set()),
        1 : list(t - fire_nodes),
        2 : list(t - right_nodes),
        3 : list(t - left_nodes),
        4 : list(t - rightfire_nodes),
        5 : list(t - leftfire_nodes),
    }

    i = 0
    bs = envs.batch_size
    total_iters = iters / bs
    greedy = np.ones(bs).astype(int)

    closest_nodes = [(None,None,None,99999999)] * 10
    
    #import pdb; pdb.set_trace()
    while i < total_iters:
        i+=1
        agent_state = torch.cat(list(agent_state_history), dim=1)#torch.cat(list(agent_state_history), dim=1)

        z_a = agent(agent_state)
        logits = agent.pi(z_a)
        p = F.softmax(logits, dim=1)
        
        real_actions = p.max(1)[1].data.cpu().numpy()

        if np.random.random_sample() < 0.2:
            actions = np.random.randint(6, size=bs)
            actions = (real_actions * greedy) + (actions * (1-greedy))
        else:
            actions = real_actions

        z_numpy = z_a.cpu().data.numpy()
        for b in range(bs):

            for j in node_map[real_actions[b]]:

                if real_actions[b] == nodes_list[j][2]: continue

                cur_dist = np.linalg.norm(z_numpy[b]-nodes_list[j][0])
                
                if cur_dist < closest_nodes[j][3]:
                    closest_nodes[j] = (z_numpy[b], new_frame_rgb[b], real_actions[b],cur_dist)

        


        new_frame_rgb, new_frame_bw, _, done, _ = envs.step(actions)
        agent_state_history.append(Variable(torch.Tensor(ablate_screen(new_frame_bw, missing)).cuda()))
        if np.sum(done) > 0:
            for j, d in enumerate(done):
                if d:
                    greedy[d] = (np.random.rand(1)[0] > (1 - 0.2)).astype(int)
                    
        if i % 100 == 0:
            print("{} processed, {:.2f}% complete".format(i*bs, 100 * (i/ total_iters)))

    return closest_nodes


def main():
    #load models
    #load up an atari game
    #run (and save) every frame of the game
    #args = parse_args()

    args = parse_args()
    if args.missing == "none":
        args.seed = 45
        frames = [20, 30 ,40 ,65 ,84 ,100 ,111 ,136 ,146 ,193 ]
    elif args.missing == "agent":
        frames = [20, 30, 40, 50, 65, 76, 86, 170, 255, 313]
        args.seed = 13
    else:
        exit("bad missing param")


    MAX_ITERS = args.iters
    speed = args.speed
    frames_to_cf = args.frames_to_cf
    seed = args.seed
    img_dir = args.img_dir
    if img_dir == None:
        img_dir = "nn_imgs_miss-{}_{}".format(args.missing, args.iters)


    if not os.path.isfile(args.agent_file):
        args.agent_file = args.env + ".model.80.tar"
        if not os.path.isfile(args.agent_file):
            print("bad agent_file")
            exit()


    map_loc = {
            'cuda:0': 'cuda:'+str(args.gpu),
            'cuda:1': 'cuda:'+str(args.gpu),
            'cuda:2': 'cuda:'+str(args.gpu),
            'cuda:3': 'cuda:'+str(args.gpu),
            'cuda:4': 'cuda:'+str(args.gpu),
            'cuda:5': 'cuda:'+str(args.gpu),
            'cuda:7': 'cuda:'+str(args.gpu),
            'cuda:6': 'cuda:'+str(args.gpu),
            'cuda:8': 'cuda:'+str(args.gpu),
            'cuda:9': 'cuda:'+str(args.gpu),
            'cuda:10': 'cuda:'+str(args.gpu),
            'cuda:11': 'cuda:'+str(args.gpu),
            'cuda:12': 'cuda:'+str(args.gpu),
            'cuda:13': 'cuda:'+str(args.gpu),
            'cuda:14': 'cuda:'+str(args.gpu),
            'cuda:15': 'cuda:'+str(args.gpu),
            'cpu': 'cpu',
    }

    if args.frame_skip % 2 ==0 and args.env == 'SpaceInvaders-v0':
        print("SpaceInvaders needs odd frameskip due to bullet alternations")
        args.frame_skip = args.frame_skip - 1

    #run every model on all frames (4*n frames))
    print('Loading model...')
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    #number of updates to discriminator for every update to generator
    envs = MultiEnvironment(args.env, 1, args.frame_skip)

    agent = model.Agent(envs.get_action_size(), args.agent_latent).cuda() #cuda is fine here cause we are just using it for perceptual loss and copying to discrim
    agent.load_state_dict(torch.load(args.agent_file))

    os.makedirs(img_dir, exist_ok=True)
    
    print('finished loading models: running game')
    
    #run_game(encoder, generator, agent, Q, P, envs, seed, img_dir, frames_to_cf, speed, MAX_ITERS, args.cf_all_actions, args.salient_intensity, args.last_frame_diff)


    nodes_list = run_game(agent, frames, envs, seed, img_dir, args.salient_intensity, args.missing)
    

    envs = MultiEnvironment(args.env, args.batch_size, args.frame_skip)

    action_description = envs.get_action_meanings()
    print("finding nearest neighbors")
    nn_nodes = build_node_dict(agent, envs, seed, nodes_list, args.iters, args.missing)
    for i in range(10):
        atari_frame = nodes_list[i][3]
        original_state = nodes_list[i][1]
        nn_state = nn_nodes[i][1]

        actions = [nodes_list[i][2]]
        a = nn_nodes[i][2]

        saliency_img = generate_saliency(atari_frame, original_state, nn_state, args.salient_intensity) /255

        #original input, saliency, CF
        demo_img = np.hstack([original_state, saliency_img[35:195,:], nn_state]) * 255
        text_to_add = "Original action a:                                                                 Saliency, Time Step:                                                      Counterfactual action a': "
        text_to_add2 = "\n{} {: <9}                                                                                 {:04d}                                                                             {} {}".format(actions[0], action_description[actions[0]], i, a, action_description[a])

        file_details = '{:04d}_action{}_cf{}{}.png'.format(frames[i], action_description[actions[0]],  a,  action_description[a])
        file = img_dir + '/demo' + file_details #/demo_{:04d}_action{}r{}_cf{}r{}{}.png'.format(i, actions[0], calculate_rank(ddict_ranks, actions[0]), a,  calculate_rank(ddict_ranks, a), action_description[a])
        immsave(file, demo_img, text_to_add + text_to_add2)
        immsave(img_dir + '/output1_' + file_details, original_state* 255)
        immsave(img_dir + '/output2_' + file_details, saliency_img[35:195,:]* 255)
        immsave(img_dir + '/output3_' + file_details, nn_state* 255)




if __name__ == '__main__':
    main()

