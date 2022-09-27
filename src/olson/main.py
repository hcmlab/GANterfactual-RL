import argparse

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import src.olson.model as model
from src.star_gan.data_loader import get_loader
from src.util import restrict_tf_memory, array_to_pil_format, denorm

matplotlib.use('Agg')
import os
#import logutil
import time
from src.olson.atari_data import MultiEnvironment, ablate_screen, prepro_dataset_batch

os.environ['OMP_NUM_THREADS'] = '1'

from collections import deque

#ts = logutil.TimeSeries('Atari Distentangled Auto-Encoder')


restrict_tf_memory()


print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epsilon', type=float, default=.2)
parser.add_argument('--lr', type=float, default=1e-4)
# Output directory of the model that creates counterfactual states
parser.add_argument('--checkpoint_dir', type=str, default='../../res/models/SpaceInvaders_Abl_Olson')
parser.add_argument('--epochs', type=int, default=300)

parser.add_argument('--latent', type=int, default=16)
parser.add_argument('--wae_latent', type=int, default=128)
parser.add_argument('--agent_latent', type=int, default=32)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--env', type=str, default="SpaceInvadersNoFrameskip-v4")
# Directory of the used wasserstein encoder
parser.add_argument('--Q', type=str, default="../../res/models/SpaceInvaders_Abl_Olson_wae/Q")
parser.add_argument('--P', type=str, default="../../res/models/SpaceInvaders_Abl_Olson_wae/P")
parser.add_argument('--missing', type=str, default="agent")
# Directory of the RL-Agent that should be explained
parser.add_argument('--agent_file', type=str, default="../../res/agents/abl_agent.tar")
parser.add_argument('--enc_lam', type=float, default=5)
parser.add_argument('--clip', type=float, default=.0001)
parser.add_argument('--gen_lam', type=float, default=.5)
parser.add_argument('--starting_epoch', type=int, default=0)
parser.add_argument('--info', type=str, default="")
parser.add_argument('--cf_loss', type=str, default="None")
parser.add_argument('--m_frames', type=int, default=40)
parser.add_argument('--fskip', type=int, default=4)
parser.add_argument('--use_agent', type=int, default=1)
parser.add_argument('--gpu', type=int, default=7)

parser.add_argument('--use_dataset', type=bool, default=True)
# The dataset used for training the model
parser.add_argument('--dataset_dir', type=str, default="../../res/datasets/SpaceInvaders_Abl")
parser.add_argument('--img_size', type=str, default=160)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--action_size', type=int, default=6)
parser.add_argument('--is_pacman', type=bool, default=False)


args = parser.parse_args()

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


if __name__ == '__main__':
    action_size = args.action_size
    if not args.use_dataset:
        print('Initializing OpenAI environment...')
        if args.fskip % 2 == 0 and args.env == 'SpaceInvaders-v0':
            print("SpaceInvaders needs odd frameskip due to bullet alternations")
            args.fskip = args.fskip -1

        envs = MultiEnvironment(args.env, args.batch_size, args.fskip)

    print('Building models...')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device('cuda:0')
    if not (os.path.isfile(args.agent_file) and  os.path.isfile(args.agent_file) and  os.path.isfile(args.agent_file)):
        print("need an agent file")
        exit()
        args.agent_file = args.env + ".model.80.tar"

    if args.agent_file.endswith(".h5"):
        agent = model.KerasAgent(args.agent_file, num_actions=action_size, latent_size=args.agent_latent)
    elif args.agent_file.endswith(".pt"):
        agent = model.ACER_Agent(num_actions=action_size, latent_size=args.agent_latent).cuda()
        agent.load_state_dict(torch.load(args.agent_file))
    else:
        agent = model.Agent(action_size, args.agent_latent).cuda()
        agent.load_state_dict(torch.load(args.agent_file, map_location=map_loc))

    Z_dim = args.latent
    wae_z_dim = args.wae_latent

    encoder = model.Encoder(Z_dim).cuda()
    generator = model.Generator(Z_dim, action_size, pac_man=args.is_pacman).cuda()
    discriminator = model.Discriminator(Z_dim, action_size).cuda()
    Q = model.Q_net(args.wae_latent, agent_latent=args.agent_latent).cuda()
    P = model.P_net(args.wae_latent, agent_latent=args.agent_latent).cuda()


    Q.load_state_dict(torch.load(args.Q, map_location="cuda:0"))
    P.load_state_dict(torch.load(args.P, map_location="cuda:0"))
    encoder.train()
    generator.train()
    discriminator.train()
    Q.eval()
    P.eval()

    optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
    optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))

    print('finished building model')

    def zero_grads():
        optim_enc.zero_grad()
        optim_gen.zero_grad()
        optim_disc.zero_grad()

    bs =  args.batch_size
    enc_lambda = args.enc_lam

    clip = torch.tensor(args.clip, dtype=torch.float).cuda()

    TINY = 1e-9

    def model_step(state, p):
        #get variables
        z = encoder(state)
        reconstructed = generator(z, p)
        disc_pi, _ = discriminator(z)

        #different loss functions
        ae_loss, enc_loss_pi = autoencoder_step(p, disc_pi + TINY, reconstructed, state)
        disc_loss_pi = disc_step(z.detach(), p)

        return ae_loss, enc_loss_pi, disc_loss_pi

    loss = nn.NLLLoss()
    def autoencoder_step(p, disc_pi, reconstructed, state):
        zero_grads()

        #disentanglement loss and L2 loss
        #enc_loss =  enc_lambda * (1 -((p, value - disc_labels)**2).mean())
        #disc_pi, disc_v = disc_labels
        #enc_loss_v = (((disc_v - running_average)/value)**2).mean()
        enc_loss_pi = 1 + (disc_pi * torch.log(disc_pi)).mean()


        ae_loss = .5*torch.sum(torch.max((reconstructed - state)**2, clip) ) / bs
        enc_loss = enc_lambda*(enc_loss_pi)
        (enc_loss + ae_loss).backward()

        optim_enc.step()
        optim_gen.step()

        return ae_loss.item(), enc_loss_pi.item()


    def disc_step(z, p):
        zero_grads()

        disc_pi, _ = discriminator(z)

        #disc_loss_v  = (((real_v - disc_v)/real_v)**2).mean()
        disc_loss_pi = ((p - disc_pi) **2).mean()

        (disc_loss_pi).backward()
        optim_disc.step()

        return disc_loss_pi.item()




    #running_average = torch.zeros(1)
    mil = 1000000
    def train(epoch, data_loader):
        #import pdb; pdb.set_trace()
        data_iter = iter(data_loader)
        if args.use_dataset:
            x_real, label_org = next(data_iter)
            batch = array_to_pil_format(denorm(x_real).detach().permute(0, 2, 3, 1).cpu().numpy())
            new_frame_rgb, new_frame_bw = prepro_dataset_batch(batch, pacman=args.is_pacman)
            done = False
        else:
            new_frame_rgb, new_frame_bw = envs.reset()

        #agent_state = Variable(torch.Tensor(new_frame.mean(3)).unsqueeze(1)).cuda()
        #agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)

        state = Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda()


        agent_state = Variable(torch.Tensor(ablate_screen(new_frame_bw, args.missing)).cuda())
        agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)

        actions_size = args.action_size
        greedy = np.ones(args.batch_size).astype(int)

        #global running_average
        #running_average = agent.value(agent(torch.cat(list(agent_state_history), dim=1))).mean().detach() if running_average.mean().item() == 0 else running_average

        fs = 0
        for i in range(int( mil / bs)):
            agent_state = torch.cat(list(agent_state_history), dim=1)#torch.cat(list(agent_state_history), dim=1)

            z_a = agent(agent_state)
            #value = agent.value(z_a).detach()
            p = F.softmax(agent.pi(z_a), dim=1)
            real_actions = p.max(1)[1].data.cpu().numpy()
            #running_average = (running_average *.999) + (value.mean() * .001)
            #if fs >= 10000:
            #    import pdb; pdb.set_trace()

            #loss functions
            ae_loss, enc_loss_pi, disc_loss_pi = model_step(state, p.detach())

            #import pdb; pdb.set_trace()
            if args.use_dataset:
                # getting new images and creating a new iter if the last iteration is complete
                try:
                    x_real, label_org = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, label_org = next(data_iter)
                if len(x_real) != bs:
                    data_iter = iter(data_loader)
                    x_real, label_org = next(data_iter)
                batch = array_to_pil_format(denorm(x_real).detach().permute(0, 2, 3, 1).cpu().numpy())
                new_frame_rgb, new_frame_bw = prepro_dataset_batch(batch, pacman=args.is_pacman)
            else:
                if np.random.random_sample() < args.epsilon:
                    actions = np.random.randint(actions_size, size=bs)
                    actions = (real_actions * greedy) + (actions * (1 - greedy))
                else:
                    actions = real_actions
                new_frame_rgb, new_frame_bw, _, done, _ = envs.step(actions)

            agent_state_history.append(Variable(torch.Tensor(ablate_screen(new_frame_bw, args.missing)).cuda()))
            state = Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda()

            if np.sum(done) > 0:
                for j, d in enumerate(done):
                    if d:
                        greedy[j] = (np.random.rand(1)[0] > (1 - args.epsilon)).astype(int)


            if i % 20 == 0:
                print("Recon: {:.3f} --Enc entropy: {:.3f} --disc pi: {:.3f}".format(ae_loss, enc_loss_pi, disc_loss_pi))
                if i % 300 == 0:
                    fs = (i * args.batch_size) + (epoch * mil)
                    #print("running running_average is: {}".format(running_average.item()))
                    print("{} frames processed. {:.2f}% complete".format(fs , 100* (fs / (args.m_frames * mil))))




    def save_models(epoch):
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen{}'.format(epoch)))
        torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, 'enc{}'.format(epoch)))
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc{}'.format(epoch)))

    def main():
        data_loader = get_loader(args.dataset_dir, None, None, crop_size=args.img_size, image_size=args.img_size,
                                 batch_size=args.batch_size, dataset='RaFD', mode='train', num_workers=1,
                                 image_channels=args.img_channels)

        print('creating directories')
        if args.checkpoint_dir == '':
            args.checkpoint_dir = "{}_agent{}_lambda-e{}_missing-{}_fskip{}".format(args.info , args.agent_file, args.enc_lam, args.missing, args.fskip)

        os.makedirs(args.checkpoint_dir , exist_ok=True)


        for i in range(args.m_frames):
            img_dir = os.path.join(args.checkpoint_dir, "imgs"+str(i))
            os.makedirs(img_dir, exist_ok=True)
            encoder.train()
            generator.train()
            train(i, data_loader)

            print("saving models to directory: {}". format(args.checkpoint_dir))

            if i % 5 == 4 or i == args.m_frames -1:
                save_models(i)#args.m_frames)
            print("Evaluating the Autoencoder")
            # test_env = MultiEnvironment(args.env, 1, args.fskip)

            # top_entropy_counterfactual.run_game(encoder, generator, agent, Q, P, test_env, args.seed, img_dir, args.missing, frames_to_cf = 50)

        #test_encoding.eval_autoencoder(envs, img_dir, encoder, generator, Z_dim, args.agent_file)


    start = time.time()
    main()
    print((time.time() - start) / 60)
