import datetime
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tensorflow.python.estimator import keras
from torchvision.utils import save_image

from src.atari_wrapper import AtariWrapper
from src.star_gan.model import Discriminator
from src.star_gan.model import Generator
from src.util import restrict_tf_memory, get_agent_prediction, load_baselines_model
import src.olson.model as olson_model

from baselines.common.tf_util import adjust_shape

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Extensions
        torch.cuda.empty_cache()
        self.image_channels = config.image_channels
        self.agent_type = config.agent_type
        if config.agent_path is None:
            self.agent = None
        elif self.agent_type == "deepq":
            restrict_tf_memory()
            self.pacman = True
            self.agent = keras.models.load_model(config.agent_path)
        elif self.agent_type == "olson":
            restrict_tf_memory()
            self.pacman = False
            self.agent = olson_model.Agent(config.c_dim, 32).cuda()
            self.agent.load_state_dict(torch.load(config.agent_path, map_location=lambda storage, loc: storage))
        elif self.agent_type == "acer":
            restrict_tf_memory()
            self.pacman = True
            self.agent = load_baselines_model(config.agent_path, num_actions=5, num_env=self.batch_size)
        else:
            raise NotImplementedError("Known agent-types are: deepq, olson and acer")
        self.lambda_counter = config.lambda_counter
        self.counter_mode = config.counter_mode
        self.selective_counter = config.selective_counter
        self.ablate_agent = config.ablate_agent

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()


    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.image_channels, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.image_channels, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            # 2 for mask vector.
            self.G = Generator(self.g_conv_dim, self.image_channels, self.c_dim+self.c2_dim+2, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.image_channels, self.c_dim+self.c2_dim,
                                   self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from src.star_gan.logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def preprocess_batch_for_agent(self, batch):
        batch = self.denorm(batch)
        batch = batch.detach().permute(0, 2, 3, 1).cpu().numpy()

        preprocessed_batch = []
        for i, frame in enumerate(batch):
            frame = (frame * 255).astype(np.uint8)
            if self.agent_type == "deepq":
                frame = AtariWrapper.preprocess_frame(frame)
            elif self.agent_type == "acer":
                frame = AtariWrapper.preprocess_frame_ACER(frame)
            else:
                frame = AtariWrapper.preprocess_space_invaders_frame(frame, ablate_agent=self.ablate_agent)

            frame = np.squeeze(frame)
            stacked_frames = np.stack([frame for _ in range(4)], axis=-1)
            if not self.pacman:
                stacked_frames = AtariWrapper.to_channels_first(stacked_frames)
            preprocessed_batch.append(stacked_frames)
        return np.array(preprocessed_batch)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start models...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake
            if self.lambda_rec != 0 or self.lambda_cls != 0 or self.lambda_gp != 0:
                d_loss += self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake
                if self.lambda_rec != 0 or self.lambda_cls != 0 or self.lambda_gp != 0:
                    g_loss += self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls

                # Counter loss - only calculate and add if the target domain is not the original domain, otherwise
                # it would counteract the reconstruction loss in these cases
                if self.agent is not None: # and label_org[0] != label_trg[0]:
                    # Permute to get channels last
                    x_fake_keras = self.preprocess_batch_for_agent(x_fake)
                    if self.agent_type == "deepq":
                        agent_prediction = self.agent.predict_on_batch(x_fake_keras)
                    elif self.agent_type == "acer":
                        sess = self.agent.step_model.sess
                        feed_dict = {self.agent.step_model.X: adjust_shape(self.agent.step_model.X, x_fake_keras)}
                        agent_prediction = sess.run(self.agent.step_model.pi, feed_dict)
                    else:
                        torch_state = torch.Tensor(x_fake_keras).cuda()
                        agent_prediction = torch.softmax(self.agent.pi(self.agent(torch_state)).detach(), dim=-1)
                        agent_prediction = agent_prediction.cpu().numpy()
                    if isinstance(agent_prediction, list):
                        # extract action distribution in case the agent is a dueling DQN
                        agent_prediction = agent_prediction[0]
                    agent_prediction = torch.from_numpy(agent_prediction)

                    if self.selective_counter:
                        # filter samples in order to only calculate the counter-loss on samples
                        # where label_trg != label_org
                        relevant_samples = (label_trg != label_org).nonzero(as_tuple=True)
                        relevant_agent_prediction = agent_prediction[relevant_samples]
                        relevant_c_trg = c_trg[relevant_samples].cpu()
                        relevant_label_trg = label_trg[relevant_samples].cpu()
                    else:
                        relevant_agent_prediction = agent_prediction
                        relevant_c_trg = c_trg.cpu()
                        relevant_label_trg = label_trg.cpu()

                    # using advantage/softmax since the DQN output is not softmax-distributed
                    if self.counter_mode == "raw":
                        # mse
                        g_loss_counter = torch.mean(torch.square(relevant_agent_prediction - relevant_c_trg))
                    elif self.counter_mode == "softmax":
                        fake_action_softmax = torch.softmax(relevant_agent_prediction, dim=-1)
                        # mse
                        g_loss_counter = torch.mean(torch.square(fake_action_softmax - relevant_c_trg))
                    elif self.counter_mode == "advantage":
                        # convert Q-values to advantage values
                        mean_q_values = torch.mean(relevant_agent_prediction, dim=-1)
                        advantages = torch.empty(relevant_agent_prediction.size())
                        for action in range(self.c_dim):
                            action_q_values = relevant_agent_prediction[:, action]
                            advantages[:, action] = action_q_values - mean_q_values

                        # perform softmax counter loss on advantage
                        advantage_softmax = torch.softmax(advantages, dim=-1)
                        g_loss_counter = torch.mean(torch.square(advantage_softmax - relevant_c_trg))
                    elif self.counter_mode == "z-score":
                        trg_action_q_values = torch.gather(relevant_agent_prediction, 1,
                                                           torch.unsqueeze(relevant_label_trg, dim=-1))
                        fake_action_z_score = (trg_action_q_values - torch.mean(relevant_agent_prediction, dim=-1)) / \
                                              torch.std(relevant_agent_prediction, dim=-1)
                        g_loss_counter = -torch.mean(fake_action_z_score)
                    else:
                        raise NotImplementedError("Known counter-modes are: 'raw', 'softmax', 'advantage' and"
                                                  "'z-score'")

                    g_loss += self.lambda_counter * g_loss_counter

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                if self.agent is not None: # and label_org[0] != label_trg[0]:
                    loss['G/loss_counter'] = g_loss_counter.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    if self.image_channels == 4:
                        x_fake_list = [torch.unsqueeze(x_fixed[:, -1, :, :], dim=1)]
                    elif self.image_channels == 12:
                        x_fake_list = [x_fixed[:, 9:, :, :]]
                    else:
                        x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        if self.image_channels == 4:
                            x_fake_list.append(torch.unsqueeze(self.G(x_fixed, c_fixed)[:, -1, :, :], dim=1))
                        elif self.image_channels == 12:
                            x_fake_list.append(self.G(x_fixed, c_fixed)[:, 9:, :, :])
                        else:
                            x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.png'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
