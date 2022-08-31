# DCGAN-like generator and discriminator
import numpy as np
import torch
from tensorflow import keras
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import torch.nn.init as weight_init

channels = 3

def normalize_vector(x, eps=.0001):
    # Add epsilon for numerical stability when x == 0
    norm = torch.norm(x, p=2, dim=1) + eps
    return x / norm.expand(1, -1).t()



class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        self.conv0 = (nn.Conv2d(channels, 32, 3, stride=2, padding=(1,1)))
        self.batch0 = nn.BatchNorm2d(32)
         # Input: 40x40x?
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2, padding=(1,1))
        self.batch1 = nn.BatchNorm2d(64)
        # 40 x 40 x 64
        self.conv2 =  nn.Conv2d(64, 128, 4, stride=2, padding=(1,1))
        self.batch2 = nn.BatchNorm2d(128)
        # 20 x 20 x 128
        self.conv3 =  nn.Conv2d(128, 256, 4, stride=2, padding=(1,1))
        self.batch3 = nn.BatchNorm2d(256)
        # 10 x 10 x 256
        self.conv4 =  nn.Conv2d(256, 256, 4, stride=2, padding=(1,1))
        self.batch4 = nn.BatchNorm2d(256)
        # 5 x 5 x 256
        self.conv5 =  nn.Conv2d(256, 256, 3, stride=1, padding=(0,0))
        self.batch5 = nn.BatchNorm2d(256)
        # 3 x 3 x 256

        self.hidden_units = 3 * 3 * 256
        self.fc = nn.Linear(self.hidden_units, latent_size)



    def forward(self, x):
        #(hx, cx) = memory
        x = self.leaky(self.batch0(self.conv0(x)))
        x = self.leaky(self.batch1(self.conv1(x)))
        x = self.leaky(self.batch2(self.conv2(x)))
        x = self.leaky(self.batch3(self.conv3(x)))
        x = self.leaky(self.batch4(self.conv4(x)))
        x = self.leaky(self.batch5(self.conv5(x)))
        x = x.contiguous().view((-1, self.hidden_units))

        return self.fc(x)

def catv(x , y):
    bs = x.size(0)
    y = y.unsqueeze(2).unsqueeze(3)
    size_x = x.size(2)
    size_y = x.size(3)

    v_to_cat = y.expand(bs, y[0].size(0), size_x, size_y )

    return torch.cat([x,v_to_cat], dim = 1)

class Generator(nn.Module):
    def __init__(self, z_dim, action_size, pac_man=False):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        use_value = 0
        action_size += use_value
        
        self.fc = nn.Linear(z_dim + action_size, z_dim)
        self.deconv1 = nn.ConvTranspose2d(z_dim +  action_size, 512, 4, stride=2)
        self.batch1 = nn.BatchNorm2d(512)
        if pac_man:
            self.deconv2 = nn.ConvTranspose2d(512 + action_size, 256, 4, stride=3, padding=(1,1)) # 11
        else:
            self.deconv2 = nn.ConvTranspose2d(512 + action_size, 256, 4, stride=2, padding=0) # 10
        self.batch2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256 + action_size, 128, 4, stride=2, padding=(1,1)) #20
        self.batch3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128 + action_size , 128, 4, stride=2, padding=(1,1)) #40
        self.batch4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128 + action_size , 64, 4, stride=2, padding=(1,1))
        self.batch5 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64 + action_size , channels, 4, stride=2, padding=(1,1))


    def forward(self, x, y):
        x = F.relu(self.fc(torch.cat([x,y], dim = 1)))
        x = x.view((-1, self.z_dim, 1, 1))
        x = F.relu(self.batch1(self.deconv1(catv(x,y))))
        x = F.relu(self.batch2(self.deconv2(catv(x,y))))
        x = F.relu(self.batch3(self.deconv3(catv(x,y))))
        x = F.relu(self.batch4(self.deconv4(catv(x,y))))
        x = F.relu(self.batch5(self.deconv5(catv(x,y))))
        x = self.deconv6(catv(x,y))

        return torch.sigmoid(x)



class Discriminator(nn.Module):
    def __init__(self,latent_size, action_size):
        super(Discriminator, self).__init__()

        self.lin1 = nn.Linear(latent_size, latent_size)
        self.lin2 = nn.Linear(latent_size, latent_size)
        self.pi = nn.Linear(latent_size, action_size)
        self.v = nn.Linear(latent_size, 1)


   
    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(x)

        return F.softmax(self.pi(x), dim=1), self.v(x)

N = 32
def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x
#Encoder
class Q_net(nn.Module):
    def __init__(self, z_dim, agent_latent=32):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(agent_latent, N)
        self.bn1 = nn.BatchNorm1d(N)
        self.lin2 = nn.Linear(N, N)
        self.bn2 = nn.BatchNorm1d(N)
        self.lin3gauss = nn.Linear(N, z_dim)

    def forward(self, x):
        #x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        #x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        #x = F.relu(x)#leaky(x)
        xgauss = self.lin3gauss(x)
        return norm(xgauss)

# Decoder
class P_net(nn.Module):
    def __init__(self, z_dim, agent_latent=32):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.bn1 = nn.BatchNorm1d(N)
        self.lin2 = nn.Linear(N, N)
        self.bn2 = nn.BatchNorm1d(N)
        self.lin3 = nn.Linear(N, agent_latent)

    def forward(self, x):
        #x = self.lin1(x)
        #x = F.dropout(x, p=0.25, training=self.training)
        #x = F.relu(x)#leaky(x)
        #x = self.lin2(x)
        #x = F.dropout(x, p=0.25, training=self.training)
        #x = F.relu(x)#leaky(x)
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.lin3(x)
        return x

# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self, z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)#leaky(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)#leaky(x)
        return torch.sigmoid(self.lin3(x))


class Agent(torch.nn.Module): # an actor-critic neural network
    def __init__(self, num_actions, latent_size = 256):
        super(Agent, self).__init__()

        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 5 * 5, self.latent_size )
        self.critic_linear, self.actor_linear = nn.Linear(latent_size, 1), nn.Linear(latent_size, num_actions)

    def get_latent_size(self):
        return self.latent_size

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.linear(x.view(-1, 32 * 5 * 5))
        return x
        #return self.critic_linear(x), self.actor_linear(x)

    def pi(self, x):
        return self.actor_linear(x)

    def value(self, x):
        return self.critic_linear(x)


class KerasAgent(torch.nn.Module):
    def __init__(self, agent_file, latent_size=256, num_actions=9):
        super(KerasAgent, self).__init__()
        self.agent = keras.models.load_model(agent_file)
        nb_layers = len(self.agent.layers)
        if nb_layers == 9:
            # special case for dueling
            latent_layer_idx = 6
        else:
            latent_layer_idx = nb_layers - 2
        self.latent_size = latent_size

        self.latent_model = keras.models.Model(inputs=self.agent.input,
                                               outputs=self.agent.layers[latent_layer_idx].output)

        # transform the last layer to a pytorch layer because pytorch gradients are needed for gradient descent
        self.action_layer = nn.Linear(latent_size, num_actions)
        action_layer_weights = self.agent.layers[latent_layer_idx + 1].get_weights()[0]
        action_layer_biases = self.agent.layers[latent_layer_idx + 1].get_weights()[1]
        self.action_layer.weight.data = torch.from_numpy(np.transpose(action_layer_weights))
        self.action_layer.bias.data = torch.from_numpy(action_layer_biases)
        self.action_layer.cuda()

    def get_latent_size(self):
        return self.latent_size

    def forward(self, inputs):
        keras_inputs = inputs.detach().permute(0, 2, 3, 1).cpu().numpy()
        prediction = self.latent_model.predict(keras_inputs)
        return torch.from_numpy(prediction).cuda()

    def pi(self, x):
        # keras_x = x.detach().cpu().numpy()
        # prediction = self.action_model.predict(keras_x)
        # return torch.from_numpy(prediction).cuda()
        return self.action_layer(x)

    def value(self, x):
        raise NotImplementedError()


class ACER_Agent(nn.Module):
    """
    The ACER model as used by the openai-baselines repository.
    """

    def __init__(self, num_actions=5, latent_size=512):
        super(ACER_Agent, self).__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, num_actions)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def transpose_conv(self, x):
        """
        For the flatten output to be identical to the one in tensorflow, we have to transpose the output of the last
         conv layer. This should be equivalent to np.transpose(0,2,3,1)
        """
        x = torch.transpose(x, 1, 3)
        return torch.transpose(x, 1, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.transpose_conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        latent = self.relu(x)
        return latent

    def get_latent_size(self):
        return self.latent_size

    def pi(self, x):
        logits = self.fc2(x)
        return logits

    def value(self, x):
        raise NotImplementedError()
