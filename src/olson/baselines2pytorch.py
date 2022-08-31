""" Converts a model saved in openai baselines to a pytorch state_dict"""

import torch

import numpy as np
import joblib
import os


def conv_weight_transpose(x):
    x = np.transpose(x,(3,2,0,1))
    return torch.Tensor(x)


def conv_bias(x):
    x = np.squeeze(x)
    return torch.Tensor(x)


def fc_weight_transpose(x):
    x = np.transpose(x, (1,0))
    return torch.Tensor(x)


def fc_bias(x):
    return torch.Tensor(x)


def baselines_to_pytorch(path):
    """ Converts the saved openai-baselines weights in the *path* into a pytorch state_dict"""

    variables_dict = joblib.load(os.path.expanduser(path))

    torch_state_dict = {"conv1.weight": conv_weight_transpose(variables_dict["acer_model/pi/c1/w:0"]),
                        "conv1.bias": conv_bias(variables_dict["acer_model/pi/c1/b:0"]),
                        "conv2.weight": conv_weight_transpose(variables_dict["acer_model/pi/c2/w:0"]),
                        "conv2.bias": conv_bias(variables_dict["acer_model/pi/c2/b:0"]),
                        "conv3.weight": conv_weight_transpose(variables_dict["acer_model/pi/c3/w:0"]),
                        "conv3.bias": conv_bias(variables_dict["acer_model/pi/c3/b:0"]),
                        "fc1.weight": fc_weight_transpose(variables_dict["acer_model/pi/fc1/w:0"]),
                        "fc1.bias": fc_bias(variables_dict["acer_model/pi/fc1/b:0"]),
                        "fc2.weight": fc_weight_transpose(variables_dict["acer_model/pi/w:0"]),
                        "fc2.bias": fc_bias(variables_dict["acer_model/pi/b:0"])}

    torch.save(torch_state_dict, path + ".pt")


if __name__ == "__main__":
    baselines_to_pytorch(r"../../res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3")