import torch
import torch.nn as nn
from collections import OrderedDict


def weight_init(module):
    """Initialize layer weight based on Xavier normal
    Only supported layer types are nn.Linear and nn.LSTMCell

    Args:
        module (class): Layer to initialize weight, including bias
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        module.bias.data.zero_()

    if isinstance(module, nn.LSTMCell):
        nn.init.xavier_normal_(module.weight_ih)
        nn.init.xavier_normal_(module.weight_hh)
        module.bias_ih.data.zero_()
        module.bias_hh.data.zero_()


def to_numpy(tensor, dtype):
    """Convert torch.Tensor to numpy with specific dtype
    Args:
        tensor (tensor.Tensor): Tensor to change to numpy
        dtype (dtype): Target data type in numpy
    """
    return tensor.detach().cpu().numpy().astype(dtype)


def get_parameters(network):
    """Return parameters that consist of network

    Args:
        network (class): Network that consists of torch parameters or variables

    Returns:
       parameters (generator): Set of parameters that consist of network
    """
    if isinstance(network, OrderedDict):
        return network.values()
    elif isinstance(network, nn.Parameter):
        return [network]
    elif isinstance(network, torch.Tensor):
        return [network]
    else:
        return network.parameters()
