import torch.nn as nn
import torch.nn.functional as F
from misc.torch_utils import weight_init

LOG_STD_MIN = -20.
LOG_STD_MAX = 2.


class GaussianMLP(nn.Module):
    """Actor network with LSTM that outputs action
    Args:
        n_input (int): Input dimension to network
        n_output (int): Output dimension of network
        name (str): Prefix for each layer
        log (dict): Dictionary that contains python logging
        args (argparse): Python argparse that contains arguments
        device (torch.device): Torch device to process tensors
    """
    def __init__(self, n_input, n_output, name, log, args, device) -> None:
        super(GaussianMLP, self).__init__()

        self.name = name

        setattr(self, name + "_l1", nn.Linear(n_input, args.n_hidden, device=device))
        setattr(self, name + "_l2", nn.Linear(args.n_hidden, args.n_hidden, device=device))
        setattr(self, name + "_l3_mu", nn.Linear(args.n_hidden, n_output, device=device))
        setattr(self, name + "_l3_var", nn.Linear(args.n_hidden, n_output, device=device))
        self.apply(weight_init)
        log[args.log_name].info("[{}] {}".format(name, self))

    def forward(self, x):
        x = getattr(self, self.name + "_l1")(x)
        x = F.relu(x)
        x = getattr(self, self.name + "_l2")(x)
        x = F.relu(x)
        mu = getattr(self, self.name + "_l3_mu")(x)
        logvar = getattr(self, self.name + "_l3_var")(x)

        return mu, logvar.clamp(min=LOG_STD_MIN, max=LOG_STD_MAX)
