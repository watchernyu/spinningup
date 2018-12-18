import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon


    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        ## TODO not sure how exactly this part works
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        Implement: tanh(mu + sigma * eksee)
        with eksee~N(0,1)
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal( ## this part is eksee~N(0,1)
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample()
        )
        # z.requires_grad_() ## TODO this line probably not useful?

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            hidden_init=fanin_init,
            b_init_value=0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        ## here we use ModuleList so that the layers in it can be
        ## detected by .parameters() call
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc_layer.weight)
            fc_layer.bias.data.fill_(b_init_value)
            self.hidden_layers.append(fc_layer)

        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output

class TanhGaussianPolicy(Mlp):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            init_w=1e-3,
            hidden_activation=F.relu,
            hidden_init=fanin_init,
    ):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            init_w=init_w,
            hidden_activation=hidden_activation,
            hidden_init=hidden_init
        )

        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        ## this is the layer that gives log_std
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

    def get_env_action(self, obs_np, action_limit, deterministic=False):
        ## convert observations to pytorch tensors first
        ## and then use the forward method
        obs_tensor = torch.Tensor(obs_np).unsqueeze(0)
        action_tensor = self.forward(obs_tensor, deterministic=deterministic)[0].detach()
        ## convert it into the form that can put into the env
        ## action_limit here is for scaling the action from range (-1,1) to, for example, range (-3,3)
        action_np = action_tensor.numpy().reshape(-1) * action_limit
        return action_np

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=True,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        mean = self.last_fc_layer(h)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample( ## rsample means there is gradient
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample( ## sample means there is no gradient
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, std, pre_tanh_value,
        )

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def soft_update_model1_with_model2(model1, model2, rou):
    """
    see openai spinup sac psudocode line 16
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(
            rou*model1_param.data + (1-rou)*model2_param.data
        )
