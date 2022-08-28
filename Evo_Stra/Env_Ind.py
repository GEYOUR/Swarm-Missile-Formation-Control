import os
import sys

from Common.Definition import ROOT_DIR

sys.path.append("..")
import numpy as np
import torch
from typing import Dict
from torch import nn
from Common.Config import get_args_from_json
from Evo_Stra.network import mlp, RNN

class Missile():

    def __init__(self, args=None):
        if args is None:
            self.args = get_args_from_json()
        else:
            self.args = args
        n_actions = self.args.action_dim  # continuous action
        obs_shape = self.args.obs_dim
        if self.args.Network_Type == "MLP":
            self.net = mlp(sizes=[obs_shape] + list(self.args.policy_hidden_size) + [n_actions],
                              activation=nn.Sigmoid,
                              output_activation=nn.Tanh)
        elif self.args.network_type == "RNN":
            self.rnn_hidden = torch.zeros(64, )
            self.net = RNN(obs_shape, 64, n_actions)

    @staticmethod
    def from_params(params: Dict[str, torch.Tensor], args=None):
        agent = Missile(args)
        agent.net.load_state_dict(params)
        return agent

    def get_params(self) -> Dict[str, torch.Tensor]:
        return self.net.state_dict()

    def save_params(self, id, iter):

        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        torch.save(self.net.state_dict(), os.path.join(ROOT_DIR, self.args.model_dir, f'evo_net_{id}_at_{iter}.pth'))

    def action(self, obs):
        """ Action from the network part
        :param obs:
        :return: ndarray
        """
        with torch.no_grad():
            if self.args.Network_Type == "RNN":
                action, self.rnn_hidden = self.net(
                    torch.tensor(obs, dtype=torch.float32).reshape(-1, self.args.obs_dim), self.rnn_hidden)
                return np.squeeze(action.detach().numpy())
            else:
                return self.net(torch.tensor(obs, dtype=torch.float32)).detach().numpy()

    def action_with_grad(self, obs):
        """
        :param obs:
        :return:
        """
        if self.args.network_type == "RNN":
            action, self.rnn_hidden = self.net(torch.tensor(obs, dtype=torch.float32))
            return action
        else:
            return self.net(torch.tensor(obs, dtype=torch.float32))


if __name__ == '__main__':
    missile = Missile()
    print(missile.action(np.zeros(missile.args.obs_dim, dtype=float)))
