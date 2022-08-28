import torch
import torch.nn as nn
import torch.nn.functional as f


def mlp(sizes, activation, output_activation=nn.Identity):
    """Multi-Layer-Perceptron"""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def mlp_wn(sizes, activation, output_activation=nn.Identity):
    """Multi-Layer-Perceptron with Weight-normalization Layer"""
    layers = []
    for j in range(len(sizes) - 1):

        if j < len(sizes) - 2:
            act = activation
            nl = nn.LayerNorm(sizes[j + 1])  # torchgan.layers.VirtualBatchNorm(sizes[j+1], eps=1e-05)
        else:
            act = output_activation
            nl = nn.Identity()  # no vbn for output layer
        layers += [nn.Linear(sizes[j], sizes[j + 1]), nl, act()]
    return nn.Sequential(*layers)


class RNN(nn.Module):

    def __init__(self, input_shape, rnn_hidden_dim, action_dim):
        super(RNN, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)  # 64
        self.fc2 = nn.Linear(rnn_hidden_dim, action_dim)
        self.output = nn.Tanh()

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        u = self.output(q)
        return u, h

# no use
class ESTFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, vto_dim, hidden_size=(256, 256),
                 activation=nn.Sigmoid):
        super().__init__()
        self.est_net = mlp([obs_dim + action_dim] + list(hidden_size) + [vto_dim],
                           activation=activation)  # output activation is none

    def forward(self, obs, act):
        est_vto = self.est_net(torch.cat([obs, act], dim=-1))
        return est_vto

    def compute_est_loss(self, data):
        """
        :param data: ['target': with size of x_n]
        :return:
        """
        o, a, target = data['obs'], data['action'], data['vtor']
        estimation = self.forward(o, a)
        # MSE loss
        loss_est = ((estimation - target) ** 2).mean()

        return loss_est

    def compute_cost(self, obs, action_fun):
        # sum(k_i * E(x_i|s,a)
        weight = torch.tensor([0.8, 0.2, 0.2], requires_grad=False)
        output = self.forward(obs, action_fun)
        cost = torch.sum(output * weight, dim=-1)
        return cost

    def save_model(self, id):
        torch.save(self.est_net.state_dict(), f"../Evo_Stra/model/est_net_{id}.pth")
