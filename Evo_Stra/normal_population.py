from typing import Iterable, Dict, Callable, Union

import numpy as np
import torch
import torch as t
import torch.distributions as d

from Common.Config import get_args_from_json
from Evo_Stra.population import Population, Individual


class NormalPopulation(Population):
    """
    A distribution over individuals whose parameters are sampled from normal distributions
    """
    def __init__(self,
                 individual_parameter_shapes: Dict[str, t.Size],
                 individual_constructor: Callable[[Dict[str, t.Tensor]], Individual],
                 std: Union[float, str],
                 mirror_sampling: bool = True,
                 device='cpu'
                 ):
        """
        A distribution over individuals whose parameters are sampled from normal distributions

        The individuals are sampled by sampling their parameters from independent normal distributions and then calling individual_constructor with the sampled parameters.

        :param individual_parameter_shapes: The shapes of the parameters of an individual.
        :param individual_constructor: A function that constructs an individual from parameters (with shapes equal to individual_parameter_shapes).
        :param std: The standard deviation of the normal distributions.
        If a float, it is treated as a constant hyper-parameter. Equivalent to OpenAI ES [1].
        If it's a str it must be either 'shared' or 'diagonal':
            If it's 'shared' all parameters will share a single learned std deviation.
            if it's 'diagonal' each parameter will have their own learned std deviation, similar to PEPG [2].
        :param mirror_sampling: Whether or not individuals are sampled in pairs symmetrically around the mean. See [1].
        [1] - Salimans, Tim, et al. "Evolution strategies as a scalable alternative to reinforcement learning." arXiv preprint arXiv:1703.03864 (2017).
        [2] - Sehnke, Frank, et al. "Parameter-exploring policy gradients." Neural Networks 23.4 (2010): 551-559.
        """
        assert type(std) in {float, str}, "std must be a float or str"
        if type(std) == float:
            assert std > 0.0, "std must be greater than 0"
            self.param_logstds = {k: t.log(t.scalar_tensor(std, device=device)) for k in individual_parameter_shapes.keys()}
        if type(std) == str:
            assert std in {'shared', 'diagonal'}, "std must be 'shared' or 'diagonal'"
            if std == 'shared':
                self.shared_log_std = t.scalar_tensor(0.0, requires_grad=True, device=device)
                self.param_logstds = {k: self.shared_log_std for k in individual_parameter_shapes.keys()}
            else:
                self.param_logstds = {k: t.zeros(shape, requires_grad=True, device=device) for k, shape in individual_parameter_shapes.items()}

        self.std = std
        # self.param_means = {
        #     k: torch.zeros(shape, requires_grad=True) for k, shape in individual_parameter_shapes.items()}
        # xiver initialization
        self.param_means = {k: d.uniform.Uniform(-np.sqrt(1.0)/np.sqrt(sum(shape)), np.sqrt(1.0)/np.sqrt(sum(shape))).sample(shape).requires_grad_() for k, shape in individual_parameter_shapes.items()} # mean of each parameter is set to be 0
        self.constructor = individual_constructor
        self.mirror_sampling = mirror_sampling

        # Init Est function
        self.args = get_args_from_json()
        # self.est_fun = ESTFunction(args.obs_dim, args.n_actions, vto_dim=args.vto_dim,
        #                            hidden_size=args.est_hidden_size)
        # # add optimizer
        # self.est_optimizer = Adam(self.est_fun.parameters(), lr=args.est_lr)

        # buffer
        # self.buffer = ReplayBuffer(args.obs_dim, args.n_actions, args.vto_dim)

        # record
        # self.pi_update_num = 0
        # self.observer_loss_record = []
        # self.pi_loss_record = []

    def parameters(self) -> Iterable[t.Tensor]:
        # pass `self.param_means` to optimizer
        if type(self.std) == float:
            std_params = []
        else:
            if self.std == 'shared':
                std_params = [self.shared_log_std]
            else:
                std_params = list(self.param_logstds.values())

        mean_params = list(self.param_means.values())
        return mean_params + std_params

    def sample(self, n) -> Iterable[Individual]:
        assert not self.mirror_sampling or n % 2 == 0, "if mirror_sampling is true, n must be an even number"
        # sample n individuals with param \theta + noise
        # return [individual , log_prob(Θ+ε)]
        n_samples = n // 2 if self.mirror_sampling else n

        # std decay | I think std decay is still beneficial.
        for k in self.param_logstds.keys():
            self.param_logstds[k] += np.log(1 - self.args.sigma_decay)

        for i in range(n_samples):
            noise = {k: d.Normal(loc=t.zeros_like(v), scale=t.exp(self.param_logstds[k])).sample() for k, v in self.param_means.items()}
            yield (
                self.constructor({k: self.param_means[k] + n for k, n in noise.items()}),
                sum([d.Normal(self.param_means[k], scale=t.exp(self.param_logstds[k])).log_prob((self.param_means[k] + n).detach()).sum() for k, n in noise.items()]),
                [torch.exp(d.Normal(self.param_means[k], scale=t.exp(self.param_logstds[k])).log_prob((self.param_means[k] + n).detach())) for k, n in noise.items()]
            )
            if self.mirror_sampling:
                yield (
                    self.constructor({k: self.param_means[k] - n for k, n in noise.items()}),
                    sum([d.Normal(self.param_means[k], scale=t.exp(self.param_logstds[k])).log_prob((self.param_means[k] - n).detach()).sum() for k, n in noise.items()]),
                    [torch.exp(d.Normal(self.param_means[k], scale=t.exp(self.param_logstds[k])).log_prob((self.param_means[k] + n).detach())) for k, n in noise.items()]
                )

    def save_model(self, id, iter):
        individual = self.constructor(self.param_means)
        individual.save_params(id, iter)

    def _DEEPCOPY(self, pop_args):
        """
        :return: A new population with completely independent parameters
        """
        new_pop = NormalPopulation(*pop_args)
        new_pop.param_logstds = {k: t.clone(v).detach() for k, v in self.param_logstds.items()}
        # copy gradient information
        for k, v in self.param_means.items():
            new_param = t.clone(v).detach()
            new_param.grad = t.clone(v.grad)
            new_pop.param_means[k] = new_param
        return new_pop
