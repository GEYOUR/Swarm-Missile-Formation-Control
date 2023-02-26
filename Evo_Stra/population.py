import torch
import torch as t
from torch.optim import Adam
from typing import Iterable, Callable, Tuple
from torch.multiprocessing import Pool

import sys
sys.path.append("./")
from Evo_Stra.Env_Ind import Missile

class Population:
    """
    A parameterized distribution over individuals.

    Meant to be optimized with `torch.optims` optimizers, as follows:
    Each POP has its own est_fun & policy_fun( which uses its own parameters)
    Each POP has own Replay buffer.
    """
    def parameters(self) -> Iterable[t.Tensor]:
        """
            :return: The parameters of this population distribution.
            """

        raise NotImplementedError

    def sample(self, n):
        """
        Sample n individuals and compute their log probabilities. The log probability computation MUST be differentiable.

        :param n: How many individuals to sample
        :return: n individuals and their log probability of being sampled: [(ind_1, log_prob_1), ..., (ind_n, log_prob_n)]
        """
        raise NotImplementedError

    # def get_action(self, obs):
    #     """ get action from param_mean"""
    #     agent = self.constructor(self.param_means)
    #     action = agent(obs)
    #     return action

    def fitness_grads(
            self,
            n_samples: int,
            pool: Pool = None,
            fitness_shaping_fn: Callable[[Iterable[float]], Iterable[float]] = lambda x: x
    ):
        """
        Computes the (approximate) gradients of the expected fitness of the population.

        Uses torch autodiff to compute the gradients. The Individual.fitness does NOT need to be differentiable,
        but the log probability computations in Population.sample MUST be.

        :param n_samples: How many individuals to sample to approximate the gradient
        :param pool: Optional process pool to use when computing the fitness of the sampled individuals.
        :param fitness_shaping_fn: Optional function to modify the fitness, e.g. normalization, etc. Input is a list of n raw fitness floats. Output must also be n floats.
        :return: A (n,) tensor containing the raw fitness (before fitness_shaping_fn) for the n individuals.
        """

        samples = self.sample(n_samples)  # Generator
        individuals = []
        grads = []
        for individual, log_prob in samples:  # Compute gradients one at a time so only one log prob computational graph needs to be kept in memory at a time.
            assert log_prob.ndim == 0 and log_prob.isfinite() and log_prob.grad_fn is not None, "log_probs must be differentiable finite scalars"
            individuals.append(individual)
            grads.append([g.cpu() for g in t.autograd.grad(log_prob, self.parameters())])

        # get raw fitness of all organisms in one population.
        if pool is not None:
            raw_fitness = pool.map(_fitness_fn_no_grad, individuals)
        else:
            raw_fitness = list(map(_fitness_fn_no_grad, individuals))

        # fitness shaping for single agent fitness
        fitness = fitness_shaping_fn(raw_fitness)
        # set gradient of each parameter
        for i, p in enumerate(self.parameters()):
            p.grad = -t.mean(t.stack([ind_fitness * grad[i] for grad, ind_fitness in zip(grads, fitness)]), dim=0).to(p.device)

        return t.tensor(raw_fitness)

    # def compute_pi_loss(self, data):
    #     o = data['obs']
    #     est_cost = self.compute_cost(self.est_fun.forward(o, self.policy_fun.action(o)))
    #     return est_cost.mean()  # gradient descent
    #
    # def compute_cost(self, output):
    #     weight = torch.tensor([0.8, 0.2, 0.2], requires_grad=False)
    #     cost = output * weight
    #     return cost


def _fitness_fn_no_grad(ind): # return individual fitness
    with t.no_grad():
        return ind.fitness()

class Individual:
    """
    An individual which has a fitness.
    """

    def fitness(self) -> float:
        """
        :return: The fitness of the individual. Does NOT need to be differentiable.
        """
        raise NotImplementedError
