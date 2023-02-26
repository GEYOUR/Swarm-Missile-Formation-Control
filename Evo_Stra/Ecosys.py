import copy
import pickle
from typing import Callable, Iterable
import os
import sys
from Common.Config import get_default_config
import multiprocessing
from multiprocessing import Process

import numpy as np
import torch
import torch as t
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")
from torch import optim
from torch.multiprocessing import Pool

from Common.Config import get_args_from_json
from Common.utils import compute_centered_ranks
from Evo_Stra.normal_population import NormalPopulation
from environment.SwarmEnv import swarmMissileEnv
from Common.Definition import ROOT_DIR

class Ecosystem:
    def __init__(self, pop_args, args):
        self.pop_args = pop_args

        self.config = args
        self.tool_env = swarmMissileEnv(config=self.config)
        self.agent_num = self.config.agent_num
        self.constructor = pop_args[1]
        # self.init_pop_size = copy.deepcopy(self.config.pop_size)
        self.variance = self.config.std**2
        # pop size is changeable
        parameter_num = (self.config.obs_dim * self.config.policy_hidden_size[0] + self.config.policy_hidden_size[0]) + \
                        sum([(self.config.policy_hidden_size[i] * self.config.policy_hidden_size[i+1] +
                              self.config.policy_hidden_size[i+1]) for i in range(len(self.config.policy_hidden_size)-1)]) + \
                        (self.config.policy_hidden_size[-1] * self.config.action_dim + self.config.action_dim)

        self.pop_size = int(10 + 5 * np.log(parameter_num))
        self.pop_size = self.pop_size if self.pop_size % 2 == 0 else self.pop_size + 1  # make it even
        self.pop_size_min = self.pop_size
        self.pop_size_max = 4 * self.pop_size_min
        self.pop_size_max = self.pop_size_max if self.pop_size_max % 2 == 0 else self.pop_size_max + 1
        if not self.config.population_adaptation:
            self.pop_size = 3*self.pop_size
            self.pop_size = self.pop_size if self.pop_size % 2 == 0 else self.pop_size + 1  # make it even

        self.beta = self.config.beta

        # record
        self.group_fitness = []
        self.group_std = []
        self.group_pop = []

        self.evolution_path = []
        self.pop_size_history = [self.pop_size]
        self.learning_rate = self.config.learning_rate

        # add population group
        for id in range(self.agent_num):
            pop = NormalPopulation(*pop_args)
            self.group_pop.append(pop)

        self.optimizers = [
            optim.Adam(self.all_parameters()[i], lr=self.learning_rate, weight_decay=self.config.lr_decay)
            for i in range(self.agent_num)]
        self.pool = Pool()

        self.sample_counter = 0

    def all_parameters(self) -> object:
        """
        :return: List[{parameters of the agent}(agent_num)]
        """
        all_params = []
        for i in range(self.agent_num):
            all_params.append(self.group_pop[i].parameters())
        return all_params

    def fitness_gradients(self,
                          n_samples: int,
                          pool: Pool = None,
                          fitness_shaping_fn: Callable[[Iterable[float]], Iterable[float]] = lambda x: x):
        samples_generators = self.sample(n_samples)
        all_individuals = []  # List*[(n_samples)*[(agent_num)]]
        all_grads = []
        all_probs = []
        for id in range(self.agent_num):
            it = 0
            for individual, log_prob, prob in samples_generators[id]:
                if id == 0:  # append agent_0 first
                    assert log_prob.ndim == 0 and log_prob.isfinite() and log_prob.grad_fn is not None, "log_probs must be differentiable finite scalars"
                    all_individuals.append([individual])
                    all_grads.append([[g.cpu() for g in t.autograd.grad(log_prob, self.group_pop[id].parameters())]])
                    all_probs.append([prob])
                else:
                    all_individuals[it].append(individual)
                    all_grads[it].append([g.cpu() for g in t.autograd.grad(log_prob, self.group_pop[id].parameters())])
                    all_probs[it].append(prob)
                it += 1

        if pool is not None:
            outputs = pool.map(get_fitness, all_individuals)
        else:
            raise BrokenPipeError

        # # List*[(n_samples)*[(agent_num)]]
        raw_fitness = [outputs[i] for i in range(len(outputs))]
        raw_fitness = np.stack(raw_fitness)
        fitness = fitness_shaping_fn(raw_fitness)
        # update grads
        for id in range(self.agent_num):
            comm_id = self.tool_env.set_of_neighbors[id]
            for i, p in enumerate(self.group_pop[id].parameters()):
                p.grad = -t.mean(t.stack(
                    [inds_fitness[id] * grad[id][i] * self._rescale_prob(prob, comm_id, i) for grad, inds_fitness, prob
                     in zip(all_grads, fitness, all_probs)]), dim=0).to(p.device)
        self.sample_counter += 1
        return raw_fitness, fitness

    def _rescale_prob(self, prob, comm_ids, i):
        rescale_prob = torch.ones_like(prob[0][i])
        if len(comm_ids) == 0:
            return 1
        else:
            for c_id in comm_ids:
                rescale_prob *= prob[c_id][i]
            return rescale_prob

    def update(self):

        history_parameter = [torch.cat([ts.reshape(-1) for ts in self.group_pop[i].parameters()]) for i in range(self.agent_num)]

        for i in range(self.config.agent_num):
            self.optimizers[i].zero_grad()
        # calculate gradient base on fitness
        if self.config.Fitness_Reshape:
            raw_fitness, fitness = self.fitness_gradients(self.pop_size, self.pool, compute_centered_ranks)
        else:
            raw_fitness, fitness = self.fitness_gradients(self.pop_size, self.pool)
        self.group_fitness.append([raw_fitness[:, i].mean().item() for i in range(self.config.agent_num)])
        self.group_std.append([raw_fitness[:, i].std().item() for i in range(self.config.agent_num)])
        # step update
        for i in range(self.config.agent_num):
            self.optimizers[i].step()

        new_parameter = [torch.cat([ts.reshape(-1) for ts in self.group_pop[i].parameters()]) for i in
                         range(self.agent_num)]
        # represent the parameter moving distance after update. The longer it moves, the smaller the pop size should be
        delta_theta = sum([((his_theta - theta).T @ (his_theta - theta) / self.variance).item() for his_theta, theta in
                       zip(history_parameter, new_parameter)])


        self.evolution_path.append(delta_theta)

        return raw_fitness

    def _pure_update(self):
        assert self.group_pop[0].param_means['0.weight'].grad is not None, "Gradient missing!"
        for i in range(self.config.agent_num):
            self.optimizers[i].step()

    def sample(self, n_samples):
        samples_generators = [] # List*[(agent_num)*[Generator(n_samples)]]
        for id in range(self.agent_num):
            samples_generators.append(self.group_pop[id].sample(n_samples))

        return samples_generators

    def evaluate(self, iter):
        """
        evaluate policy networks.
        :param iter:
        :return:
        """
        r_tot = self.evaluate_fitness(iter)
        p2 = Process(target=self.plot_fitness, args=(self.group_fitness, self.group_std, self.config))
        p2.start()
        p2.join()
        p3 = Process(target=self.plot_pop_his, args=(self.pop_size_history, self.config))
        p3.start()
        p3.join()

        self.save_model(iter)
        self.save_training_data()
        return r_tot

    def evaluate_fitness(self, iter):
        n_individuals = [self.constructor(self.group_pop[id].param_means) for id in range(self.agent_num)]
        env = swarmMissileEnv(config=self.config, analytical=True)
        obs = env.reset()
        terminated = False
        r_tot = np.zeros(self.agent_num)
        while not terminated:
            actions = np.stack([n_individuals[id].action(obs[id]) for id in range(env.agent_num)]).T
            obs, r, terminated = env.step(actions)
            r_tot += r
        env.close()
        env.snap_shot(iter)
        env.plot_analytic_data("last")
        return r_tot

    def save_model(self, iter):
        for id in range(self.agent_num):
            self.group_pop[id].save_model(id, iter)
        print("group models saved!")


    def save_training_data(self):
        # save pop_size_history
        with open(os.path.join(ROOT_DIR, self.config.result_dir, "Analytic_data", 'pop_size_his.pkl'), 'wb') as f:
            pickle.dump(self.pop_size_history, f)
        # dump result
        with open(os.path.join(ROOT_DIR, self.config.result_dir, "Analytic_data", 'mean_fits.pkl'), 'wb') as f:
            pickle.dump(self.group_fitness, f)

    @staticmethod
    def plot_fitness(groups_fitness, groups_std, args):
        """ Plot :
            1, mean fitness
            2, std fitness
        :param groups_fitness:
        :param groups_std:
        :return:
        """
        plt.figure()
        plt.cla()
        plt.xlabel("iterations")
        plt.ylabel("fitness")
        plt.plot(range(len(groups_fitness)), groups_fitness, linewidth=0.5,
                 label=[f'agent_{i}' for i in range(args.agent_num)])
        plt.legend()
        plt.title("Mean fitness")
        plt.savefig(os.path.join(ROOT_DIR, args.result_dir, 'mean_fits.svg'), format='svg')

        plt.cla()
        plt.xlabel("iterations")
        plt.ylabel("fitness std")
        plt.plot(range(len(groups_std)), groups_std, linewidth=0.5,
                 label=[f'agent_{i}' for i in range(args.agent_num)])
        plt.title("fitness std")
        plt.legend()
        plt.savefig(os.path.join(ROOT_DIR, args.result_dir, "std_fits.svg"), format='svg')
        plt.close()

    @staticmethod
    def plot_pop_his(pop_size_his, args):
        plt.figure()
        plt.cla()
        plt.xlabel("iterations")
        plt.ylabel("population size")
        plt.plot(range(len(pop_size_his)), pop_size_his, linewidth=0.5,
                 )
        plt.title("Pop Size")
        plt.savefig(os.path.join(ROOT_DIR, args.result_dir, 'pop_his.svg'), format='svg')

        plt.close()




def get_fitness(n_individuals):
    env = swarmMissileEnv(config=get_args_from_json())

    obs = env.reset()
    terminated = False
    r_tot = np.zeros(env.config.agent_num)
    assert len(n_individuals) == env.agent_num, "Agent num does not match the Env!"

    while not terminated:
        actions = np.stack([n_individuals[id].action(obs[id]) for id in range(env.agent_num)]).T
        obs, r, terminated = env.step(actions)
        r_tot += r
    env.close()

    return r_tot
