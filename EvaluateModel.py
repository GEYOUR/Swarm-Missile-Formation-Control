import argparse
import sys
import os
import time

import numpy as np
import torch

from environment.SwarmEnv import swarmMissileEnv
from Common.Config import get_args_from_json
from Evo_Stra.Env_Ind import Missile


class Policy:
    def __init__(self, model_path, config):
        self.agent_num = config.agent_num
        self.agent_list = []
        model_names = os.listdir(model_path)

        for i in range(self.agent_num):
            # fist of the matched model
            model_name = [s for s in model_names if f"evo_net_{i}" in s][0]
            self.agent_list.append(Missile().from_params(torch.load(os.path.join(model_path, model_name)), config))

    def action(self, obs):
        return np.array([self.agent_list[i].action(obs[i]) for i in range(self.agent_num)]).T



def evaluate_model(path, render):
    model_path = os.path.join(path, "models")
    graphs_path = os.path.join(path, "Graphs")

    config = get_args_from_json(os.path.join(path, "config", "configuration.json"))
    policy = Policy(model_path, config)

    env = swarmMissileEnv(config, render=render, analytical=True)
    obs = env.reset()
    terminate = False

    while not terminate:
        action = policy.action(obs)
        obs, reward, terminate = env.step(action)
        if render:
            time.sleep(env.tau*2)

    env.plot_analytic_data(suffix="formal", save_path=graphs_path)
    env.snap_shot(suffix="formal", save_path=graphs_path)
    env.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./CasesResults/node_failure", type=str)
    args = parser.parse_args()
    evaluate_model(args.path, render=False)