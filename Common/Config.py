import argparse
import json
import sys

sys.path.append("..")
import numpy as np
import os
from Common.Definition import ROOT_DIR


def get_default_config():
    parser = argparse.ArgumentParser()

    # --- environmental ---
    # normally this section should not be changed.
    parser.add_argument("--tau", default=0.1, type=float)
    parser.add_argument("--epoch_time", default=25.0, type=float)

    parser.add_argument("--palette",
                        default=[[123, 104, 238], [255, 228, 181], [197, 79, 133], [0, 191, 255], [134, 242, 192],
                                 [60, 179, 113],
                                 [0, 255, 255], [64, 224, 208], [30, 144, 255], [138, 43, 226], [139, 0, 139]],
                        type=list, help="colors for missiles & their flight tails")

    # formation control
    parser.add_argument("--formation_pattern", default="RegularPolygon", type=str, help="assign pattern shape"
                                                                                        "options:"
                                                                                        "RegularPolygon,"
                                                                                        "StraightLine")
    parser.add_argument("--formation_gap", default=0.5, type=float, help="controlling the gap among each node")
    parser.add_argument("--formation_rot", default=0*np.pi, type=float, help="controlling the rotation of formation")

    parser.add_argument("--Max_M_V", default=0.8, type=float, help="Unit, km/s")
    parser.add_argument("--Min_M_V", default=0.3, type=float)
    parser.add_argument("--Max_Velocity_Acceleration", default=300 * 0.001, type=float, help="Unit, km/s^2")
    parser.add_argument("--Max_Lateral_Acceleration", default=400 * 0.001, type=float, help="Unit, unknown")

    parser.add_argument("--action_dim", default=2, type=int)


    # --- optional ---
    parser.add_argument("--target_mobility_strategy", default="GoStraightAlongX", type=str, help="Options:" 
                                                                                                 "GoStraightAlongX"
                                                                                                 "GoStraightAlongY"
                                                                                                 "GoALongDiagonal"
                                                                                                 "GoRoundInCircle"
                                                                                                 "GoInSpiral"
                                                                                                 "GoInSinusoidal"
                                                                                                 "GoRandomly")
                                                                                        # modify code in SwarmEnv.py 325
    parser.add_argument("--swarm_born_condition", default="In_Formation", type=str, help="Options:"
                                                                                         "In_Formation"
                                                                                         "RandomlySpread")
    parser.add_argument("--target_born_condition", default="Formation_Center", type=str, help="Options:"
                                                                                              "Formation_Center"
                                                                                              "AwayFromFormation")
    parser.add_argument("--barrier_setup", default=False, type=bool)
    parser.add_argument("--switchFormationStrategy", default="switchPattern", type=str, help="Options:"
                                                                                             "switchPattern"
                                                                                             "switchSize")

    parser.add_argument("--frameskip", default=1, type=int)
    parser.add_argument("--agent_num", default=4, type=int)
    parser.add_argument("--population_adaptation", default=True, type=bool)
    parser.add_argument("--early_stop", default=True, type=bool)

    parser.add_argument("--Topology_Type", default="Undirected", type=str, help="Options:"
                                                                                "Undirected"
                                                                                "AdaptiveTopology")
    parser.add_argument("--Node_Failure", default="None", type=str, help="Options:"
                                                                         "None"
                                                                         "OneFail"
                                                                         "TwoFail")
    parser.add_argument("--Network_Type", default="MLP", type=str)

    # policy network
    parser.add_argument("--policy_hidden_size", default=[16], type=list)
    parser.add_argument("--optimizer", default="Adam", type=str)

    # --- algorithmic ---
    parser.add_argument("--Fitness_Reshape", default=True, type=bool)
    parser.add_argument("--evaluate_cycle", default=50, type=int)
    parser.add_argument("--learning_rate", default=0.02, type=float)
    parser.add_argument("--std", default=0.2, type=float)
    parser.add_argument("--sigma_decay", default=6e-4, type=float, help='decay factor for sampling std')

    parser.add_argument("--lr_decay", default=6e-4, type=float)
    parser.add_argument("--beta", default=0.84, type=float)

    parser.add_argument("--Iterations", default=5000, type=int)
    parser.add_argument("--K_C", default=[0.15, 0.15, 0.1], type=list)

    # --- Indifferent --
    parser.add_argument("--World_Size", default=[[0, 20], [0, 20]], type=list)
    parser.add_argument("--Debug_mode", default=False, type=str)
    # directories are defined based on ROOT_DIR
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--result_dir", default="Result", type=str)
    parser.add_argument("--replay_dir", default="replay", type=str)

    return parser.parse_args()


def load_default_config():
    config_args = get_default_config()
    config_dict = vars(config_args)
    with open(os.path.join(ROOT_DIR, 'Common', "configuration.json"), 'w') as f:
        json.dump(config_dict, f)
    print("Configuration file reset!")


def get_args_from_json(path=None):
    if path is None:
        with open(os.path.join(ROOT_DIR, 'Common', 'configuration.json'), 'r') as f:
            config_dict = json.load(f)
    else:
        with open(os.path.join(ROOT_DIR, path), 'r') as f:
            config_dict = json.load(f)
    # insecure correct obs_dim
    args = argparse.Namespace(**config_dict)
    if args.barrier_setup:
        args.obs_dim = 6
    else:
        args.obs_dim = 5

    return args


def set_json_from_dict(dict):
    """
    Reset json file
    :param dict:
    :return:
    """
    # the initial are reset randomly every time parameter reset button is pushed.
    if dict['swarm_born_condition'] == "RandomlySpread":
        dict['init_positions'] = np.concatenate((np.random.uniform(low=7, high=11, size=(dict['agent_num'], 1)),
                                                np.random.uniform(low=2, high=5, size=(dict['agent_num'], 1))), axis=1).tolist()

    with open(os.path.join(ROOT_DIR, 'Common', 'configuration.json'), 'w') as f:
        json.dump(dict, f)
    print('configuration updated!')


def update_json_from_dict(dict):
    """
    Update partial configurations
    :param dict:
    :return:
    """
    with open(os.path.join(ROOT_DIR, "Common", 'configuration.json'), 'r') as f:
        config_dict = json.load(f)
    for k, v in dict.items():
        config_dict[k] = v
    with open(os.path.join(ROOT_DIR, 'Common', 'configuration.json'), 'w') as f:
        json.dump(config_dict, f)
    print("configuration updated!")


if __name__ == '__main__':
    load_default_config()
