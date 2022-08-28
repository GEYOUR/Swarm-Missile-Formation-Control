import sys

sys.path.append("..")

from typing import Iterable
import numpy as np
import torch as t
import matplotlib as mpl
from Common.Config import get_args_from_json


def generate_topology_mtx(n_agents, type):
    communication_topology = np.zeros((n_agents, n_agents), dtype=int)
    if type == "Undirected":
        for i in range(n_agents):
            for j in range(n_agents):
                if abs(i - j) == 1 or abs(i - j) == n_agents - 1:
                    communication_topology[i][j] = 1
    elif type == "Leader_Follower":
        for i in range(n_agents):
            if i + 1 < (n_agents - 1) / 2 + 1:
                communication_topology[i][i + 1] = 1
            elif i + 1 > (n_agents - 1) / 2 + 1:
                communication_topology[i][i - 1] = 1
    elif type == "Directed":
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j+1 or i+n_agents-1 == j:
                    communication_topology[i][j] = 1

    return communication_topology


def compute_centered_ranks(raw_fits):
    """
    Compute centered ranks, e.g. [-81.0, 11.0, -0.5] --> [-0.5, 0.5, 0.0]
    arrange fitnesse between[-0.5, 0.5] according to ranks
    """
    raw_fits = np.array(raw_fits)
    fits = np.zeros_like(raw_fits)
    n_agents = len(raw_fits[0])

    for id in range(n_agents):
        x = raw_fits[:, id]
        assert x.ndim == 1
        ranks = t.zeros((len(x),), dtype=t.long)
        ranks[x.argsort()] = t.arange(len(x))
        ranks = ranks.to(t.float32)
        ranks = ranks / (len(x) - 1)
        ranks = ranks - 0.5
        ranks = ranks
        fits[:, id] = ranks
    return fits


def normalize(x: Iterable[float]) -> Iterable[float]:
    """
    Normalize a list of floats to have zero mean and variance 1
    """
    x = t.tensor(x)
    assert x.ndim == 1
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x.tolist()


def log_param_dict(dir, dict):
    with open(dir + "/log_params.txt", "w") as f:
        for k, v in dict.items():
            if type(v) not in (str, bool):
                f.write(f"{k}: {v} \n")


def rotation_matrix(theta, dim):
    """ Note that the rotation is counterclockwise
    :param theta:
    :param dim: 2 or 3
    :return: np.matrix
    """
    if dim == 2:
        return np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    elif dim == 3:
        return np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],
                          [0., 0., 1.]])
    else:
        raise KeyError

def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[-0.2, 0.3], [-0.2, -0.3], [0.8, 0], [-0.2, 0.3]])  # arrow shape
    angle = rot
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
        ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO, mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    # # scale
    # x0 = np.amin(arr[:, 0])
    # x1 = np.amax(arr[:, 0])
    # y0 = np.amin(arr[:, 1])
    # y1 = np.amax(arr[:, 1])
    # scale = np.amax(np.abs([x0, x1, y0, y1]))

    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker


if __name__ == '__main__':
    print(generate_topology_mtx(4, type="Directed"))
