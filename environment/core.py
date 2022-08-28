import numpy as np


def local_To_bullet_position(position):
    # apply for both agent and target
    return np.array(position) + np.array([-10, -10, 0])


def bullet_To_local_position(position):
    # apply for both agent and target
    # represent bullet position in self defined coordinate
    # unit: km
    return np.array(position, dtype=np.float64) + np.array([10, 10, 0], dtype=np.float64)


def local_To_bullet_orientation(orientation):
    return np.array(orientation) + np.array([np.pi / 2, 0., -np.pi / 2])


def bullet_To_local_orientation(orientation):
    # angle: [0, 2*pi]
    orient = np.array(orientation) + np.array([-np.pi / 2, 0., np.pi / 2])
    orient[orient<-0.01] += 2*np.pi
    return orient

def local_To_bullet_target_orientation(orientation):
    return np.array(orientation) + np.array([0., 0., -np.pi / 2])


def bullet_To_local_target_orientation(orientation):
    orient = np.array(orientation) + np.array([0., 0., np.pi / 2])
    orient[orient<-0.01] += 2*np.pi

    return orient


def get_communication_topology(agent_num, commun_type, formation_def, failure_nodes=None):
    """
    Additionally, please record historical set_of_neighbors every time it is called, for plotting trajectory
    :param agent_num:
    :param commun_type:
    :param formation_def:
    :param failure_nodes: List/Tuple[(num)]. The index of failure nodes begin from zero.
    :return:
    """
    commun_topology = np.zeros((agent_num, agent_num), dtype=int)
    if commun_type == "Undirected":
        for i in range(agent_num):
            for j in range(agent_num):
                if abs(i - j) == 1 or abs(i - j) == agent_num - 1:
                    commun_topology[i][j] = 1
        cluster_head = None
        if failure_nodes is not None:
            for node in failure_nodes:
                commun_topology[:, node] = 0.
                commun_topology[node, :] = 0.
    elif commun_type == "AdaptiveTopology":
        # change formation_def -> generate new topology -> fill in failure nodes with zeros
        if failure_nodes is not None:
            # you must expand missing dimension in ascending order
            failure_nodes = sorted(failure_nodes)
            effective_agent_num = agent_num - len(failure_nodes)
            assert effective_agent_num >= 3, "Unstable Topology"
            formation_def = np.delete(formation_def, list(failure_nodes), axis=0)
        else:
            effective_agent_num = agent_num
        # when passed to generate the partial_topology,
        # cluster_head is the effective node with the minimal index
        cluster_head = 0
        commun_topology = generate_adaptive_topology(effective_agent_num, formation_def, cluster_head)

        # complete the topology
        if failure_nodes is not None:
            # we need to select a feasible cluster_head is failure_nodes exist
            feasible_set_of_CH = list(range(agent_num))
            for node in failure_nodes:
                commun_topology = np.insert(np.insert(commun_topology, node, 0, axis=0), node, 0, axis=1)
                feasible_set_of_CH.remove(node)
            cluster_head = min(feasible_set_of_CH)
    else:
        raise NotImplementedError("This type of communication topology has not been implemented.")

    # print(f"Topology: {commun_topology}, cluster head: {cluster_head}")
    return commun_topology, cluster_head

def generate_adaptive_topology(agent_num, formation_def, cluster_head):

    commun_topology = np.zeros((agent_num, agent_num), dtype=int)

    inter_distances = np.zeros((agent_num, agent_num), dtype=float)
    for id in range(agent_num):
        inter_distances[id] = [np.linalg.norm(formation_def[id] - formation_def[i]) for i in range(agent_num)]
    # make sure its symmetric matrix
    assert (inter_distances == inter_distances.T).all()
    # make sure cluster head & itself will not be selected as the second connected node
    np.fill_diagonal(inter_distances, 999.)
    inter_distances[:, cluster_head].fill(999.)
    inter_distances = inter_distances.round(decimals=5)
    # generate communication topology
    commun_topology[cluster_head] = np.zeros((agent_num,), dtype=float)
    for id in range(1, agent_num):
        # each cluster member is directly connected with 2 nodes
        commun_topology[id, 0] = 1.
        commun_topology[id, np.argmin(inter_distances[id])] = 1.

    return commun_topology


def get_set_of_neighbors_form_adjacent_matrix(adj_matrix):
    """
    :param adj_matrix:
    :return: A two-dimensional list
    """
    agent_num = len(adj_matrix)
    set_of_neighbors = []
    for i in range(agent_num):
        temp_list = []
        for j in range(agent_num):
            if adj_matrix[i][j] != 0:
                temp_list.append(j)
        set_of_neighbors.append(temp_list)
    return set_of_neighbors


def formation_maintenance_error(zero_order_state, formation_def, set_of_neighbors):
    """
    :param zero_order_state:
    :param formation_def:
    :param set_of_neighbors:
    :return: ndarray*[(agent_num)*[(3)]]
    """
    m_e = np.zeros_like(zero_order_state)
    for i in range(len(zero_order_state)):
        m_e[i] = np.sum([[formation_def[i] - formation_def[j] - zero_order_state[i] + zero_order_state[j]] for j in
                         set_of_neighbors[i]], axis=0)
    return m_e


def reference_tracking_error(zero_order_state_ref, zero_order_state, formation_def, miu):
    """ Note that tracking error and other error are signed.
    :param zero_order_state_ref:
    :param zero_order_state:
    :param formation_def:
    :param miu: reference relationship vector
    :return: ndarray*[(agent_num)*[(3)]]
    """
    t_e = np.zeros_like(zero_order_state)
    for i in range(len(zero_order_state)):
        t_e[i] = miu[i] * (zero_order_state_ref + formation_def[i] - zero_order_state[i])
    return t_e


def rotate_error_matrix(e, alpha):
    # rotate along z axis.
    e_r = np.zeros_like(e)
    for i in range(e.shape[0]):
        e_r[i] = np.matmul(np.array([[np.cos(alpha[i]), np.sin(alpha[i]), 0.],
                                     [-np.sin(alpha[i]), np.cos(alpha[i]), 0],
                                     0., 0., 1.], dtype=float), e[i])
    return e_r


def get_dot_of_er(e, action, action_target, env):
    """ Remember to check if the shape of 1-d matrix is right.
    always call it AT THE END OF THE STEP
    and CHECK ACTION UNIT
    """
    e_r_dot = np.zeros((env.config.agent_num, 3))

    assert len(action) == 2, "check action dimension."
    a_v = np.array(action[0]) * env.config.Max_Velocity_Acceleration
    a_l = np.array(action[1]) * env.config.Max_Lateral_Acceleration

    u = np.matrix([a_v, a_l])
    u_r = np.matrix(action_target).T

    for i in range(env.config.agent_num):
        G_i = np.array([[-env.tau * (len(env.set_of_neighbors[i]) + env.miu[i]), e[i][1] / env.M_speeds[i]],
                        [0, -e[i][0] / env.M_speeds[i]],
                        [0, -(len(env.set_of_neighbors[i]) + env.miu[i]) / env.M_speeds[i]]], dtype=float)
        F_ij = [np.array([[np.cos(env.alpha[j] - env.alpha[i]) * env.tau, 0],
                          [np.sin(env.alpha[j] - env.alpha[i]) * env.tau, 0],
                          [0, 1 / env.M_speeds[j]]], dtype=float) for j in env.set_of_neighbors[i]]
        D_i = np.array([[env.miu[i] * np.cos(env.alpha_target - env.alpha[i]), 0],
                        [env.miu[i] * np.sin(env.alpha_target - env.alpha[i]), 0],
                        [0, env.miu[i]]], dtype=float)
        H_i = np.matrix([[-(len(env.set_of_neighbors[i]) + env.miu[i]) * env.M_speeds[i] +
                          np.sum([np.cos(env.alpha[j] - env.alpha[i]) * env.M_speeds[j] for j in env.set_of_neighbors[i]])],
                         [np.sum([np.sin(env.alpha[j] - env.alpha[i]) * env.M_speeds[j] for j in env.set_of_neighbors[i]])],
                         [0.]])
        e_r_dot[i] = np.squeeze(G_i @ u[:, i] + \
                                np.sum([F_ij[k] @ u[:, j] for k, j in enumerate(env.set_of_neighbors[i])], axis=0) +
                                D_i @ u_r +
                                H_i)
    return e_r_dot




class objective_observer():
    def __init__(self, env):
        self.config = env.config
        self.agent_num = env.config.agent_num
        self.tau = env.config.tau
        self.frameskip = env.config.frameskip
        self.weight_matrix = np.matrix([100, 100, 50]).T
        self.K_C = np.diag(env.config.K_C)
        self.env = env
        self.early_stop = env.config.early_stop
        self.threshold = 1.1e-1

    def reward_function(self, inp_actions):
        """
        As long as action fit > 0, it means that the action leads to reduction in error
        :param er:
        :param inp_actions:
        :param action_target:
        :param env:
        :return:
        """
        reward = np.zeros(self.agent_num)
        er = self.env.resultant_error # when the resultant error is zero, make it 1.
        er_dot = get_dot_of_er(er, inp_actions, self.env.action_target, self.env)
        self.er_next = er + er_dot * self.tau * self.config.frameskip
        er_delta = np.abs(self.er_next) - np.abs(er)

        if self.early_stop:
            stop = True if np.max(er_delta) > self.threshold else False
        else:
            stop = False
        # formation resultant error reward
        for i in range(self.agent_num):
            # reward[i] = np.exp(-np.sum(abs(e[i])))
            reward[i] = np.exp(-np.matrix(er[i]) @ self.K_C @ np.matrix(er[i]).T)
        # model predictive reward
        # action_fit = -er_delta @ self.weight_matrix  # matrix[agent_num*1]
        # action_fit = np.array(1 / (1 + np.exp(-(action_fit-9)/4))).squeeze() # logistic function
        # action_fit = np.array(np.tanh(action_fit)).squeeze()
        # reward += action_fit
        return reward, stop

