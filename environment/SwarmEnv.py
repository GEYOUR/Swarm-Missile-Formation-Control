import copy
import os.path
import platform
import random
import sys
import threading
import time
from multiprocessing import Process

import numpy as np

import Common.Config

sys.path.append("..")
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")
import pybullet as pb
from pybullet_utils import bullet_client
import pybullet_data

from Common.Config import get_default_config, get_args_from_json
from Common.Definition import ROOT_DIR
from Common.utils import rotation_matrix, gen_arrow_head_marker
from environment.core import *


class swarmMissileEnv():
    def __init__(self, config, render=False, focus_view=True, analytical=False):
        """

        :param config:
        :param render:
        :param focus_view:
        :param analytical: enable this operation, analytical data will be collected which may bring computational cost
        """
        self.config = config
        self.analytical = analytical
        self.agent_num = config.agent_num
        self.tau = config.tau  # timestep
        self.total_steps = int(self.config.epoch_time / (self.tau * self.config.frameskip))  # for single step event

        self.sim_time = 0
        self.episode_count = 0
        self.focus_view = focus_view
        self.render = render
        self.obj_obs = objective_observer(self)
        self.barrier_on = config.barrier_setup
        self.maxAngularV = self.config.Max_Lateral_Acceleration / self.config.Max_M_V

        # if config.leader_state_comm_stra == "All_Receive":
        #     self.miu = [1.] * self.agent_num
        # elif config.leader_state_comm_stra == "FirstOne_Receive":
        #     self.miu = [1.] + [0.] * (self.agent_num - 1)
        # connect client
        if render:
            self.bulletClient = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            self.bulletClient = bullet_client.BulletClient(connection_mode=pb.DIRECT)

        self.bulletClient.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bulletClient.setTimeStep(self.tau)  # default timestep is 1/240s.
        self.bulletClient.setGravity(0, 0, 0)
        #
        self.add_debug_elements()
        self.bulletClient.configureDebugVisualizer(flag=self.bulletClient.COV_ENABLE_WIREFRAME)
        self.bulletClient.resetDebugVisualizerCamera(cameraDistance=10., cameraYaw=0.,
                                                     cameraPitch=-89.5, cameraTargetPosition=[-2, -2, 1])
        self.formation_def = self.get_formation_form_pattern(pattern=config.formation_pattern,
                                                             rotate=config.formation_rot, gap=config.formation_gap)

        # ALL THIS PART OF CODE SHOULD BE MODIFIED IF TOPOLOGY CHANGED.
        self.commun_topology, self.cluster_head = get_communication_topology(self.agent_num, self.config.Topology_Type,
                                                                             self.formation_def)
        self.set_of_neighbors = get_set_of_neighbors_form_adjacent_matrix(self.commun_topology)
        self.miu = np.zeros((self.agent_num,), dtype=float)
        self.topologyChangeStep = int(self.total_steps / 2) if self.config.Node_Failure != "None" else None
        if self.config.Topology_Type == "AdaptiveTopology":
            self.miu[self.cluster_head] = 1.  # only head perceive the target
        else:  # self-defined
            self.miu[:] = 1
        # -------------------------------------------------------------------
        self.palette = np.array(config.palette, dtype=float) / 255

        self.center_position = [5., 5., 0.]
        self.init_positions, self.init_orientations, self.init_velocities, self.init_angular_velocities \
            = self.swarm_born(self.config.swarm_born_condition, self.formation_def, self.center_position)
        self.center_position = np.average(self.init_positions, axis=0).tolist()[:2] + [0]
        self.init_position_t, self.init_orientation_t, self.init_velocity_t, self.init_angular_velocity_t \
            = self.target_born(self.config.target_born_condition, self.center_position)

        # LOAD models
        # warning: z coordinate must be > 0, otherwise it will collide with the plane.
        self.swarm_ids = self.load_swarm(self.init_positions, self.init_orientations)
        planeId = self.bulletClient.loadURDF("plane.urdf")
        self.target_id = self.load_target(self.center_position, [0., 0., 0.])
        if self.barrier_on:
            # LOAD barrier
            self.barrier_size = [0.1, 1, 0.4]  # size correspond with edge length along y, x, z
            self.br_ctr_pos = [12., 12., 0.5]  # again, z axis offset is important
            self.LBR_POS, self.RBR_POS = self.load_barrier(self.barrier_size, center_position=self.br_ctr_pos,
                                                           gap_width=1.)
            self.min_dis = 2.0 * self.agent_num * self.config.formation_gap
            self.trans_dis = 0.7 * self.min_dis
            self.formation_signal = 0 if self.config.formation_pattern == "RegularPolygon" else 1

        # collect analytical data
        if self.analytical:
            self.resultant_error_record = []
            self.formationMaintenance_error_record = []
            self.referenceTracking_error_record = []

            self.m_speed_record = []
            self.t_speed_record = []
            self.m_alpha_record = []
            self.t_alpha_record = []

    def reset(self):
        """
        start client and reset to initial states
        :return: None
        """
        # reset missile initial state
        for index, missile_id in enumerate(self.swarm_ids):
            self.bulletClient.changeDynamics(bodyUniqueId=missile_id, linkIndex=-1, linearDamping=0,
                                             angularDamping=0)
            self.bulletClient.resetBaseVelocity(objectUniqueId=missile_id, linearVelocity=self.init_velocities[index],
                                                angularVelocity=self.init_angular_velocities[index])
            self.bulletClient.resetBasePositionAndOrientation(bodyUniqueId=missile_id, posObj=local_To_bullet_position(
                self.init_positions[index]),
                                                              ornObj=self.bulletClient.getQuaternionFromEuler(
                                                                  local_To_bullet_orientation(
                                                                      self.init_orientations[index])))
        # reset target initial state
        self.bulletClient.changeDynamics(bodyUniqueId=self.target_id, linkIndex=-1, linearDamping=0,
                                         angularDamping=0)
        self.bulletClient.resetBaseVelocity(objectUniqueId=self.target_id, linearVelocity=self.init_velocity_t,
                                            angularVelocity=self.init_angular_velocity_t)
        self.bulletClient.resetBasePositionAndOrientation(bodyUniqueId=self.target_id,
                                                          posObj=local_To_bullet_position(self.init_position_t),
                                                          ornObj=self.bulletClient.getQuaternionFromEuler(
                                                              local_To_bullet_target_orientation(
                                                                  self.init_orientation_t)))

        # initialize system variables
        self.M_positions = np.array(self.init_positions, dtype=float)
        self.M_orientations = np.array(self.init_orientations, dtype=float)
        self.alpha = self.M_orientations[:, 2]
        self.M_speeds = np.sqrt(self.init_velocities[:, 0] ** 2 + self.init_velocities[:, 0] ** 2)
        self.M_angular_velocities = self.init_angular_velocities[:, 2]

        self.T_speed = np.sqrt(self.init_velocity_t[0] ** 2 + self.init_velocity_t[1] ** 2)

        # retrieve initial system states form engine
        for id in range(self.agent_num):
            PosAndOrn = self.bulletClient.getBasePositionAndOrientation(bodyUniqueId=self.swarm_ids[id])
            self.M_positions[id] = bullet_To_local_position(PosAndOrn[0])
            self.M_orientations[id] = bullet_To_local_orientation(
                self.bulletClient.getEulerFromQuaternion(PosAndOrn[1]))
            self.alpha[id] = self.M_orientations[id][2]
        self.zero_order_state = np.concatenate((self.M_positions[:, :2], np.expand_dims(self.alpha, axis=1)), axis=1)

        PosAndOrn_ref = self.bulletClient.getBasePositionAndOrientation(bodyUniqueId=self.target_id)
        self.T_position = bullet_To_local_position(PosAndOrn_ref[0])
        self.alpha_target = \
            bullet_To_local_target_orientation(self.bulletClient.getEulerFromQuaternion(PosAndOrn_ref[1]))[2]
        self.zero_order_state_ref = np.array([self.T_position[0],
                                              self.T_position[1],
                                              self.alpha_target], dtype=float)
        # predefine target mobility
        if self.config.target_mobility_strategy == "GoStraightAlongX":
            self.action_target = [0.5, 0.]
        elif self.config.target_mobility_strategy == "GoRoundInCircle":
            self.action_target = [0.65, 0.15]
        elif self.config.target_mobility_strategy == "GoInSpiral":
            self.action_target = [0.65, 0.1]
        elif self.config.target_mobility_strategy == "GoInSinusoidal":
            self.action_target = [0.5, 0.0]
        elif self.config.target_mobility_strategy == "GoRandomly":
            self.action_target = [random.random() * 0.6, random.random() * 0.6]
        elif self.config.target_mobility_strategy == "GoStraightAlongY":
            self.action_target = [0.5, 0.]
        elif self.config.target_mobility_strategy == "GoALongDiagonal":
            self.action_target = [0.6, 0.]
        else:
            raise NotImplementedError
        target_velocity = [self.action_target[0] * np.cos(self.alpha_target),
                           self.action_target[0] * np.sin(self.alpha_target), 0]
        self.bulletClient.resetBaseVelocity(objectUniqueId=self.target_id, linearVelocity=target_velocity,
                                            angularVelocity=[0, 0, self.action_target[1]])

        # record
        self.sim_time = 0
        self.step_count = 0
        self.episode_count += 1  # indicating how many episodes this environment has been used.
        self.swarm_trajectory = [self.zero_order_state]  # ndarray*[(total_steps)*[(agent_num)*[(state_dims)]]]
        self.ref_trajectory = [self.zero_order_state_ref]  # ndarray*[(total_steps)*[(state_dims)]]
        self.cumul_reward = np.zeros(self.agent_num)  # cumulative reward
        self.set_of_neighbors_his = [self.set_of_neighbors]
        self.st_r_record = []
        # initialize debug view
        if self.focus_view:
            focus_position = local_To_bullet_position(np.average(self.M_positions, axis=0))
            self.bulletClient.resetDebugVisualizerCamera(cameraDistance=10., cameraYaw=0.,
                                                         cameraPitch=-89.5, cameraTargetPosition=focus_position)

        obs = self.obtain_observation(observe_option="Trans_Signal" if self.barrier_on else None)
        return obs

    def step(self, action):
        """
        :param action: 2-dimensional array,
        List/ndarray*[{velocity acceleration}List/ndarray*[(agent_num)],{lateral acceleration}List/ndarray*[(agent_num)]]
        :return:
        """
        assert len(action) == 2, "check action dimension."
        assert np.max(action) <= 1, "check action feasible region"
        a_v = np.array(action[0]) * self.config.Max_Velocity_Acceleration
        # update angular_Velocity
        a_l = np.array(action[1]) * self.config.Max_Lateral_Acceleration
        if self.config.target_mobility_strategy == "GoInSpiral":
            self.action_target[1] = np.clip(0.1 + 0.01 * self.sim_time, -self.maxAngularV, self.maxAngularV)
            self.action_target[0] = np.clip(0.65 - 0.01 * self.sim_time, self.config.Min_M_V,
                                            self.config.Max_M_V)
        elif self.config.target_mobility_strategy == "GoInSinusoidal":
            self.action_target[1] = np.clip(0.2 * np.sin(2 * np.pi / 20 * self.sim_time), -self.maxAngularV,
                                            self.maxAngularV)
            self.action_target[0] = np.clip(0.5, self.config.Min_M_V,
                                            self.config.Max_M_V)

        # step simulation with frame skip
        # update speed
        for i in range(self.config.frameskip):

            self.M_speeds += a_v * self.tau
            self.M_speeds = np.clip(self.M_speeds, a_min=self.config.Min_M_V, a_max=self.config.Max_M_V)
            angular_v = a_l / self.M_speeds
            self.T_speed = self.action_target[0]
            angular_v_T = self.action_target[1]
            # resultant velocity / make sure that the speeds is along the missiles' heading angles.
            velocity = [self.M_speeds * np.cos(self.alpha), self.M_speeds * np.sin(self.alpha),
                        np.zeros(self.agent_num)]
            velocity = np.stack(velocity).T
            angular_velocity = np.stack([np.zeros(self.agent_num), np.zeros(self.agent_num), angular_v]).T
            for i in range(self.agent_num):
                missile_id = self.swarm_ids[i]
                self.bulletClient.resetBaseVelocity(objectUniqueId=missile_id, linearVelocity=velocity[i],
                                                    angularVelocity=angular_velocity[i])
            # update target velocity
            target_velocity = [self.action_target[0] * np.cos(self.alpha_target),
                               self.action_target[0] * np.sin(self.alpha_target), 0]
            self.bulletClient.resetBaseVelocity(objectUniqueId=self.target_id, linearVelocity=target_velocity,
                                                angularVelocity=[0., 0., angular_v_T])
            # take one step
            self.bulletClient.stepSimulation()

        self.sim_time += self.tau * self.config.frameskip
        self.step_count += self.config.frameskip

        # retrieve and update first order system states
        for id in range(self.agent_num):
            PosAndOrn = self.bulletClient.getBasePositionAndOrientation(bodyUniqueId=self.swarm_ids[id])
            self.M_positions[id] = bullet_To_local_position(PosAndOrn[0])
            self.M_orientations[id] = bullet_To_local_orientation(
                self.bulletClient.getEulerFromQuaternion(PosAndOrn[1]))
            self.alpha[id] = self.M_orientations[id][2]

        self.zero_order_state = np.concatenate((self.M_positions[:, :2], np.expand_dims(self.alpha, axis=1)), axis=1)
        # update target first order state
        PosAndOrn_ref = self.bulletClient.getBasePositionAndOrientation(bodyUniqueId=self.target_id)
        self.T_position = bullet_To_local_position(PosAndOrn_ref[0])
        self.alpha_target = \
            bullet_To_local_target_orientation(self.bulletClient.getEulerFromQuaternion(PosAndOrn_ref[1]))[2]
        self.zero_order_state_ref = np.array([self.T_position[0],
                                              self.T_position[1],
                                              self.alpha_target], dtype=float)

        # terminate condition
        if self.sim_time >= self.config.epoch_time:
            terminate = True
        else:
            terminate = False

        # node failure.
        # agent num will not change, only communication topology will be changed. Failed nodes have no neighbors.
        # node failure applies for both conventional fixed topology and Adaptive_Topology.
        if self.config.Node_Failure != "None":
            if self.step_count == self.topologyChangeStep:
                if self.config.Node_Failure == "OneFail":
                    self.commun_topology, self.cluster_head = get_communication_topology(self.agent_num,
                                                                                         self.config.Topology_Type,
                                                                                         self.formation_def,
                                                                                         failure_nodes=[0, ])
                elif self.config.Node_Failure == "TwoFail":
                    self.commun_topology, self.cluster_head = get_communication_topology(self.agent_num,
                                                                                         self.config.Topology_Type,
                                                                                         self.formation_def,
                                                                                         failure_nodes=[0, 3])
                else:
                    raise NotImplementedError
                self.set_of_neighbors = get_set_of_neighbors_form_adjacent_matrix(self.commun_topology)
                self.miu = np.zeros((self.agent_num,), dtype=float)
                self.miu[self.cluster_head] = 1.

                self.set_of_neighbors_his.append(self.set_of_neighbors)
        # record trajectory
        self.swarm_trajectory.append(self.zero_order_state)
        self.ref_trajectory.append(self.zero_order_state_ref)

        # adjust camera
        if self.render and self.focus_view:
            if self.step_count % 50 == 0:  # update view every * second.
                focus_position = local_To_bullet_position(np.average(self.M_positions, axis=0))
                self.bulletClient.resetDebugVisualizerCamera(cameraDistance=5., cameraYaw=0.,
                                                             cameraPitch=-89.5, cameraTargetPosition=focus_position)

        obs = self.obtain_observation(observe_option="Trans_Signal" if self.barrier_on else None)
        if self.barrier_on:
            if self.formation_signal != obs[1, -1]:  # once change signal detected
                self.formation_signal = obs[1, -1]
                if self.config.switchFormationStrategy == "switchPattern":
                    self.formation_def = self.get_formation_form_pattern(pattern=self.config.formation_pattern,
                                                                         rotate=self.config.formation_rot,
                                                                         gap=self.config.formation_gap) \
                        if self.formation_signal == 0 else self.get_formation_form_pattern(pattern="StraightLine",
                                                                                           rotate=-np.pi / 4,
                                                                                           gap=self.config.formation_gap)
                elif self.config.switchFormationStrategy == "switchSize":
                    self.formation_def = self.get_formation_form_pattern(pattern=self.config.formation_pattern,
                                                                         rotate=self.config.formation_rot,
                                                                         gap=self.config.formation_gap) \
                        if self.formation_signal == 0 else self.get_formation_form_pattern(
                        pattern=self.config.formation_pattern,
                        rotate=self.config.formation_rot,
                        gap=0.2)

        # print(f"formation_def: {self.formation_def}")
        # print(f"Trans signal： {obs[:, -1]}")
        step_reward, early_stop = self.obj_obs.reward_function(action)

        # nonreactive period
        if self.barrier_on:
            if self.nonreactive:
                step_reward = np.zeros(self.agent_num, dtype=float)

        self.cumul_reward += step_reward
        self.st_r_record.append(step_reward)
        if early_stop:
            terminate = True
        return obs, step_reward, terminate

    def obtain_observation(self, observe_option=None):
        """
        Return specified observation,
        calculate resultant error based on current system state and formation_def.
        and Record analytical data
        :param observe_option: if "Trans_Signal", include transformation signal in observation
        :return: ndarray*[(agent_num)*[(6)]]
        """
        f_m_e = formation_maintenance_error(self.zero_order_state, self.formation_def, self.set_of_neighbors)
        r_t_e = reference_tracking_error(self.zero_order_state_ref, self.zero_order_state, self.formation_def, self.miu)

        self.resultant_error = f_m_e + r_t_e
        # base_obs = np.concatenate((self.zero_order_state, self.resultant_error), axis=-1)
        base_obs = np.concatenate((np.expand_dims(self.alpha, 1), np.expand_dims(self.M_speeds, 1),
                                   self.resultant_error), axis=-1)

        if self.analytical:
            # take sum of errors in x and y-axis
            self.resultant_error_record.append(self.resultant_error.sum(-1))
            self.formationMaintenance_error_record.append(f_m_e.sum(-1))
            self.referenceTracking_error_record.append(r_t_e.sum(-1))
            self.m_speed_record.append(copy.deepcopy(self.M_speeds))
            self.m_alpha_record.append(copy.deepcopy(self.alpha))
            self.t_speed_record.append(copy.deepcopy(self.T_speed))
            self.t_alpha_record.append(copy.deepcopy(self.alpha_target))

        if observe_option == None:
            return base_obs
        elif observe_option == "Trans_Signal":
            obs_distance = np.sqrt((self.T_position[1] - self.br_ctr_pos[1]) ** 2 + (
                    self.T_position[0] - self.br_ctr_pos[0]) ** 2)  # subtraction of y coordinates.
            trans_detected = np.empty((self.agent_num, 1))
            trans_detected.fill(1 if obs_distance <= self.min_dis else 0)
            self.nonreactive = True if self.trans_dis <= obs_distance <= self.min_dis else False
            return np.append(base_obs, trans_detected, axis=1)
        else:
            raise NameError

    def close(self):
        # must remember to close
        self.bulletClient.disconnect()

    def get_formation_form_pattern(self, pattern, rotate, gap):
        if pattern == "RegularPolygon":
            assert self.agent_num >= 3, f"Wrong formation pattern for {self.agent_num} agents!"
            radius = gap
            formation_def = np.array([[radius * np.sin(theta), radius * np.cos(theta), 0]
                                      for theta in np.linspace(0, 2 * np.pi, num=self.agent_num, endpoint=False)],
                                     dtype=float)

        elif pattern == "StraightLine":
            tot_length = gap * (self.agent_num - 1)
            formation_def = np.array([[0, ypos, 0] for ypos in
                                      np.linspace(tot_length / 2, -tot_length / 2, num=self.agent_num, endpoint=True)])
        else:
            raise NotImplementedError("Your specified formation shape has not been implemented.")
        formation_def = rotation_matrix(rotate, dim=3) @ formation_def.T
        # filter results
        return np.asarray(formation_def.T)

    def swarm_born(self, condition, formation_def, center_position):
        """
        Note: Initial height is 1 km higher than center_position by default.
        :param condition:
        :param formation_def:
        :return:
        """
        HEIGHT = 1.5
        if condition == "In_Formation":
            init_positions = np.concatenate((formation_def[:, :2], np.full((self.agent_num, 1), HEIGHT, dtype=float)),
                                            axis=1)
            # define center position
            init_positions += np.array([center_position] * self.agent_num)
            init_velocities = np.array([[self.config.Min_M_V, 0., 0.]] * self.agent_num)
            init_angular_velocity = np.array([[0., 0., 0.]] * self.agent_num)
        elif condition == "RandomlySpread":
            # init_positions = np.concatenate((self.config.init_positions,
            #                                  np.full((self.agent_num, 1), HEIGHT, dtype=float)), axis=1)

            init_positions = np.concatenate((np.random.uniform(low=7, high=11, size=(self.agent_num, 1)),
                                             np.random.uniform(low=2, high=5, size=(self.agent_num, 1))),
                                            axis=1).tolist()
            init_positions = np.concatenate((self.config.init_positions,
                                             np.full((self.agent_num, 1), HEIGHT, dtype=float)), axis=1)
            init_velocities = np.array([[self.config.Min_M_V, self.config.Min_M_V, 0]] * self.agent_num)
            init_angular_velocity = np.array([[0., 0., 0.]] * self.agent_num)
        else:
            raise NotImplementedError
        # define init_orientation depending on manoeuvring of the target.
        if self.config.target_mobility_strategy in ["GoStraightAlongY", "GoRandomly"]:
            init_orientations = np.array([[0., 0., np.pi / 2]] * self.agent_num, dtype=float)
        elif self.config.target_mobility_strategy in ["GoStraightAlongX", "GoRoundInCircle", "GoInSpiral",
                                                      "GoInSinusoidal"]:
            init_orientations = np.array([[0., 0., 0.]] * self.agent_num, dtype=float)
        elif self.config.target_mobility_strategy in ["GoALongDiagonal"]:
            init_orientations = np.array([[0., 0., np.pi / 4]] * self.agent_num, dtype=float)
        else:
            raise NameError
        return init_positions, init_orientations, init_velocities, init_angular_velocity

    def target_born(self, condition, center_position):
        HEIGHT = 0.5
        if condition == "Formation_Center":
            init_position = center_position[:2] + [HEIGHT]
            if self.config.target_mobility_strategy == "GoStraightAlongX":
                init_orientation = [0., 0., 0.]
            elif self.config.target_mobility_strategy == "GoStraightAlongY":
                init_orientation = [0., 0., np.pi / 2]
            elif self.config.target_mobility_strategy == "GoALongDiagonal":
                init_orientation = [0., 0., np.pi / 4]
            else:
                init_orientation = [0., 0., 0.]
        elif condition == "AwayFromFormation":
            init_position = [9., 5., HEIGHT]
            if self.config.target_mobility_strategy == "GoStraightAlongX":
                init_orientation = [0., 0., 0.]
            elif self.config.target_mobility_strategy == "GoStraightAlongY":
                init_orientation = [0., 0., np.pi / 2]
            elif self.config.target_mobility_strategy == "GoALongDiagonal":
                init_orientation = [0., 0., np.pi / 4]
            else:
                init_orientation = [0., 0., 0.]
        else:
            raise NotImplementedError
        init_velocity = [0.3, 0.0, 0.]
        init_angular_velocity = [0., 0., 0.]
        return init_position, init_orientation, init_velocity, init_angular_velocity

    def add_debug_elements(self):
        # user Debug line
        coordinate_offset = 0.1
        self.bulletClient.addUserDebugLine(lineFromXYZ=[-10, -10, coordinate_offset],
                                           lineToXYZ=[10, -10, coordinate_offset],
                                           lineColorRGB=[1, 0, 0],
                                           lineWidth=1,
                                           lifeTime=0)
        self.bulletClient.addUserDebugLine(lineFromXYZ=[-10, -10, coordinate_offset],
                                           lineToXYZ=[-10, 10, coordinate_offset],
                                           lineColorRGB=[0, 1, 0],
                                           lineWidth=1,
                                           lifeTime=0)
        self.bulletClient.addUserDebugText(text="X(km)", textPosition=[-8, -9.7, coordinate_offset], textSize=0.5,
                                           textColorRGB=[0.1, 0.1, 0.1])
        self.bulletClient.addUserDebugText(text="Y(km)", textPosition=[-9.7, -8, coordinate_offset], textSize=0.5,
                                           textColorRGB=[0.1, 0.1, 0.1])
        for i in range(21):  # add ticks
            self.bulletClient.addUserDebugText(text=f"{i}", textPosition=[-10 + 1 * i, -10.2, coordinate_offset],
                                               textSize=0.5,
                                               textColorRGB=[0.1, 0.1, 0.1])
            self.bulletClient.addUserDebugText(text=f"{i}", textPosition=[-10.2, -10 + 1 * i, coordinate_offset],
                                               textSize=0.5,
                                               textColorRGB=[0.1, 0.1, 0.1])

    def load_swarm(self, positions, orientations, model_rescale=1e-3):
        """
        load swarm agents defined by .obj model
        Note that problems exist when creating collision-shape with .obj file on Linux platform,
        so this part will be omitted.
        position unit: km
        angular unit: rad
        :return: List[agent_id]
        """

        if platform.system() == "Linux":
            visualShapeId = self.bulletClient.createVisualShape(shapeType=self.bulletClient.GEOM_CAPSULE,
                                                                radius=0.02,
                                                                length=0.1,
                                                                rgbaColor=[0, 0, 0, 1])
            # collisionShapeId = self.bulletClient.createCollisionShape(shapeType=self.bulletClient.GEOM_CAPSULE,
            #                                                           radius=0.0002,
            #                                                           height=0.001
            #                                                           )

            swarm_ids = []
            for i in range(self.agent_num):
                missile_id = self.bulletClient.createMultiBody(baseMass=1,
                                                               baseInertialFramePosition=[0, 0, 0],
                                                               baseVisualShapeIndex=visualShapeId,
                                                               basePosition=local_To_bullet_position(positions[i]),
                                                               baseOrientation=self.bulletClient.getQuaternionFromEuler(
                                                                   local_To_bullet_orientation(orientations[i])),
                                                               useMaximalCoordinates=True)
                swarm_ids.append(missile_id)

        else:
            meshscale = [model_rescale] * 3
            shift = [0, -0.22, 0]
            visualShapeId = self.bulletClient.createVisualShape(shapeType=self.bulletClient.GEOM_MESH,
                                                                fileName="../src/missile.obj",
                                                                rgbaColor=[0, 0, 0, 1],
                                                                specularColor=[0.4, 0.4, 0],
                                                                visualFramePosition=shift,
                                                                meshScale=meshscale,
                                                                )
            collisionShapeId = self.bulletClient.createCollisionShape(shapeType=self.bulletClient.GEOM_MESH,
                                                                      fileName="../src/missile.obj",
                                                                      collisionFramePosition=shift,
                                                                      meshScale=meshscale,
                                                                      )

            swarm_ids = []
            for i in range(self.agent_num):
                missile_id = self.bulletClient.createMultiBody(baseMass=1,
                                                               baseInertialFramePosition=[0, 0, 0],
                                                               baseCollisionShapeIndex=collisionShapeId,
                                                               baseVisualShapeIndex=visualShapeId,
                                                               basePosition=local_To_bullet_position(positions[i]),
                                                               baseOrientation=self.bulletClient.getQuaternionFromEuler(
                                                                   local_To_bullet_orientation(orientations[i])),
                                                               useMaximalCoordinates=True)
                swarm_ids.append(missile_id)
        return swarm_ids

    def load_target(self, position, orientation, model_scale=1e-5):
        if platform.system() == "Linux":
            visualShapeId = self.bulletClient.createVisualShape(shapeType=self.bulletClient.GEOM_SPHERE,
                                                                radius=0.05,
                                                                rgbaColor=[0, 0, 0, 1])
            collisionShapeId = self.bulletClient.createCollisionShape(shapeType=self.bulletClient.GEOM_SPHERE,
                                                                      radius=0.005,
                                                                      )
        else:
            meshscale = [model_scale] * 3
            shift = [0, 0, -0.2]
            visualShapeId = self.bulletClient.createVisualShape(shapeType=self.bulletClient.GEOM_MESH,
                                                                fileName="../src/submarine_v4.obj",
                                                                rgbaColor=[0, 1, 0, 1],
                                                                specularColor=[0.4, .4, 0],
                                                                visualFramePosition=shift,
                                                                meshScale=meshscale,
                                                                )
            collisionShapeId = self.bulletClient.createCollisionShape(shapeType=self.bulletClient.GEOM_MESH,
                                                                      fileName="../src/submarine_v4.obj",
                                                                      collisionFramePosition=shift,
                                                                      meshScale=meshscale,
                                                                      )

        target_id = self.bulletClient.createMultiBody(baseMass=1,
                                                      baseInertialFramePosition=[0, 0, 0],
                                                      baseCollisionShapeIndex=collisionShapeId,
                                                      # no collision for target
                                                      baseVisualShapeIndex=visualShapeId,
                                                      basePosition=local_To_bullet_position(position),
                                                      baseOrientation=self.bulletClient.getQuaternionFromEuler(
                                                          local_To_bullet_target_orientation(orientation)),
                                                      useMaximalCoordinates=True)
        return target_id

    def load_barrier(self, size, center_position, gap_width):
        """
        :param size:
        :param center_position:
        :param gap_width:
        :return: Coordinates of the lower left corner of the rectangles
        """
        visualShapeId = self.bulletClient.createVisualShape(shapeType=self.bulletClient.GEOM_BOX,
                                                            halfExtents=size,
                                                            rgbaColor=[0., 1., 0., 1.])
        collisionShapeId = self.bulletClient.createCollisionShape(shapeType=self.bulletClient.GEOM_BOX,
                                                                  halfExtents=size, )

        position_L = np.array(center_position) + [-(gap_width / 2 + size[1]), 0., 0.]
        position_R = np.array(center_position) + [(gap_width / 2 + size[1]), 0., 0.]
        orientation = [0., 0., 0.]

        barrierL_id = self.bulletClient.createMultiBody(baseMass=100,
                                                        baseInertialFramePosition=[0, 0, 0],
                                                        baseCollisionShapeIndex=collisionShapeId,
                                                        baseVisualShapeIndex=visualShapeId,
                                                        basePosition=local_To_bullet_position(position_L),
                                                        baseOrientation=self.bulletClient.getQuaternionFromEuler(
                                                            local_To_bullet_target_orientation(orientation)),
                                                        useMaximalCoordinates=True)

        barrierR_id = self.bulletClient.createMultiBody(baseMass=100,
                                                        baseInertialFramePosition=[0, 0, 0],
                                                        baseCollisionShapeIndex=collisionShapeId,
                                                        baseVisualShapeIndex=visualShapeId,
                                                        basePosition=local_To_bullet_position(position_R),
                                                        baseOrientation=self.bulletClient.getQuaternionFromEuler(
                                                            local_To_bullet_target_orientation(orientation)),
                                                        useMaximalCoordinates=True)
        return position_L[:2] + [-size[1], -size[0]], position_R[:2] + [-size[1], -size[0]]

    def snap_shot(self, suffix, save_path=None):
        p = Process(target=self._snap_shot, args=(self, suffix, save_path))
        p.start()
        p.join()

    def plot_analytic_data(self, suffix, save_path=None):
        p = Process(target=self._plot_analytic_data, args=(self, suffix, save_path))
        p.start()
        p.join()

    # plotting area: All plot functions are decorated with @staticmethod to be feasible in multiprocessing runtime.
    @staticmethod
    def _snap_shot(env, suffix, save_path=None):
        """
        Plot trajectory of the swarm, save in svg file.
        Self-adaptive world size.
        Note that this function cost a bit more time, so do not call it frequently.
        :return: None
        """
        env.swarm_trajectory = np.stack(env.swarm_trajectory, axis=0)

        env.ref_trajectory = np.stack(env.ref_trajectory, axis=0)
        plt.figure(figsize=(8, 8), dpi=150)
        plt.cla()
        if suffix != "formal":
            plt.title("Swarm trajectory")

        screen_width = 900
        screen_height = 900
        world_width = world_width = env.config.World_Size[0][1] - env.config.World_Size[0][0]
        lowest = np.round(np.min(env.swarm_trajectory[:, :, :2]))
        highest = np.round(np.max(env.swarm_trajectory[:, :, :2]))
        World_Size = [[lowest - 3, lowest - 3], [highest + 3, highest + 3]]

        print(World_Size)
        # plot target
        plt.scatter(env.zero_order_state_ref[0], env.zero_order_state_ref[1], color=[0.8, 0.1, 0.1], s=100, marker="*",
                    label="Target", alpha=0.5)

        if suffix != "formal":
            # plot notation
            plt.text(0.2, 19.5, f"time(s): {np.round(env.sim_time, decimals=2)}", fontsize=10)
            plt.text(0.2, 19, f"cumulative reward: {np.round(env.cumul_reward, decimals=2)}")

        # plot trajectories
        for i in range(env.agent_num):
            # slice the trajectory of the failed nodes
            if env.config.Node_Failure != "None" and env.step_count >= env.topologyChangeStep \
                    and not any(i in sublist for sublist in env.set_of_neighbors_his[-1]) \
                    and len(env.set_of_neighbors_his[-1][i]) == 0:
                traj_of_one_node = env.swarm_trajectory[:env.topologyChangeStep, i]
            else:
                traj_of_one_node = env.swarm_trajectory[:, i]
            plt.plot(traj_of_one_node[:, 0], traj_of_one_node[:, 1], linestyle="dashed", linewidth=1,
                     color=env.palette[i], alpha=0.5)
        plt.plot(env.ref_trajectory[:, 0], env.ref_trajectory[:, 1], linestyle="dashed", linewidth=0.5,
                 color=[0.8, 0.1, 0.1])

        # plot missiles at both terminate and historical state
        steps = np.linspace(0, env.step_count / env.config.frameskip, num=5, endpoint=True, dtype=int)
        for step in steps:
            for i in range(env.agent_num):
                if env.config.Node_Failure != "None" and step > env.topologyChangeStep and not any(i in sublist for sublist in env.set_of_neighbors_his[-1]) \
                        and len(env.set_of_neighbors_his[-1][i]) == 0:
                    pass
                else:
                    plt.scatter(env.swarm_trajectory[step, i, 0], env.swarm_trajectory[step, i, 1],
                                color=env.palette[i, :], s=100, marker=gen_arrow_head_marker(
                            env.swarm_trajectory[step, i, 2]), alpha=1.0, zorder=10)

        # plot legend
        for i in range(env.agent_num):
            plt.scatter(-100, -100,
                        color=env.palette[i, :], s=100, marker=gen_arrow_head_marker(0), label=fr"$M_{i + 1}$",
                        alpha=1.)

        # plot communication relationship debug line
        if env.config.Node_Failure == "None" or env.step_count < env.topologyChangeStep:
            edges = []
            for i in range(env.agent_num):
                for j in env.set_of_neighbors[i]:
                    if (i, j) not in edges and (j, i) not in edges:
                        edges.append((i, j))
            for step in steps:
                for edge in edges:
                    plt.plot([env.swarm_trajectory[step, edge[0], 0], env.swarm_trajectory[step, edge[1], 0]],
                             [env.swarm_trajectory[step, edge[0], 1], env.swarm_trajectory[step, edge[1], 1]],
                             color=[1, 0, 0], linestyle="solid", linewidth=0.6, alpha=0.3, zorder=11)
        else:
            listOfEdges = []
            assert len(env.set_of_neighbors_his) >= 2, "Neighbors set has not been changed!"
            for set_of_neighbors in env.set_of_neighbors_his:
                edges = []
                for i in range(env.agent_num):
                    for j in set_of_neighbors[i]:
                        if (i, j) not in edges and (j, i) not in edges:
                            edges.append((i, j))
                listOfEdges.append(edges)

            for step in steps:
                if step <= env.topologyChangeStep:
                    edges = listOfEdges[0]
                else:
                    edges = listOfEdges[1]
                for edge in edges:
                    plt.plot([env.swarm_trajectory[step, edge[0], 0], env.swarm_trajectory[step, edge[1], 0]],
                             [env.swarm_trajectory[step, edge[0], 1], env.swarm_trajectory[step, edge[1], 1]],
                             color=[1, 0, 0], linestyle="solid", linewidth=0.6, alpha=0.3, zorder=11)

        # plot barriers (if there are)
        if env.barrier_on:
            b1 = plt.Rectangle(xy=env.LBR_POS, width=env.barrier_size[1] * 2, height=env.barrier_size[0] * 2)
            b2 = plt.Rectangle(xy=env.RBR_POS, width=env.barrier_size[1] * 2, height=env.barrier_size[0] * 2)
            plt.gcf().gca().add_artist(b1)
            plt.gcf().gca().add_artist(b2)

        ## format setup
        # set x axis
        # set x axis
        plt.xlabel(r"$X$(km)")
        plt.xlim(World_Size[0][0], World_Size[1][0])
        plt.xticks(np.arange(World_Size[0][0], World_Size[1][0], 1), rotation=45, fontsize=12)

        # set y axis
        plt.ylabel(r"$Y$(km)")
        plt.ylim(World_Size[0][1], World_Size[1][1])
        plt.yticks(np.arange(World_Size[0][1], World_Size[1][1], 1), fontsize=12)
        plt.legend(fontsize=15, frameon=False, fancybox=False)

        if save_path is None:
            plt.savefig(os.path.join(ROOT_DIR, env.config.replay_dir, f"Snapshot_{suffix}.svg"), format='svg')
        else:
            plt.savefig(os.path.join(save_path, f"Snapshot_{suffix}.svg"), format='svg')
        plt.close()

    @staticmethod
    def _plot_analytic_data(env, pointer, save_path=None):
        plt.figure()

        r_e_record = np.stack(env.resultant_error_record, axis=0)
        f_m_e_record = np.stack(env.formationMaintenance_error_record, axis=0)
        r_t_e_record = np.stack(env.referenceTracking_error_record, axis=0)
        m_s_record = np.stack(env.m_speed_record, axis=0)
        m_a_record = np.stack(env.m_alpha_record, axis=0)
        t_s_record = np.array(env.t_speed_record)
        t_a_record = np.array(env.t_alpha_record)
        st_r_record = np.stack(env.st_r_record, axis=0)
        args = env.config

        assert r_e_record.ndim == 2 & f_m_e_record.ndim == 2 & r_t_e_record.ndim == 2 & m_s_record.ndim == 2 & \
               m_a_record.ndim == 2
        # r_e_record
        plt.cla()
        plt.xlabel("time(s)")
        plt.xticks(np.arange(0, len(r_e_record), 3 / args.tau),  # unit: 3 seconds
                   labels=np.arange(0, len(r_e_record), 3 / args.tau) * args.tau)
        plt.ylabel(r'$|e_{ri}|$')
        plt.ylim(np.min(r_e_record) - 1, np.max(r_e_record) + 1)
        plt.plot(range(len(r_e_record)), r_e_record, label=[fr'$M_{i + 1}$' for i in range(len(r_e_record[0]))])
        plt.legend()
        if env.topologyChangeStep is not None:
            plt.text(x=env.topologyChangeStep-100, y=-0.4, s="node failure", fontsize=10, fontstyle='italic')
            # plt.arrow(x=env.topologyChangeStep-10, y=-0.3, dx=10, dy=0.1, width=0.03)
            plt.annotate("", xy=(env.topologyChangeStep, -0.15), xytext=(env.topologyChangeStep-30, -0.3), arrowprops=dict(arrowstyle="->"))
            # 局部放大图
            ax = plt.gca()
            axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                               bbox_to_anchor=(0.5, 0.1, 0.8, 0.8),
                               bbox_transform=ax.transAxes)
            axins.plot(range(len(r_e_record)), r_e_record)
            axins.set_xlim(18/args.tau, 22/args.tau)

        if save_path is None:
            plt.savefig(os.path.join(ROOT_DIR, args.result_dir, f"resultant_error_{pointer}.svg"), format='svg')
        else:
            plt.savefig(os.path.join(save_path, f"resultant_error_{pointer}.svg"), format='svg')
        # f_m_record
        plt.cla()
        plt.xlabel("time(s)")
        plt.xticks(np.arange(0, len(f_m_e_record), 3 / args.tau),  # unit: 3 seconds
                   labels=np.arange(0, len(f_m_e_record), 3 / args.tau) * args.tau)
        plt.ylabel(r'$e_f$')
        plt.plot(range(len(f_m_e_record)), f_m_e_record, label=[fr'$M_{i + 1}$' for i in range(len(f_m_e_record[0]))])
        plt.legend()
        if save_path is None:
            plt.savefig(os.path.join(ROOT_DIR, args.result_dir, f"formationMaintenance_error_{pointer}.svg"),
                        format='svg')
        else:
            plt.savefig(os.path.join(save_path, f"formationMaintenance_error_{pointer}.svg"), format='svg')
        # r_t_error
        plt.cla()
        plt.xlabel("time(s)")
        plt.xticks(np.arange(0, len(r_t_e_record), 3 / args.tau),  # unit: 3 seconds
                   labels=np.arange(0, len(r_t_e_record), 3 / args.tau) * args.tau)
        plt.ylabel(r'$e_t$')
        plt.plot(range(len(r_t_e_record)), r_t_e_record, label=[fr'$M_{i + 1}$' for i in range(len(r_t_e_record[0]))])
        plt.legend()
        if save_path is None:
            plt.savefig(os.path.join(ROOT_DIR, args.result_dir, f"formationTracking_error_{pointer}.svg"), format='svg')
        else:
            plt.savefig(os.path.join(save_path, f"formationTracking_error_{pointer}.svg"), format='svg')
        # m_s_record
        plt.cla()
        plt.xlabel("time(s)")
        plt.xticks(np.arange(0, len(m_s_record), 3 / args.tau),  # unit: 3 seconds
                   labels=np.arange(0, len(m_s_record), 3 / args.tau) * args.tau)
        plt.ylabel(r'$speed (km/s)$')
        plt.plot(range(len(m_s_record)), m_s_record, label=[fr'$M_{i + 1}$' for i in range(len(m_s_record[0]))])
        plt.plot(range(len(t_s_record)), t_s_record, label="Reference Target")
        plt.legend()
        if save_path is None:
            plt.savefig(os.path.join(ROOT_DIR, args.result_dir, f"missileSpeeds_{pointer}.svg"), format='svg')
        else:
            plt.savefig(os.path.join(save_path, f"missileSpeeds_{pointer}.svg"), format='svg')
        # m_a_record
        plt.cla()
        plt.xlabel("time(s)")
        plt.xticks(np.arange(0, len(m_a_record), 3 / args.tau),  # unit: 3 seconds
                   labels=np.arange(0, len(m_a_record), 3 / args.tau) * args.tau)
        plt.ylabel(r'$\alpha(\degree)$')
        plt.ylim(np.min(m_a_record * 180 / np.pi) - 0.5, np.max(m_a_record * 180 / np.pi) + 0.5)

        plt.plot(range(len(m_a_record)), m_a_record * 180 / np.pi,
                 label=[fr'$M_{i + 1}$' for i in range(len(m_a_record[0]))])
        plt.plot(range(len(t_a_record)), t_a_record * 180 / np.pi, label="Reference Target")
        plt.legend()
        if save_path is None:
            plt.savefig(os.path.join(ROOT_DIR, args.result_dir, f"missileAlphaAngle_{pointer}.svg"), format='svg')
        else:
            plt.savefig(os.path.join(save_path, f"missileAlphaAngle_{pointer}.svg"), format='svg')

        # st_r_record
        plt.cla()
        plt.xlabel("time(s)")
        plt.xticks(np.arange(0, len(st_r_record), 3 / args.tau),  # unit: 3 seconds
                   labels=np.arange(0, len(st_r_record), 3 / args.tau) * args.tau)
        plt.plot(range(len(st_r_record)), st_r_record,
                 label=[fr'$M_{i + 1}$' for i in range(len(st_r_record[0]))])
        plt.legend()
        if save_path is None:
            plt.savefig(os.path.join(ROOT_DIR, args.result_dir, f"reward_Curve_{pointer}.svg"), format='svg')
        else:
            plt.savefig(os.path.join(save_path, f"reward_Curve_{pointer}.svg"), format='svg')

        plt.close()


def simple_simulate():
    args = Common.Config.get_args_from_json()
    env = swarmMissileEnv(args, analytical=True)
    env.reset()
    terminated = False
    while not terminated:
        action = [[0.1] * env.agent_num, [0.] * env.agent_num]
        obs, r, terminated = env.step(action)
        # print(r)
    env.snap_shot("test")
    env.plot_analytic_data(suffix="test")
    env.close()


# sim_start_time = 0
# for i in range(10000):
#     self.bulletClient.stepSimulation()
#     time.sleep(time_step)  # sleep in order to present a RealTime rendering view/ simulation stagnate during sleep
#     print(f"box position: {local_To_bullet_position(self.bulletClient.getBasePositionAndOrientation(missile_id)[0])}")
#     print(f"box velocity: {self.bulletClient.getBaseVelocity(missile_id)[0]}")
#     print(f"simulation time: {time_step*(i+1)}")

if __name__ == '__main__':

    # config = get_default_config()
    config = get_args_from_json()
    # config.target_mobility_strategy = "GoRoundInCircle"
    # config.agent_num = 5
    # config.formation_pattern = "RegularPolygon"
    # config.Topology_Type = "AdaptiveTopology"
    # config.Node_Failure = "OneFail"

    env = swarmMissileEnv(config=config, render=False, focus_view=False, analytical=True)
    env.reset()
    terminate = False
    print(f"initial zero order state: {env.zero_order_state}")
    print(f"initial zero order state of target: {env.zero_order_state_ref}")
    print(f"Communication Topology: \n{env.commun_topology}")

    while not terminate:
        action = [[0.1] * env.agent_num, [0.] * env.agent_num]
        obs, r, terminate = env.step(action=action)
        # print(f"Communication Topology: \n{env.commun_topology}")
        # print(f"resultant error: {env.resultant_error}")
        # print(f"step reward: {r}")
        # time.sleep(env.tau*1)

    env.snap_shot("test")
    env.plot_analytic_data(suffix="test")
    env.close()
    print(f"total sim time:  {env.sim_time}")
