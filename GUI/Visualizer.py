import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtGui, QtCore

from Common.Config import get_default_config
from environment.SwarmEnv import swarmMissileEnv


class Oscilloscope(object):
    def __init__(self, env, horizon, Items):

        pg.setConfigOptions(antialias=True)
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle('Missile flight Data')
        self.Items = Items

        self.all_plots = []
        self.all_data = []
        self.all_curve = []
        for index, item in enumerate(Items):
            data = np.zeros((horizon,), dtype=float)
            p = self.win.addPlot(title=f'{item}', row=index+1, col=1)
            curve = p.plot(data)
            self.all_plots.append(p)
            self.all_data.append(data)
            self.all_curve.append(curve)

        self.ptr = 0
        self.item_num = len(Items)

        self.env = env

    def start(self):

        # environment initial setup
        self.env.reset()
        self.terminate = False

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    def update(self):

        # step simulation
        if not self.terminate:
            # piecewise control------------------
            # if self.env.sim_time <= 10:
            #     action = [[0.2] * self.env.agent_num, [0.5] * self.env.agent_num]
            # else:
            #     action = [[-0.2] * self.env.agent_num, [0.5] * self.env.agent_num]
            action = [[0.1] * self.env.agent_num, [0.] * self.env.agent_num]
            # ------------------------------------
            obs, r, self.terminate = self.env.step(action=action)

        else:
            print("Simulation End")
            self.env.snap_shot("test")
            self.env.close()
            sys.exit()
        input_list = []
        for attr in self.Items:
            input = getattr(self.env, attr) # speed of first missile
            input = np.around(input, decimals=4) # filter extreme value
            dim = len(np.shape(input))
            for i in range(dim): # only retrieve the first value in first dimension
                input = input[0]

            input_list.append(input)

        assert len(input_list) == self.item_num, "Input data does not correspond with " \
                                            "plot nums"
        # refresh display data
        for index, data in enumerate(self.all_data):
            # shift data
            data[:-1] = data[1:]
            data[-1] = input_list[index]

            self.ptr += 1
            # refresh curves
            self.all_curve[index].setData(data)
            self.all_curve[index].setPos(self.ptr, 0)

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(int(self.env.tau*1000*5))
        self.start()


if __name__ == '__main__':
    config = get_default_config()
    config.target_mobility_strategy = "GoRoundInCircle"
    config.agent_num = 6
    env = swarmMissileEnv(config=config, render=True, focus_view=False)

    vis = Oscilloscope(env, 100, ["M_positions", "T_position", "action_target", "step_count"])
    vis.animation()

