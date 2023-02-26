import sys

sys.path.append("..")

import os
import shutil
import numpy as np

from Common.Definition import ROOT_DIR
from environment.SwarmEnv import simple_simulate

from Common.Config import get_args_from_json, set_json_from_dict
from Train import Natural_Evoltuion_Strategy

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QSize, QThread, QCoreApplication
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateEdit, QDateTimeEdit,
                             QDial, QDoubleSpinBox, QFontComboBox, QLabel, QLCDNumber, QLineEdit, QMainWindow,
                             QProgressBar, QPushButton, QRadioButton, QSlider, QSpinBox, QTimeEdit, QVBoxLayout,
                             QHBoxLayout, QWidget, QMessageBox, QToolBar, QAction, QStatusBar)


class MainWindow(QMainWindow):
    EXIT_CODE_REBOOT = -123

    def __init__(self, args):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Swarm Formation Control")
        self.show()
        self.args = args
        self.args_dict = vars(args)

        # load config

        toolbar1 = QToolBar("load config toolbar2")
        toolbar1.setIconSize(QSize(30, 30))
        self.addToolBar(toolbar1)
        toolbar1.addWidget(QLabel("Load Configuration:"))

        button_action_2 = QAction(QIcon("../src/blue-document-number-1.png"), "&Case 1", self)
        button_action_2.setStatusTip("load config for case 1")
        button_action_2.triggered.connect(lambda state, case="case1": self.onChooseCaseClick(case))
        button_action_2.setCheckable(True)
        button_action_2.setChecked(False)
        toolbar1.addAction(button_action_2)
        #
        toolbar1.addSeparator()
        #

        button_action_2 = QAction(QIcon("../src/blue-document-number-2.png"), "&Case 2", self)
        button_action_2.setStatusTip("load config for case 2")
        button_action_2.triggered.connect(lambda state, case="case2": self.onChooseCaseClick(case))
        button_action_2.setCheckable(True)
        button_action_2.setChecked(False)
        toolbar1.addAction(button_action_2)
        #
        toolbar1.addSeparator()

        button_action_3 = QAction(QIcon("../src/blue-document-number-3.png"), "&Case 3", self)
        button_action_3.setStatusTip("load config for case 3")
        button_action_3.triggered.connect(lambda state, case="case3": self.onChooseCaseClick(case))
        button_action_3.setCheckable(True)
        button_action_3.setChecked(False)
        toolbar1.addAction(button_action_3)
        #
        toolbar1.addSeparator()
        #

        button_action_4 = QAction(QIcon("../src/node.png"), "&Case node failure", self)
        button_action_4.setStatusTip("load config for case node failure")
        button_action_4.triggered.connect(lambda state, case="caseNodeFailure": self.onChooseCaseClick(case))
        button_action_4.setCheckable(True)
        button_action_4.setChecked(False)
        toolbar1.addAction(button_action_4)
        #
        toolbar1.addSeparator()
        #

        # save config

        toolbar2 = QToolBar("save config toolbar2")
        toolbar2.setIconSize(QSize(30, 30))
        self.addToolBar(toolbar2)
        toolbar2.addWidget(QLabel("Save to Configuration:"))

        button_action_2 = QAction(QIcon("../src/blue-document-number-1.png"), "&Case 1", self)
        button_action_2.setStatusTip("save to case1.json")
        button_action_2.triggered.connect(lambda state, case="case1": self.onSaveConfigClick(case))
        button_action_2.setCheckable(True)
        button_action_2.setChecked(False)
        toolbar2.addAction(button_action_2)
        #
        toolbar2.addSeparator()
        #

        button_action_2 = QAction(QIcon("../src/blue-document-number-2.png"), "&Case 2", self)
        button_action_2.setStatusTip("save to case2.json")
        button_action_2.triggered.connect(lambda state, case="case2": self.onSaveConfigClick(case))
        button_action_2.setCheckable(True)
        button_action_2.setChecked(False)
        toolbar2.addAction(button_action_2)
        #
        toolbar2.addSeparator()

        button_action_3 = QAction(QIcon("../src/blue-document-number-3.png"), "&Case 3", self)
        button_action_3.setStatusTip("save to case3.json")
        button_action_3.triggered.connect(lambda state, case="case3": self.onSaveConfigClick(case))
        button_action_3.setCheckable(True)
        button_action_3.setChecked(False)
        toolbar2.addAction(button_action_3)
        #
        toolbar2.addSeparator()

        button_action_4 = QAction(QIcon("../src/node.png"), "&Case node failure ", self)
        button_action_4.setStatusTip("save to caseNodeFailure.json")
        button_action_4.triggered.connect(lambda state, case="caseNodeFailure": self.onSaveConfigClick(case))
        button_action_4.setCheckable(True)
        button_action_4.setChecked(False)
        toolbar2.addAction(button_action_4)
        #
        toolbar2.addSeparator()
        #

        self.setStatusBar(QStatusBar(self))

        # optional parameters
        OptionalSetup_Layouts = []
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Optional Parameters"))
        OptionalSetup_Layouts.append(title_layout)
        # specific widgets
        for k, v in self.args_dict.items():
            if k == "frameskip":
                widget_layout = QHBoxLayout()
                # widget type
                self.GAP_widget = QSpinBox()
                # widget format
                self.GAP_widget.setValue(v)
                self.GAP_widget.setMaximum(100)
                self.GAP_widget.setMaximumWidth(70)
                # connect
                self.GAP_widget.valueChanged.connect(lambda new_value: self._set_value("frameskip", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.GAP_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "tau":
                widget_layout = QHBoxLayout()
                # widget type
                self.GAP_widget = QDoubleSpinBox()
                # widget format
                self.GAP_widget.setDecimals(4)
                self.GAP_widget.setValue(v)
                self.GAP_widget.setMaximum(1)
                self.GAP_widget.setMaximumWidth(100)
                # connect
                self.GAP_widget.valueChanged.connect(lambda new_value: self._set_value("tau", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.GAP_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "epoch_time":
                widget_layout = QHBoxLayout()
                # widget type
                self.EPT_widget = QDoubleSpinBox()
                # widget format
                self.EPT_widget.setDecimals(1)
                self.EPT_widget.setValue(v)
                self.EPT_widget.setMaximum(50)
                self.EPT_widget.setMaximumWidth(100)
                # connect
                self.EPT_widget.valueChanged.connect(lambda new_value: self._set_value("epoch_time", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.EPT_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "agent_num":
                widget_layout = QHBoxLayout()
                # widget type
                self.n_agents_widget = QSpinBox()
                # widget format
                self.n_agents_widget.setValue(v)
                self.n_agents_widget.setMaximum(10)
                self.n_agents_widget.setMaximumWidth(70)
                # connect
                self.n_agents_widget.valueChanged.connect(lambda new_value: self._set_value("agent_num", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.n_agents_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "swarm_born_condition":
                widget_layout = QHBoxLayout()
                # widget type
                self.BP_widget = QComboBox()
                # widget format
                self.BP_widget.addItems(["In_Formation",
                                         "RandomlySpread"
                                         ])
                self.BP_widget.setCurrentText(v)
                self.BP_widget.setMaximumWidth(200)
                # connect
                self.BP_widget.currentTextChanged.connect(
                    lambda new_value: self._set_value("swarm_born_condition", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.BP_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "formation_pattern":
                widget_layout = QHBoxLayout()
                # widget type
                self.FP_widget = QComboBox()
                # widget format
                self.FP_widget.addItems(["RegularPolygon",
                                         "StraightLine"
                                         ])
                self.FP_widget.setCurrentText(v)
                self.FP_widget.setMaximumWidth(200)
                # connect
                self.FP_widget.currentTextChanged.connect(
                    lambda new_value: self._set_value("formation_pattern", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.FP_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "formation_gap":
                widget_layout = QHBoxLayout()
                # widget type
                self.GAP_widget = QDoubleSpinBox()
                # widget format
                self.GAP_widget.setDecimals(2)
                self.GAP_widget.setMaximum(3)
                self.GAP_widget.setMaximumWidth(100)
                self.GAP_widget.setValue(v)
                # connect
                self.GAP_widget.valueChanged.connect(lambda new_value: self._set_value("formation_gap", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.GAP_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "formation_rot":
                widget_layout = QHBoxLayout()
                # widget type
                self.FRT_widget = QDoubleSpinBox()
                # widget format
                # display in degree
                self.FRT_widget.setDecimals(1)
                self.FRT_widget.setMaximum(180.)
                self.FRT_widget.setMinimum(-180.)
                self.FRT_widget.setValue(v * 180. / np.pi)
                self.FRT_widget.setMaximumWidth(100)
                # connect
                self.FRT_widget.valueChanged.connect(
                    lambda new_value: self._set_value("formation_rot", new_value / 180. * np.pi))
                # label
                label = QLabel(k + ('(degree)'))
                widget_layout.addWidget(self.FRT_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "target_mobility_strategy":
                widget_layout = QHBoxLayout()
                # widget type
                self.TM_widget = QComboBox()
                # widget format
                self.TM_widget.addItems(
                    ["GoStraightAlongX", "GoStraightAlongY", "GoALongDiagonal", "GoRoundInCircle", "GoInSpiral",
                     "GoRandomly", "GoInSinusoidal"])
                self.TM_widget.setCurrentText(v)
                self.TM_widget.setMaximumWidth(200)
                # connect
                self.TM_widget.currentTextChanged.connect(
                    lambda new_value: self._set_value("target_mobility_strategy", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.TM_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)
            if k == "target_born_condition":
                widget_layout = QHBoxLayout()
                # widget type
                self.TB_widget = QComboBox()
                # widget format
                self.TB_widget.addItems(["Formation_Center", "AwayFromFormation"])
                self.TB_widget.setCurrentText(v)
                self.TB_widget.setMaximumWidth(200)
                # connect
                self.TB_widget.currentTextChanged.connect(
                    lambda new_value: self._set_value("target_born_condition", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.TB_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "Topology_Type":
                widget_layout = QHBoxLayout()
                # widget type
                self.CT_widget = QComboBox()
                # widget format
                self.CT_widget.addItems(["Undirected", "AdaptiveTopology"])
                self.CT_widget.setCurrentText(v)
                self.CT_widget.setMaximumWidth(150)
                # connect
                self.CT_widget.currentTextChanged.connect(lambda new_value: self._set_value("Topology_Type", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.CT_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)
            if k == "Node_Failure":
                widget_layout = QHBoxLayout()
                # widget type
                self.NF_widget = QComboBox()
                # widget format
                self.NF_widget.addItems(["None", "OneFail", "TwoFail"])
                self.NF_widget.setCurrentText(v)
                self.NF_widget.setMaximumWidth(150)
                # connect
                self.NF_widget.currentTextChanged.connect(lambda new_value: self._set_value("Node_Failure", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.NF_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            if k == "policy_hidden_size":
                widget_layout = QHBoxLayout()
                # widget type
                self.PHS_widget = QLineEdit()
                # widget format
                self.PHS_widget.setText(str(v))
                self.PHS_widget.setMaximumWidth(150)
                self.PHS_widget.setMinimumWidth(100)
                self.PHS_widget.setDisabled(True)
                # # connect
                # self.PHS_widget.textChanged.connect(
                #     lambda new_value: self._set_value("policy_hidden_size", [int(new_value[:2]), int(new_value[-2:])]))

                # label
                label = QLabel(k)
                widget_layout.addWidget(self.PHS_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)
            if k == "Network_Type":
                widget_layout = QHBoxLayout()
                # widget type
                self.NT_widget = QComboBox()
                # widget format
                self.NT_widget.addItems(["MLP",
                                         "RNN"])
                self.NT_widget.setCurrentText(v)
                self.NT_widget.setMaximumWidth(150)
                # connect
                self.NT_widget.currentTextChanged.connect(lambda new_value: self._set_value("network_type", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.NT_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)
            if k == "sigma_decay":
                widget_layout = QHBoxLayout()
                # widget type
                self.SD_widget = QLineEdit()
                # widget format
                self.SD_widget.setInputMask('0e-00')
                self.SD_widget.setText(format(v, '.0e'))
                self.SD_widget.setMaximumWidth(60)
                # connect
                self.SD_widget.textChanged.connect(lambda new_value: self._set_value("sigma_decay", float(new_value)))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.SD_widget)
                widget_layout.addWidget(label)
                OptionalSetup_Layouts.append(widget_layout)

            # contain first column widgets
        first_column_layout = QVBoxLayout()

        for sub_layout in OptionalSetup_Layouts:
            first_column_layout.addLayout(sub_layout)
        # algorithm core
        algorithm_setup_layouts = []
        # part title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Algorithmic Parameters"))
        algorithm_setup_layouts.append(title_layout)
        # add widgets
        for k, v in self.args_dict.items():

            if k == "Fitness_Reshape":
                widget_layout = QHBoxLayout()
                # widget type
                self.FR_widget = QCheckBox("Fitness_Reshape")
                # widget format
                self.FR_widget.setCheckable(True)
                self.FR_widget.setChecked(v)
                # connect
                self.FR_widget.stateChanged.connect(
                    lambda new_value: self._set_value("Fitness_Reshape", False if new_value == 0 else True))
                # label
                widget_layout.addWidget(self.FR_widget)
                algorithm_setup_layouts.append(widget_layout)

            if k == "population_adaptation":
                widget_layout = QHBoxLayout()
                # widget type
                self.PopAdapt_widget = QCheckBox("population adaptation")
                # widget format
                self.PopAdapt_widget.setCheckable(True)
                self.PopAdapt_widget.setChecked(v)
                # connect
                self.PopAdapt_widget.stateChanged.connect(
                    lambda new_value: self._set_value("population_adaptation", False if new_value == 0 else True))
                # label
                widget_layout.addWidget(self.PopAdapt_widget)
                algorithm_setup_layouts.append(widget_layout)
            if k == "early_stop":
                widget_layout = QHBoxLayout()
                # widget type
                self.EarStp_widget = QCheckBox("early stop?")
                # widget format
                self.EarStp_widget.setCheckable(True)
                self.EarStp_widget.setChecked(v)
                # connect
                self.EarStp_widget.stateChanged.connect(
                    lambda new_value: self._set_value("early_stop", False if new_value == 0 else True))
                # label
                widget_layout.addWidget(self.EarStp_widget)
                algorithm_setup_layouts.append(widget_layout)

            if k == "std":
                widget_layout = QHBoxLayout()
                # widget type
                self.STD_widget = QDoubleSpinBox()
                # widget format
                self.STD_widget.setMaximum(1.0)
                self.STD_widget.setValue(v)
                self.STD_widget.setMaximumWidth(60)
                # connect
                self.STD_widget.valueChanged.connect(lambda new_value: self._set_value("std", new_value))

                # label
                label = QLabel(k)
                widget_layout.addWidget(self.STD_widget)
                widget_layout.addWidget(label)
                algorithm_setup_layouts.append(widget_layout)

            if k == "evaluate_cycle":
                widget_layout = QHBoxLayout()
                # widget type
                self.EC_widget = QSpinBox()
                # widget format
                self.EC_widget.setMaximum(200)
                self.EC_widget.setValue(v)
                self.EC_widget.setMaximumWidth(100)
                # connect
                self.EC_widget.valueChanged.connect(lambda new_value: self._set_value("evaluate_cycle", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.EC_widget)
                widget_layout.addWidget(label)
                algorithm_setup_layouts.append(widget_layout)
            if k == "learning_rate":
                widget_layout = QHBoxLayout()
                # widget type
                self.LR_widget = QDoubleSpinBox()
                # widget format
                self.LR_widget.setDecimals(3)
                self.LR_widget.setValue(v)
                self.LR_widget.setMaximumWidth(350)
                self.LR_widget.setMinimumWidth(100)
                # connect
                self.LR_widget.valueChanged.connect(lambda new_value: self._set_value("learning_rate", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.LR_widget)
                widget_layout.addWidget(label)
                algorithm_setup_layouts.append(widget_layout)

            if k == "lr_decay":
                widget_layout = QHBoxLayout()
                # widget type
                self.LD_widget = QLineEdit()
                # widget format
                self.LD_widget.setInputMask('0e-00')
                self.LD_widget.setText(format(v, '.0e'))
                self.LD_widget.setMaximumWidth(60)
                # connect
                self.LD_widget.textChanged.connect(lambda new_value: self._set_value("lr_decay", float(new_value)))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.LD_widget)
                widget_layout.addWidget(label)
                algorithm_setup_layouts.append(widget_layout)
            if k == "beta":
                widget_layout = QHBoxLayout()
                # widget type
                self.BETA_widget = QDoubleSpinBox()
                # widget format
                self.BETA_widget.setDecimals(2)
                self.BETA_widget.setValue(v)
                self.BETA_widget.setMaximumWidth(350)
                self.BETA_widget.setMinimumWidth(100)
                # connect
                self.BETA_widget.valueChanged.connect(lambda new_value: self._set_value("beta", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.BETA_widget)
                widget_layout.addWidget(label)
                algorithm_setup_layouts.append(widget_layout)

            if k == "barrier_setup":
                widget_layout = QHBoxLayout()
                # widget type
                self.BS_widget = QCheckBox("Barrier Setup")
                # widget format
                self.BS_widget.setCheckable(True)
                self.BS_widget.setChecked(v)
                # connect
                self.BS_widget.stateChanged.connect(
                    lambda new_value: self._set_value("barrier_setup", False if new_value == 0 else True))
                # label
                widget_layout.addWidget(self.BS_widget)
                algorithm_setup_layouts.append(widget_layout)

            if k == "Iterations":
                widget_layout = QHBoxLayout()
                # widget type
                self.IT_widget = QSpinBox()
                # widget format
                self.IT_widget.setMaximum(10000)
                self.IT_widget.setValue(v)
                self.IT_widget.setMaximumWidth(70)
                # connect
                self.IT_widget.valueChanged.connect(lambda new_value: self._set_value("Iterations", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.IT_widget)
                widget_layout.addWidget(label)
                algorithm_setup_layouts.append(widget_layout)

            if k == "pop_size":
                widget_layout = QHBoxLayout()
                # widget type
                self.PS_widget = QSpinBox()
                # widget format
                self.PS_widget.setMaximum(400)
                self.PS_widget.setValue(v)
                self.PS_widget.setMaximumWidth(70)
                # connect
                self.PS_widget.valueChanged.connect(lambda new_value: self._set_value("pop_size", new_value))
                # label
                label = QLabel(k)
                widget_layout.addWidget(self.PS_widget)
                widget_layout.addWidget(label)
                algorithm_setup_layouts.append(widget_layout)

        # contain second column widgets
        second_column_layout = QVBoxLayout()

        for sub_layout in algorithm_setup_layouts:
            second_column_layout.addLayout(sub_layout)

        # figure areas
        self.mean_fits_svg_widget = QSvgWidget()
        self.mean_fits_svg_widget.setFixedSize(QSize(800, 300))

        self.mean_fits_svg_widget.load(os.path.join(ROOT_DIR, 'src/mean_fits_blank.svg'))

        # self.replay_pixmap = QPixmap()
        # self.replay_label = QLabel("Replay")
        # self.replay_label.setPixmap(self.replay_pixmap)
        # self.replay_label.setScaledContents(True)
        # self.replay_label.setFixedSize(QSize(800, 800))
        self.replay_svg_widget = QSvgWidget()
        self.replay_svg_widget.setFixedSize(QSize(800, 800))

        figure_layout = QVBoxLayout()
        figure_layout.addWidget(self.replay_svg_widget)
        figure_layout.addWidget(self.mean_fits_svg_widget)

        # buttons
        button_layout = QVBoxLayout()

        self.run_button = QPushButton("Run")
        self.run_button.setFixedWidth(100)
        self.run_button.clicked.connect(self._run_button_clicked)
        button_layout.addWidget(self.run_button)

        # progress bar
        self.progress_widget = QProgressBar()
        self.progress_widget.setMaximum(self.args_dict['Iterations'] - 1)
        self.progress_widget.setMinimum(0)
        self.progress_widget.setFixedWidth(300)
        self.progress_label = QLabel("current iterations: 0")
        self.pop_size_label = QLabel("current population size: xx")

        button_layout.addWidget(self.progress_widget)
        button_layout.addWidget(self.progress_label)
        button_layout.addWidget(self.pop_size_label)

        self.setParameter_buttons = QPushButton("Set Configuration")
        self.setParameter_buttons.setMaximumWidth(300)
        self.setParameter_buttons.clicked.connect(self._set_parameter)
        button_layout.addWidget(self.setParameter_buttons)
        # overall layout
        uppermost_layout = QHBoxLayout()
        # include columns of layouts
        uppermost_layout.addLayout(first_column_layout)
        uppermost_layout.addLayout(second_column_layout)
        uppermost_layout.addLayout(figure_layout)
        uppermost_layout.addLayout(button_layout)

        # central widget
        widget = QWidget()
        widget.setLayout(uppermost_layout)

        self.setCentralWidget(widget)

    def _run_button_clicked(self, checked):
        print("program start")
        self.run_button.setDisabled(True)
        self.progress_widget.setMaximum(self.args_dict['Iterations'])

        self.Thread = MainThread(self)
        self.Thread.start()

    def _set_parameter(self):
        set_json_from_dict(self.args_dict)
        # display free-guidance simulation replay
        simple_simulate()
        self.replay_svg_widget.load(os.path.join(ROOT_DIR, self.args.replay_dir, 'Snapshot_test.svg'))

        # pop notification message
        self._pop_message(" Configuration Set Successfully")

    def _set_value(self, k, v):
        print(k, v)
        self.args_dict[k] = v
        if k == "lr_adaptation":
            self.LS_widget.setDisabled(not v)
        print("successfully set value")

    def renew_images(self, cur_iter):
        self.mean_fits_svg_widget.load(os.path.join(ROOT_DIR, self.args.result_dir, 'mean_fits.svg'))

        self.replay_svg_widget.load(os.path.join(ROOT_DIR, self.args.replay_dir, f'Snapshot_{cur_iter}.svg'))

    def renew_description(self, it, size):
        self.progress_widget.setValue(it)
        self.progress_label.setText(f"current iterations: {it}")
        self.pop_size_label.setText(f"current population size: {size}")

    def onChooseCaseClick(self, case):
        """
        Copy desired .json file to common dir
        :param case:
        :return:
        """
        try:
            shutil.copy(src=os.path.join(ROOT_DIR, "Common", "PredefinedConfigs", f"{case}.json"),
                        dst=os.path.join(ROOT_DIR, "Common", "configuration.json"))
        except:
            raise FileNotFoundError
        self._pop_message(f"Configuration for {case} has been loaded\n app will be restarted to apply the setup.")
        self.restart()

    def onSaveConfigClick(self, case):
        try:
            shutil.copy(src=os.path.join(ROOT_DIR, "Common", "configuration.json"),
                        dst=os.path.join(ROOT_DIR, "Common", "PredefinedConfigs", f"{case}.json"))
        except:
            raise FileNotFoundError
        self._pop_message(
            f"Current Configuration have been saved for {case}\n Make sure you have called set_config button!")

    def _pop_message(self, str):
        message_box = QMessageBox(self)
        message_box.setWindowTitle("Attention")
        message_box.setText(str)
        message_box.exec()

    def restart(self):
        QCoreApplication.exit(MainWindow.EXIT_CODE_REBOOT)
        # QtGui.qApp.exit(MainWindow.EXIT_CODE_REBOOT)


class MainThread(QThread):
    def __init__(self, window):
        super(MainThread, self).__init__()
        self.window = window

    def run(self):
        EvolutionProcess = Natural_Evoltuion_Strategy(emitSignal=True)
        EvolutionProcess.update_image_signal.connect(self.window.renew_images)
        EvolutionProcess.update_description_signal.connect(self.window.renew_description)
        EvolutionProcess.RunEvolution()


if __name__ == '__main__':
    currentExitCode = MainWindow.EXIT_CODE_REBOOT
    while currentExitCode == MainWindow.EXIT_CODE_REBOOT:
        args = get_args_from_json()
        app = QApplication(sys.argv)
        window = MainWindow(args)

        currentExitCode = app.exec_()
        print(currentExitCode)
        app = None  # delete app after quit
