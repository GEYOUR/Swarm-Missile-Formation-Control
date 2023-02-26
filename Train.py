import sys
import numpy as np
import torch
import tqdm
from multiprocessing import Process
from PyQt5.QtCore import QObject
from PyQt5 import QtCore


class Natural_Evoltuion_Strategy(QObject):
    update_image_signal = QtCore.pyqtSignal(int)
    update_description_signal = QtCore.pyqtSignal(int, int)

    def __init__(self, emitSignal):
        super(Natural_Evoltuion_Strategy, self).__init__()
        self.emitSignal = emitSignal

    def RunEvolution(self):
        print("Running...")
        from Evo_Stra.Ecosys import Ecosystem
        from Evo_Stra.Env_Ind import Missile
        from Common.Config import get_args_from_json

        args = get_args_from_json()

        param_shape = {k: v.shape for k, v in Missile().get_params().items()}

        pbar = tqdm.tqdm(range(args.Iterations))
        ecosystem = Ecosystem((param_shape, Missile.from_params, args.std), args)

        # main loop
        for it in pbar:
            raw_fits = ecosystem.update()

            # evaluate
            if (it+1) % args.evaluate_cycle == 0:
                fitness = ecosystem.evaluate(it)
                print(f"Evaluate at iteration {it}, fitness: {fitness}")
                if self.emitSignal:
                    self.update_image_signal.emit(it)

            # pop size adaptation
            if len(ecosystem.evolution_path) > 1:
                if ecosystem.config.population_adaptation:
                    new_pop_size = int(ecosystem.pop_size * (
                                ecosystem.beta + (1 - ecosystem.beta) * ecosystem.evolution_path[-2] / ecosystem.evolution_path[-1]))
                    ecosystem.pop_size = np.clip(new_pop_size if new_pop_size % 2 == 0 else new_pop_size + 1,
                                            ecosystem.pop_size, ecosystem.pop_size_max)  # make it even, make it non-decreasing
            ecosystem.pop_size_history.append(ecosystem.pop_size)
            if self.emitSignal:
                self.update_description_signal.emit(it, ecosystem.pop_size)

if __name__ == '__main__':
    evolution = Natural_Evoltuion_Strategy(emitSignal=False)
    evolution.RunEvolution()

