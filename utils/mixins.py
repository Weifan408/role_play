from abc import ABC
import numpy as np

from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface


class TwoPlayersTwoActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in a two player discrete game.
    Logs the frequency of each state.
    """

    def _init_info(self):
        self.cc_count = []
        self.dd_count = []
        self.cd_count = []
        self.dc_count = []

    def _reset_info(self):
        self.cc_count.clear()
        self.dd_count.clear()
        self.cd_count.clear()
        self.dc_count.clear()

    def _get_episode_info(self):
        return [
            {
                "CC": np.mean(self.cc_count).item(),
                "DD": np.mean(self.dd_count).item(),
                "CD": np.mean(self.cd_count).item(),
                "DC": np.mean(self.dc_count).item(),
            },
            {
                "CC": np.mean(self.cc_count).item(),
                "DD": np.mean(self.dd_count).item(),
                "CD": np.mean(self.dc_count).item(),
                "DC": np.mean(self.cd_count).item(),
            },
        ]

    def _accumulate_info(self, ac0, ac1):
        self.cc_count.append(ac0 == 0 and ac1 == 0)
        self.cd_count.append(ac0 == 0 and ac1 == 1)
        self.dc_count.append(ac0 == 1 and ac1 == 0)
        self.dd_count.append(ac0 == 1 and ac1 == 1)