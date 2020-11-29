"""
Representation of the Upper Confidence Bound (UCB) strategy. The goal of
exploration is to reduce uncertainty of actions.

Implementation of the Hoeffding version.
1. Compute confidence intervals for every action.
2. Pick the one with the highest upper bound.

Author: Valerie Sawirja
"""
from strategy import Strategy
import numpy as np

class UCB(Strategy):
    def __init__(self, timesteps: int, c: float, k: int, name: str):
        Strategy.__init__(self, timesteps, k, name)
        self.c: float = c


    def choose_arm(self, timepoint) -> int:
        """
        Chooses arm by calculating the upper bound around an expected reward
        :return: index of chosen arm
        """
        # count the choices per arm until now and init storage var
        choices = self.count_choices()
        upperbounds = np.zeros(len(choices))

        # because timepoint=0 error
        timepoint = timepoint + 1

        for arm, values in self.rewards.items():
            # calc means over all rewards of arms so far
            expected_reward = values.mean()

            if expected_reward == self.optim_c:
                # division by 0 error, bc arm has not actually been chosen yet
                bound_u = self.c * np.sqrt(np.log(timepoint))
            else:
                # calculate the uncertainty around the mean
                bound_u = self.c * np.sqrt(np.log(timepoint)/choices[arm])

            # append upperbound
            upperbounds[arm] = expected_reward + bound_u

        # get best arm
        best_arm = np.argmax(upperbounds)

        return best_arm