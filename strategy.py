"""
Strategy parent class for the k-bandits game.
Child classes are E_Greedy and UCB.

Author: Valerie Sawirja
"""
import numpy as np

class Strategy(object):
    """
    Strategy superclass
    - knows the received rewards of every arm
    - tracks total regret for every timestep
    """
    def __init__(self, timesteps:int, k: int, name):
        self.optim_c = 2000
        self.rewards: dict = self.init_rewards(k)
        self.regrets: list = np.zeros(timesteps)
        self.name = name
        self.accuracy: float

    def init_rewards(self, k):
        """
        Optimistic initialization
        :return: dict with key arms and high values
        """
        optimistic_values = {arm: np.array([self.optim_c]) for arm in range(k)}
        return optimistic_values

    def update_rewards(self, arm: int, reward: float):
        """
        add another value to the self.rewards.
        if the existing value is Optimistic Init, replace; otherwise, append
        """
        if self.rewards[arm][0] == self.optim_c:
            self.rewards[arm] = np.array(reward)
        else:
            self.rewards[arm] = np.append(self.rewards[arm], reward)

    def set_acc(self, acc: float):
        self.accuracy = acc

    def count_choices(self) -> dict:
        """
        Returns the number of times ever arm has been pulled
        :return:
        """
        times_chosen = dict()

        # exclude the optimistic value when counting choices
        for arm, values in self.rewards.items():
            if self.optim_c not in values:
                times_chosen[arm] = len(values)
            else:
                times_chosen[arm] = 0

        return times_chosen

    def update_regrets(self, timepoint, regret):
        """
        Adds new timepoint's expected total regret
        """
        self.regrets[timepoint] = regret