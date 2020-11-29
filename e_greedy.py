"""
Representation of the epsilon-greedy strategy.
This strategy picks a random option with probability epsilon,
and otherwise picks the greedy action (expected best, with 1-epsilon).

Author: Valerie Sawirja
"""
from strategy import Strategy
import random

class E_Greedy(Strategy):
    def __init__(self, timesteps, epsilon, k, name):
        Strategy.__init__(self, timesteps, k, name)
        self.epsilon: float = epsilon

    def calc_best_arm(self) -> int:
        """
        Returns the arm that has the highest average of drawn values
        could be improved using some dictionary comprehension

        :return: index of best arm
        """
        best = 0
        best_arm = ""
        for arm, values in self.rewards.items():
            # calc means over all rewards of arms so far
            mean_reward = values.mean()

            # pick arm of highest mean
            if mean_reward >= best:
                best = mean_reward
                best_arm = arm

        return best_arm

    def choose_arm(self) -> int:
        """
        Chooses arm with a*=1-epsilon and other_a=epsilon
        :return: index of chosen arm
        """
        best_arm = self.calc_best_arm()

        # draw random number to decide on exploit or explore action
        prob = random.random()

        # do max if prob is 1-epsilon
        if prob > self.epsilon:
            return best_arm
        # pick a random non-max arm
        else:
            other_arms = [x for x in self.rewards.keys() if x != best_arm]
            return random.choice(other_arms)