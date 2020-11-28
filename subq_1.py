"""
overall flow:

initialize the bandit parameters (arm means, q start values)
init storage dict for every strategy

play loop:
    for strategy in [eVALUE_greedy, eVALUE2_greedy, eVALUE3_greedy, UCB, UCB_c, UCB_c2]:
        for pull in T:
            choose arm based on (highest q(a) / uncertainty with highest upper bound)
            store arm choice (for % correct decisions)
            draw reward (normal distrib around arm mean)
            store reward in strategy[q(a)]
            calculate total regret
            store total regret in strategy

    compute the Lai_Robbins lower bound
    draw curves of all strategies and the Lai_Robbins bound

    compute % correct decisions for every strategy
    show table

Author: Valerie Sawirja
"""
import numpy as np
import random

class Game(object):
    def __init__(self, k):
        self.k = k
        self.arm_means: list = self.init_arm_means(k)

    def init_arm_means(self, k):
        means = np.array([random.random() for i in range(k)])
        return means

    def draw_reward(self, arm):
        """
        Draws a reward from a normal distribution with unit std around mean of arm

        :param arm_mean:
        :return: reward (float)
        """
        arm_mean = self.arm_means[arm]
        return np.random.normal(loc=arm_mean, scale=1.0, size=1)

    def play(self, strategies: list, T: int):
        """
        play games for every strategy
        :return: the strategy dicts
        """
        for strategy in strategies:
            for pull in range(T):
                # choose arm based on (highest q(a) / uncertainty with highest upper bound)
                arm = strategy.choose_arm()

                # draw reward & store reward in strategy class
                reward = self.draw_reward(arm)
                strategy.update_rewards(arm, reward)

                # todo calculate total regret
                # store total regret in strategy

    def calc_lai_robbins(self):
        pass

    def calc_accuracy(self):
        pass

    def experiment(self, T=1000):
        """
        Perform the actual k-bandit experiment

        :return:
        """
        # initialize class instances for the strategies
        e1_greedy = E_Greedy(epsilon=0.1, k=self.k)
        e3_greedy = E_Greedy(epsilon=0.3, k=self.k)
        e8_greedy = E_Greedy(epsilon=0.8, k=self.k)
        #todo UCD

        # init storage dict for every strategy(put optimistic q in it) --> can also be strategy class
        # strategies = [eVALUE_greedy, eVALUE2_greedy, eVALUE3_greedy, UCB, UCB_c, UCB_c2]
        strategies = [e1_greedy, e3_greedy, e8_greedy]

        # do play loop
        strategies = self.play(strategies, T)

        # compute the Lai_Robbins lower bound
        _ = self.calc_lai_robbins()
        # draw curves of all strategies and the Lai_Robbins bound

        # compute % correct decisions for every strategy
        _ = self.calc_accuracy()
        # show table

class Strategy(object):
    def __init__(self, k):
        self.rewards: dict = self.init_rewards(k)

    def init_rewards(self, k):
        """
        Optimistic initialization
        :return: dict with key arms and high value
        """
        optimistic_values = {arm: np.array([2000]) for arm in range(k)}
        return optimistic_values

    def update_rewards(self, arm: int, reward: float):
        """
        add another value to the self.rewards.
        if the existing value is Optimistic Init, replace; otherwise, append
        """
        if self.rewards[arm][0] > 1000:
            self.rewards[arm] = np.array(reward)
        else:
            self.rewards[arm] = np.append(self.rewards[arm], reward)


class E_Greedy(Strategy):
    def __init__(self, epsilon, k):
        Strategy.__init__(self, k)
        # self.rewards: dict = self.init_rewards(k)
        self.epsilon: float = epsilon

    # def init_rewards(self, k):
    #     """
    #     Optimistic initialization
    #
    #     :return: dict with key arms and high value
    #     """
    #     optimistic_values = {arm: np.array([2000]) for arm in range(k)}
    #     return optimistic_values
    #
    # def update_rewards(self, arm: int, reward: float):
    #     """
    #     add another value to the self.rewards.
    #     if the existing value is Optimistic Init, replace; otherwise, append
    #     """
    #     if self.rewards[arm][0] > 1000:
    #         self.rewards[arm] = np.array([reward])
    #     else:
    #         self.rewards[arm] = np.append(self.rewards[arm], reward)

    def calc_best_arm(self):
        """
        Returns the arm that has the highest average of drawn values
        could be improved using some dictionary comprehension

        :return:
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

    def choose_arm(self):
        """
        Chooses arm with a*=1-epsilon and other_a=epsilon
        :return:
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

if __name__ == "__main__":
    game = Game(k=5)
    game.experiment(T=10)