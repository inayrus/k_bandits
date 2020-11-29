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
import matplotlib.pyplot as plt
from e_greedy import E_Greedy
from ucb import UCB
import numpy as np
import random

class Game(object):
    def __init__(self, k, timepoints=1000):
        self.k = k
        self.T = timepoints
        self.arm_means: list = self.init_arm_means(k)
        self.best_arm = np.argmax(self.arm_means)
        self.delta_a: list = self.calc_delta_a()
        self.lai_rob: list

    def init_arm_means(self, k):
        """
        Chooses random values (0, 1) for the k number arms
        """
        means = np.array([random.random() for i in range(k)])
        return means

    def calc_delta_a(self):
        """
        Calculate difference between the mean of the best arm and
        mean of the others.
        :return: list with differences
        """
        differences = []
        for other_arm_mean in self.arm_means:
            diff = self.arm_means[self.best_arm] - other_arm_mean
            differences.append(diff)
        return np.array(differences)

    def draw_reward(self, arm):
        """
        Draws a reward from a normal distribution with unit std around mean of arm

        :param arm_mean:
        :return: reward (float)
        """
        arm_mean = self.arm_means[arm]
        return np.random.normal(loc=arm_mean, scale=1.0, size=1)

    def calc_regret(self, strategy) -> float:
        """
        Computes expected total regret: the sum of all difference between
        optimal arm mean and chosen arm means.
        :returns: the sum of the differences.
        """
        regret = 0

        # get the times every arm has been chosen
        times_chosen = strategy.count_choices()

        # sum the regret for every a
        for arm in times_chosen:
            # multiply (best_a - other_a) with the times other_a is chosen
            regret += self.delta_a[arm] * times_chosen[arm]

        return regret

    def calc_lai_robbins(self):
        """
        Calculates the theoretical lower bound, for comparison with
        the total regret

        exclude the optimal distrib in your calculations
        4.3 that specifies the distance between two normals
        :return:
        """
        A = 0

        # 1. compute Kullback-Leibler divergence
        KL = self.compute_KL()

        # 2. A = sum over a (without a*) of delta_a/KL(fa || f*a)
        for i, diff in enumerate(self.delta_a):
            # exclude best arm
            if diff != 0:
                A += diff / KL[i]

        # 3. get y values for Lai-Robbins
        timepoints = np.array(range(1, self.T+ 1))
        lai_rob = list(map(lambda t: A * np.log(t), timepoints))

        self.lai_rob = np.array(lai_rob)

    def sample_normal(self, x, mu):
        """
        Calculate values of a normal distribution
        """
        y = np.exp(- np.power(x - mu, 2) / 2) / np.sqrt(2 * np.pi)

        return y

    def compute_KL(self):
        """
        compute KL: draw a couple X and compare distribution fa / fa_star
        :return:
        """
        N = 500
        x = np.linspace(-10, 10, N)
        fa_star = self.sample_normal(x, mu=self.arm_means[self.best_arm])

        # init KL (minus 1 bc best arm is excluded
        KL = np.zeros(self.k)

        for i, mean in enumerate(self.arm_means):
            # draw distribution for the other arm
            fa = self.sample_normal(x, mean)

            # divide by best distrib and take log. then average over all elems
            div = np.divide(fa, fa_star)
            KL[i] = np.sum(np.log(div)) / N
        return KL

    def strats_as_dict(self, strategies: list) -> dict:
        """
        Sorts a list with different strategies to a strategy indexed dict
        (could be improved by dynamically calling derived Strategy classes)
        :return:
        """
        all_strats = {"UCB": UCB,
                      "Epsilon Greedy": E_Greedy}
        strat_dict = dict()

        for name, strat in all_strats.items():
            strat_dict[name] = list(filter(lambda x: isinstance(x, strat),
                                                 strategies))

        return strat_dict

    def plot_strats(self, strategies):
        """
        Create seperate plots for the strategies, but include the Lai_Robbins.
        :param strategies:
        :return:
        """
        strat_dict = self.strats_as_dict(strategies)

        for strat_name, strats in strat_dict.items():
            # plot
            plt.figure(figsize=(10, 4))

            for strategy in strats:
                curve = strategy.regrets
                plt.plot(curve, alpha=0.1)
                plt.plot(curve, label=f'{strategy.name}')

            # lai robbins
            plt.plot(self.lai_rob, label=f'Lai-Robbins')

            plt.legend()
            plt.title(f"Total expected regret for the {strat_name} strategy")
            plt.xlabel('Time Points')
            plt.ylabel('Expected Regret')
            plt.yticks(np.arange(0, 350, 50))

            plt.savefig(f'results/strategy_regret_{strat_name}.pdf', bbox_inches="tight")

    def calc_accuracy(self, strategies):
        """

        :param strategies:
        :return:
        """
        # count how many times the best arm was chosen
        for strat in strategies:
            acc = len(strat.rewards[self.best_arm]) / self.T
            strat.set_acc(acc)

            print(f"strategy: {strat.name}, acc: {acc}")

    def play(self, strategies: list):
        """
        play games for every strategy
        :return: the strategy objects
        """
        for strategy in strategies:
            for timepoint in range(self.T):
                # choose arm based on (highest q(a) / uncertainty with highest upper bound)
                arm = strategy.choose_arm(timepoint)

                # draw reward & store reward in strategy class
                reward = self.draw_reward(arm)
                strategy.update_rewards(arm, reward)

                # calculate total regret & store it
                regret = self.calc_regret(strategy)
                strategy.update_regrets(timepoint, regret)
        return strategies

    def experiment(self, strategies):
        """
        Perform the actual k-bandit experiment

        :return:
        """
        # do play loop
        strategies = self.play(strategies)

        # compute the Lai_Robbins lower bound
        self.calc_lai_robbins()

        # draw curves of all strategies and the Lai_Robbins bound
        self.plot_strats(strategies)

        # compute % correct decisions for every strategy & print acc
        self.calc_accuracy(strategies)



if __name__ == "__main__":
    timepoints = 1000
    k = 5
    game = Game(k, timepoints)

    # initialize class instances for the strategies
    ucb_05 = UCB(timepoints, 0.5, k, "c = 0.5")
    ucb_1 = UCB(timepoints, 1.0, k, "c = 1.0")
    ucb_3 = UCB(timepoints, 3.0, k, "c = 3.0")

    e1_greedy = E_Greedy(timepoints, epsilon=0.1, k=k, name="epsilon: 0.1")
    e3_greedy = E_Greedy(timepoints, epsilon=0.3, k=k, name="epsilon: 0.3")
    e8_greedy = E_Greedy(timepoints, epsilon=0.8, k=k, name="epsilon: 0.8")

    strategies = [ucb_05, ucb_1, ucb_3, e1_greedy, e3_greedy, e8_greedy]

    game.experiment(strategies)