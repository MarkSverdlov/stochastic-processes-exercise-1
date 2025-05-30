import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator
# Here, we interested in simulating the continuous HMC with infinitsimal generator: \begin{pmatrix} -1 & 1 & 0 // 1 & -2 & 1 // 0 & 3 & -3 \end{pmatrix}[&BE4JAAA=].
# What we need to do is the following. First, we calculate for each stage the next stage (int) and the time we stay at the current stage. After we sample these numbers, we know the exact simulation for the next few seconds.
# This allows us to generate a sequence of stages and a sequence of stages and times. Then we extract the actual place at time t by getting the first n such that t_1 + ... + t_n > T and deduicng the step is X_n.


LENGTH_OF_SIMULATION = 5
NUMBER_OF_SIMULATIONS = 500


rng = np.random.default_rng(18**2)


def get_T_and_X_from_1():
    X = 2
    T = rng.exponential(1)
    return T, X

def get_T_and_X_from_2():
    X = rng.choice([1, 3])
    T = rng.exponential(1 / 2)
    return T, X

def get_T_and_X_from_3():
    X = 2
    T = rng.exponential(1 / 3)
    return T, X

def get_T_and_X(current):
    if current == 1:
        return get_T_and_X_from_1()
    elif current == 2:
        return get_T_and_X_from_2()
    elif current == 3:
        return get_T_and_X_from_3()


class Simulation:
    def __init__(self):
        self.time = 0
        self.T = np.array([])
        self.current = 1
        self.X = np.array([self.current])

    def simulate_next(self):
        T, X = get_T_and_X(self.current)
        self.time = self.time + T
        self.current = X
        self.X = np.append(self.X, [X])
        self.T = np.append(self.T, [T])

    def grow_to(self, T):
        while self.time < T:
            self.simulate_next()
        self.T[-1] = T - self.T[:-1].sum()
        self.time = T
        self.X = self.X[:-1]  # remove last state that wasn't reached in time
        return self

    def get_state(self, t):
        if t >= self.time:
            return self.X[-1]
        id = 0
        sum = self.T[0]
        while sum < t:
            sum = sum + self.T[id + 1]
            id = id + 1
        return self.X[id]


    def plot_trace(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        T = self.T.cumsum()
        T = np.concatenate([np.zeros(1), T])
        ax.step(T[:-1], self.X, where='post')  # The ending time does not show in the graph
        ax.hlines(self.X[-1], T[-2], T[-1])
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 3]))
        ax.set_ylabel("State")
        return ax


def calc_sim_state(grown_simulation, t_axis, name=None):
    return pd.Series(np.array([grown_simulation.get_state(t) for t in t_axis]), index=t_axis, name=name)


sims = [Simulation().grow_to(LENGTH_OF_SIMULATION) for _ in range(NUMBER_OF_SIMULATIONS)]
X = np.linspace(0, LENGTH_OF_SIMULATION, 200)
samples = pd.concat([calc_sim_state(sim, X, name=f'simulation {i}') for i, sim in enumerate(sims)], axis=1)
samples.to_csv('samples.csv')


# We calculate estimate to the marginal distribuition in every step:
dist_estimates = pd.get_dummies(samples.stack()).groupby(level=0).mean()
dist_estimates.to_csv('distribuitions_estimates.csv')


# Finally, we calculate the L2 difference between the empirical distrbution at time t to the stationary distrbution.

theoretical = pd.DataFrame(index=X, columns=[1, 2, 3])
theoretical[1] = 3 / 7
theoretical[2] = 3 / 7
theoretical[3] = 1 / 7

diff = np.sqrt(((dist_estimates - theoretical)**2).sum(axis=1))
diff.to_csv('convergence.csv')
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 1, figsize=(6, 3.375), sharex=True)
sims[0].plot_trace(ax=axes[0])
axes[0].set_title('Trace Plot of the First Simulation', fontsize=12)
axes[0].text(0.01, 0.8, '(A)', weight='bold', transform=axes[0].transAxes, fontsize=8)
axes[1].plot(diff, label='Estimated Difference')
theoretical_diff = pd.Series(np.exp((-3+np.sqrt(2))*X), index=X)
axes[1].plot(theoretical_diff, ls='--', label='Theoretical Difference')
axes[1].set_title("L2 Difference Between the Estimated Marginal Distribution\nand the Stationary Distribution of the Markov Chain", fontsize=12)
axes[1].set_ylabel('L2 Difference')
axes[1].set_xlabel('Time')
axes[1].text(0.01, 0.8, '(B)', weight='bold', transform=axes[1].transAxes, fontsize=8)
axes[1].legend()
fig.tight_layout()
fig.savefig('Convergence of the Estimated Marginal Distribution.svg')

