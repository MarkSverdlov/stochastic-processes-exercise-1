import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# Here, we interested in simulating the continuous HMC with infinitsimal generator: \begin{pmatrix} -1 & 1 & 0 // 1 & -2 & 1 // 0 & 3 & -3 \end{pmatrix}[&BE4JAAA=].
# What we need to do is the following. First, we calculate for each stage the next stage (int) and the time we stay at the current stage. After we sample these numbers, we know the exact simulation for the next few seconds.
# This allows us to generate a sequence of stages and a sequence of stages and times. Then we extract the actual place at time t by getting the first n such that t_1 + ... + t_n > T and deduicng the step is X_n.

def get_T_and_X_from_1():
    X = 2
    T = np.random.exponential(1)
    return T, X

def get_T_and_X_from_2():
    X = np.random.choice([1, 3])
    T = np.random.exponential(1 / 2)
    return T, X

def get_T_and_X_from_3():
    X = 2
    T = np.random.exponential(1 / 3)
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

    def get_state(self, t):
        id = 0
        sum = self.T[0]
        while sum <= t:
            sum = sum + self.T[id + 1]
            id = id + 1
        return self.X[id]

sim = Simulation()
sim.grow_to(6000)
X = np.linspace(0, 6000, 200)
y = np.array([sim.get_state(x) for x in X])
plt.plot(X, y)
plt.show()
df = pd.DataFrame(columns=['state', 'time'])
df['state'] = sim.X[:-1]  # The last state doesn't have corresponding time yet
df['time'] = sim.T
(df.groupby('state').sum() / df['time'].sum()).to_csv('results.csv')
