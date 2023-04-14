import numpy as np
import matplotlib.pyplot as plt

t_min = 0
t_max = 4
T = t_max - t_min

x = np.linspace(0, 2, 10)

exact = [np.exp(-0.5*(T))*(np.exp(-0.5*i**2)/((np.pi)**0.25))**2 for i in x]

#lattice spacing 0.4 results, done with 1e7 number of evaluations (neval) and 10 iterations (nitn)
#computation time = 16106.1 seconds
a_04 = [0.08082022, 0.07688142, 0.06615662, 0.05149643, 0.03630287, 0.0231589, 0.01335318, 0.00696922, 0.00329297, 0.00140723]
#lattice spacing 0.5 results, done with 1e6 number of evaluations (neval) and 10 iterations (nitn)
#computation time = 1368.5 seconds
a_05 = [0.08284511, 0.07875156, 0.06765028, 0.05254049, 0.03685722, 0.02336877, 0.01338682, 0.00693364, 0.00324445, 0.00137188]
#lattice spacing 0.25 results, done with 1e6 number of evaluations (neval) and 20 iterations (nitn) with learing speed 0.2 
#and same settings of integrator training with learning speed 0.4 (default is 0.5)
#computation time = 9610.6 seconds
a_025 = [0.0752747,  0.07111113, 0.05934637, 0.04802138, 0.03445684, 0.02274903, 0.0126437,  0.00693276, 0.00329618, 0.00143581]

plt.plot(x, exact, label='exact', linestyle='dashed')
plt.plot(x, a_04, label='a=0.4')
plt.plot(x, a_05, label='a=0.5')
plt.plot(x, a_025, label='a=0.25')
plt.legend()

plt.show()