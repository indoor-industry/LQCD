import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

a = 0.25

def V(r, sigma, b, c):
    return sigma*r - b/r + c

WL_avg = np.load('data/W_rt Ncf=10, t=[1-5], imp=True, smear=True.npy')
WL_avg_err = np.load('data/W_rt_err Ncf=10, t=[1-5], imp=True, smear=True.npy')

r_max = WL_avg.shape[1]
t_max = WL_avg.shape[0]

rad = np.load('data/radius Ncf=10, t=[1-5], imp=True, smear=True.npy')

potential = np.zeros(r_max-1) 
potential_err_ratio = np.zeros(r_max-1) 
potential_err_log = np.zeros(r_max-1)
for r in rad:
    potential[r-1] = np.log(np.abs(WL_avg[t_max-2, r]/WL_avg[t_max-1, r]))
    potential_err_ratio[r-1] = (((WL_avg_err[t_max-2, r]/WL_avg[t_max-2, r])**2 + (WL_avg_err[t_max-1, r]/WL_avg[t_max-1, r])**2)**(1/2))*potential[r-1]
    potential_err_log[r-1] = potential_err_ratio[r-1]/potential[r-1]

plt.errorbar(rad, potential, potential_err_log, fmt='o', label='simulation')
plt.xlabel('r/a')
plt.ylabel('aV(r)')
plt.axis([0.1, 4.5, -0.5, 2.5])
plt.title('Static quark potential')

popt, pcov = optimize.curve_fit(V, rad, potential, sigma=potential_err_log , p0=(0.5, 0.5, 0.5), bounds=([0, 0.1, 0],[1, 1, 1]))
radius = np.linspace(0.1, 4.5, 100)
fit = [popt[0]*x -popt[1]/x + popt[2] for x in radius]
plt.plot(radius, fit, label='fit')
plt.plot(radius, np.zeros(len(radius)), linestyle='dashed', color='black')
plt.legend()

print('sigma =' + str(popt[0]))
print('b =' + str(popt[1]))
print('c =' + str(popt[2]))

plt.show()