import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

a = np.linspace(0.001, 0.05, 100)

ghost = np.empty(len(a))
for i in range(len(a)):

    def f(w):
        return np.exp(-2*a[i]*w) -16*np.exp(-a[i]*w) + 30 -16*np.exp(a[i]*w) +np.exp(2*a[i]*w) + 12*(a[i]*w)**2

    sol = sp.optimize.fsolve(f, 1/a[i])

    ghost[i] = sol


def expectation(a, C):
    return C/a

popt, pcov = sp.optimize.curve_fit(expectation, a, ghost)
fit = [expectation(x, popt) for x in a]

print(popt)

plt.title('ghost mode')
plt.plot(a, fit, label='fit')
plt.scatter(a, ghost, label='numerical solution')
plt.xlabel('lattice spacing')
plt.legend()
plt.ylabel('$\omega$')
plt.show()
