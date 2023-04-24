import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

N = 100
w_0 = 1
ghost = 2.6**2
a = np.linspace(0.01, 0.5, 100)
xi = 1
xf = 1

#define matrix of coefficients of unimproved action eq of motion
#(x_j+1 + x_j-1) - (2 + a^2*w_0^2)*x_j = 0

def coeff_matrix(a):
    A = np.zeros((N, N))
    for j in range(N):
        jm = (j-1)
        jp = (j+1)
        if jm >= 0:
            A[jm][j] = 1
        A[j][j] = -(2 - a**2*w_0**2)
        if jp < N:
            A[jp][j] = 1
    return A

#define matrix of coefficients of improved action eq of motion
#-(1/12)(x_j-2 + x_j+2) + (4/3)(x_j-1 x_j+1) - (5/2 + w_0^2*a^2)*x_j = 0

def coeff_matrix_imp(a):
    A = np.zeros((N, N))
    for j in range(N):
        jm = (j-1)
        jp = (j+1)
        jm2 = (j-2)
        jp2 = (j+2)
        if jm2 >= 0:
            A[jm2][j] = -1/12
        if jm >= 0:
            A[jm][j] = 4/3
        A[j][j] = -(5/2 - a**2*w_0**2)
        if jp < N:
            A[jp][j] = 4/3
        if jp2 < N:
            A[jp2][j] = -1/12
    return A

def sin(t, A, w, phi):
    return A*np.sin(w*t+phi)

def values():
        B=np.empty(N)
        for i in range(N):
            if i == 0:
                B[i] = xi
            elif i == N-1:
                B[i] = xf
            else:
                B[i] = 0
        return B

ws = np.empty(len(a))
for i in range(len(a)):

    A = coeff_matrix_imp(a[i])

    B = values()
    X = np.linalg.solve(A, B)

    t = np.arange(0, N, 1)*a[i]
    plt.scatter(t, X)
    w, cov = optimize.curve_fit(sin, t, X)

    fit = [sin(x, w[0], w[1], w[2]) for x in t]
    plt.plot(t, fit)

    ws[i] = w[1]**2

plt.show()

print(min(ws))
plt.plot(a, ws)
plt.show()