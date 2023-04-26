#A. WIPF - PATH INTEGRALS

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numba import njit

N = 100
w_0 = 1
a = np.linspace(0.05, 0.8, 100)
xi = 1
xf = 1
m = 1

#define matrix of coefficients of unimproved action eq of motion
#(x_j+1 + x_j-1) - (2 + a^2*w_0^2)*x_j = 0

@njit
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

@njit
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

@njit
def coeff_matrix_imp_transformed(a):
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
        A[j][j] = -(5/2 - (a*w_0)**2*(2+(a*w_0)**2/6))
        if jp < N:
            A[jp][j] = 4/3
        if jp2 < N:
            A[jp2][j] = -1/12
    return A

def sincos(t, S, w, C):
    return S*np.sin(w*t)+C*np.cos(w*t)
    
@njit
def values(a):
        B=np.empty(N)
        for i in range(N):
            if i == 0:
                B[i] = -a*xi
            elif i == N-1:
                B[i] = -a*xf
            else:
                B[i] = 0
        return B

ws2 = np.empty(len(a))
Ss = np.empty(len(a))
Cs = np.empty(len(a))

ws2_imp = np.empty(len(a))
Ss_imp = np.empty(len(a))
Cs_imp = np.empty(len(a))

Es = np.empty(len(a))
Es_imp = np.empty(len(a))
for i in range(len(a)):

    A = coeff_matrix(a[i])
    A_imp = coeff_matrix_imp(a[i])

    B = values(a[i])
    X = np.linalg.solve(A, B)
    X_imp = np.linalg.solve(A_imp, B)

    t = np.arange(0, N, 1)*a[i]
    plt.scatter(t, X)

    param, cov = optimize.curve_fit(sincos, t, X, p0=(1, 1, 1))
    param_imp, cov_imp = optimize.curve_fit(sincos, t, X_imp, p0=(1, 1, 1))

    fit = [sincos(x, param[0], param[1], param[2]) for x in t]
    plt.plot(t, fit, label=f'a={a[i]}')

    #energy of HO
    E = 0.5*m*w_0**2*(param[0]**2+param[2]**2)
    E_imp = 0.5*m*w_0**2*(param_imp[0]**2+param_imp[2]**2)

    Es[i] = E
    Ss[i] = param[0]
    ws2[i] = param[1]**2
    Cs[i] = param[2]

    Es_imp[i] = E_imp
    Ss_imp[i] = param_imp[0]
    ws2_imp[i] = param_imp[1]**2
    Cs_imp[i] = param_imp[2]

    #print(i)
plt.legend()
plt.show()

plt.plot(a, Ss, label='S sin coefficient')
plt.plot(a, Cs, label='C cos coefficient')
plt.plot(a, Ss_imp, label='S sin coefficient improved')
plt.plot(a, Cs_imp, label='C cos coefficient improved')
plt.plot(a, Es, label='energy')
plt.plot(a, Es_imp, label='energy improved')
plt.legend()
plt.show()

#expected behaviour for unimproved action
exp = [w_0**2*(1+((q*w_0)**2)/12) for q in a]
exp_imp = [w_0**2*(1+((q*w_0)**4)/90) for q in a]


plt.plot(a, ws2, label='frequency')
plt.plot(a, exp, label='expected')
plt.plot(a, ws2_imp, label='frequency improved')
plt.plot(a, exp_imp, label='expected improved')
plt.legend()
plt.show()