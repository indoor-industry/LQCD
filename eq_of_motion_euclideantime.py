#A. WIPF - PATH INTEGRALS

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numba import njit

N = 100
w_0 = 1
m = 1
a = np.linspace(0.1, 0.5, 100)

T = N/a

#boundary coniditions, if in real time sin(wt) at initial and final times is zero (Dirichelet boundary conditions), in imaginary time 
#x_i=x_{-1}=1 and x_f=x_N=(approxiamtely) 0 (for long total time evolution T), B (the functions values and values_imp) need to be defined accordingly
#xi2 is x_{-2}
#xi is x_{-1}        
#xf is x_N                
#xf2 is x_{N+1}

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
        A[j][j] = -(2 + a**2*w_0**2)
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
        A[j][j] = -(5/2 + a**2*w_0**2)
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
        if jm >= 0:
            A[jm][j] = 1
        A[j][j] = -(2 + (a*w_0)**2*(1+(a*w_0)**2/12))
        if jp < N:
            A[jp][j] = 1
    return A

def exp(t, A, w, phi):
    return A*np.exp(-w*t+phi)
    
@njit
def values(a, xi, xf):
        B=np.empty(N)
        for i in range(N):
            if i == 0:
                B[i] = -xi
            elif i == N-1:
                B[i] = -xf
            else:
                B[i] = 0
        return B

def values_imp(a, xi, xf, xi2, xf2):
        B=np.empty(N)
        for i in range(N):
            if i == 0:
                B[i] = (1/12)*xi2-(4/3)*xi
            elif i == 1:
                B[i] = xi*(1/12)
            elif i == N-2:
                B[i] = xf*(1/12)
            elif i == N-1:
                B[i] = (1/12)*xf2-(4/3)*xf
            else:
                B[i] = 0
        return B

ws2 = np.empty(len(a))
ws2_imp = np.empty(len(a))
ws2_imp_transformed = np.empty(len(a))
for i in range(len(a)):

    A = coeff_matrix(a[i])
    A_imp = coeff_matrix_imp(a[i])
    A_imp_transormed = coeff_matrix_imp_transformed(a[i])

    B = values(a[i], xi= np.exp(0), xf= np.exp(-T[i]))
    B_imp = values_imp(a[i], xi= np.exp(0), xf= np.exp(-T[i]), xi2= np.exp(-(-a[i])), xf2= np.exp(-T[i]+a[i]))
    X = np.linalg.solve(A, B)
    X_imp = np.linalg.solve(A_imp, B_imp)
    X_imp_transformed = np.linalg.solve(A_imp_transormed, B)

    t = np.arange(0, N, 1)*a[i]
    #plt.scatter(t, X, label=f'a={a[i]}')       #show unimproved action solution
    #plt.scatter(t, X_imp, label=f'a={a[i]}')    #show improved action solution
    #plt.scatter(t, X_imp_transformed, label=f'a={a[i]}')    
    #plt.legend()

    param, cov = optimize.curve_fit(exp, t, X, p0=(0.1, 1, 0))
    param_imp, cov_imp = optimize.curve_fit(exp, t, X_imp, p0=(0.1, 1, 0))
    param_imp_transformed, cov_imp_transformed = optimize.curve_fit(exp, t, X_imp_transformed, p0=(0.1, 1, 0))

    fit = [exp(x, param[0], param[1], param[2]) for x in t]            #show unimproved action solution fit
    fit_imp = [exp(x, param_imp[0], param_imp[1], param_imp[2]) for x in t] #show improved action solution fit
    fit_imp_transformed = [exp(x, param_imp_transformed[0], param_imp_transformed[1], param_imp_transformed[2]) for x in t]
    #plt.plot(t, fit, label=f'a={a[i]}')
    #plt.plot(t, fit_imp, label=f'a={a[i]}')
    #plt.plot(t, fit_imp_transformed, label=f'a={a[i]}')

    ws2[i] = param[1]**2
    ws2_imp[i] = param_imp[1]**2
    ws2_imp_transformed[i] = param_imp_transformed[1]**2

#plt.show()

#expected behaviour for unimproved action
expo = [w_0**2*(1-((q*w_0)**2)/12) for q in a]
expo_imp = [w_0**2*(1+((q*w_0)**4)/90) for q in a]

plt.title(f'N={N}, $\omega_0$={w_0}, m={m}')
plt.scatter(a, ws2, s=10, label='frequency')
plt.plot(a, expo, label='expected')
plt.scatter(a, ws2_imp, s=10, label='frequency improved')
plt.plot(a, expo_imp, label='expected improved')
plt.scatter(a, ws2_imp_transformed, s=10, label='frequency improved no ghost')
plt.xlabel('lattice spacing')
plt.ylabel('$\omega^2$')
plt.legend()
plt.show()