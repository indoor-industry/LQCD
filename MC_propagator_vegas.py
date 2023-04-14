import numpy as np
import matplotlib.pyplot as plt
import vegas
import time

potential = 'H'     #H for harmonic, Q for quartic 

time_start = time.perf_counter()

t_min = 0
t_max = 4               #initial and final time of evolution
a = 0.25                 #lattice spacing
T = t_max - t_min       #total evolution time
x_min = -5              #integration range for possible paths
x_max = 5
t = np.arange(t_min+a, t_max, a)      #timeslices for evaluation of path in path integral
ground = np.linspace(0, 2, 10)      #values for initial and final positions to sweep

def V(x):
    if potential == 'H':
        return 0.5*x**2
    elif potential == 'Q':
        return 0.5*x**4               

def main():

    p = np.empty(len(ground))
    sigma = np.empty(len(ground))
    exact = np.empty(len(ground))

    for i, q in enumerate(ground):      
        x_i = q                 #boundaries
        x_f = x_i

        def I(x):
            S_E = 0 
            for j in range(len(t)):
                if j == 0:
                    S_E += (0.5/a)*(x[j]-x_i)**2 + a*0.5*V(x_i) + a*V(x[j])
                elif j == len(t)-1:
                    S_E += (0.5/a)*((x_f-x[j])**2 + (x[j]-x[j-1])**2) + a*V(x[j]) + 0.5*a*V(x_f)
                else:
                    S_E += (0.5/a)*(x[j]-x[j-1])**2 + a*V(x[j-1])
            return ((2*np.pi*a)**(-(len(t)+1)/2))*np.exp(-S_E)

        #now integrate exp(-S) in the list of variables x trough values x_t
        integ = vegas.Integrator((len(t)) * [[x_min, x_max]])

        #train the integrator, discard results
        integ(I, nitn=20, neval=1e6, alpha=0.4)

        #integrate, keep results
        result = integ(I, nitn=20, neval=1e6, alpha=0.2)
        
        p[i] = result.mean
        sigma[i] = result.sdev

        analytic = np.exp(-0.5*(T))*(np.exp(-0.5*q**2)/((np.pi)**0.25))**2
        exact[i] = analytic

        print(result.summary())
        #print('result = %s    Q = %.2f' % (result, result.Q))
        print(i)

    print(p)
    print(exact)
    
    plt.title(f'lattice spacing={a}, {potential} potential')
    plt.plot(ground, exact, linestyle='dashed')
    plt.errorbar(ground, p, yerr=sigma, fmt="o")

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    plt.show()

if __name__ == '__main__':
    main()