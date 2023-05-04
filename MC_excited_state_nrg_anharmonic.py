#ADAPTED FROM LEPAGE'S PAPER
#https://arxiv.org/abs/hep-lat/0506036

import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit

#source and sink, choose x or x3
source = 'x'
#choose improved action discretization y or n or noghost
imp = 'y'

m = 1
w = 1
c = 2

eps = 1.4
a = [0.25, 0.5, 0.75]                                                 #lattice spacing
N = 100                                                  #no of lattice sites
N_cor = 100                                              #no of updates before storing a configuration to make samples statistically independent
N_cf = 100000                                             #number of generated random paths (configurations)

@njit
def update(x, a):
    for j in range(N):
        old_x = x[j]                                    # save original value
        old_Sj = S(j, x, a)
        x[j] = x[j] + np.random.uniform(-eps, eps)      # update x[j]
        dS = S(j, x, a) - old_Sj                           # change in action
        if dS > 0 and np.exp(-dS) < np.random.uniform(0,1):
            x[j] = old_x                                # restore old value

@njit
def S(j, x, a):                                            # harm. osc. S
    jp = (j+1)%N                                        # next site
    jm = (j-1)%N                                        # previous site
    jm2 = (j-2)%N
    jp2 = (j+2)%N
    if imp == 'n':
        return 0.5*a*m*(w**2)*(x[j]**2)*(1+c*m*w*x[j]**2) + m*x[j]*(x[j]-x[jp]-x[jm])/a
    elif imp == 'y':
        return 0.5*a*m*(w**2)*(x[j]**2)*(1+c*m*w*x[j]**2) - (m/(2*a))*x[j]*(-(x[jm2]+x[jp2])/6+(x[jm]+x[jp])*(8/3)-x[j]*(5/2))
    elif imp == 'noghost':
        return 0.5*a*m*(w**2)*(1 + c*m*w**2*x[j]**2)*x[j]**2 + (1/24)*a**3*m*w**4*(x[j] + 2*c*m*w*x[j]**3)**2 - a**2*0.25*c*m*w**3*x[j]**2 + 0.5*a**4*(0.25*c*m*w**3*x[j]**2)**2 + m*x[j]*(x[j]-x[jp]-x[jm])/a

@njit
def compute_G(x, n):                                    #returns the mean correlation of sites n timesteps (n*a) apart
    g = 0
    for j in range(N):
        if source == 'x3':
            g += (x[j]**3)*(x[(j+n)%N]**3)
        elif source == 'x':
            g += x[j]*x[(j+n)%N]
    return g/N

@njit
def MCaverage(x, G, a):
    for j in range(N):                                  # initialize x
        x[j] = 0
    for j in range(0, 10*N_cor):                         # thermalize x   
        update(x, a)
    for alpha in range(N_cf):                           # loop on random paths
        for j in range(N_cor):
            update(x, a)
        for n in range(N):
            G[alpha][n] = compute_G(x, n)

@njit
def deltaE(G_avgd_over_paths, a):                                          # Delta E(t)
    adE = np.log(np.abs(G_avgd_over_paths[:-1] / G_avgd_over_paths[1:]))
    return adE/a

@njit
def avg(G):
    sum = np.sum(G, axis=0)
    avg_G = sum/len(G)
    return avg_G

@njit
def sdev(G):                                            # std dev of G
    return (np.abs(avg(G**2)-avg(G)**2))**0.5

#generates a boostrapped copy of G[alpha] where alpha indices the different paths
@njit
def bootstrap(G):
    N_bs = len(G)
    G_bootstrap = np.empty((N_bs, N))                                    # new ensemble
    for i in range(N_bs):
        alpha = int(np.random.uniform(0, N_bs))         # choose random config
        G_bootstrap[i] = G[alpha]                       # keep G[alpha]
    return G_bootstrap

def bin(G, binsize):
    G_binned = np.empty((int(len(G)/binsize), N))       # binned ensemble
    k=0                                              
    for i in range(0, len(G), binsize):                 # loop on bins                                        
        G_avg = 0
        for j in range(binsize):                        # loop on bin elements
            G_avg += G[i+j]
        G_binned[k] = G_avg/binsize                      # keep bin avg
        k+=1            
    return G_binned

@njit
def bootstrap_deltaE(G, a, nbstrap=100):                   # Delta E + errors
    bsE = np.empty((nbstrap, N-1))
    for i in range(nbstrap):                            # bs copies of deltaE
        g = bootstrap(G)
        bsE[i] = deltaE(avg(g), a)
    sdevE = sdev(bsE)                                   # spread of deltaE’s
    avgE = avg(bsE)
    return avgE, sdevE

def main():

    time_start = time.perf_counter()

    for l in range(len(a)):

        #plot using data from bootstrap
        t = [a[l]*q for q in range(N-1)]

        x = np.zeros(N, dtype=float)
        G = np.zeros((N_cf, N), dtype=float)

        MCaverage(x, G, a[l])      #after running MCaverage G is now ready to be bootsrapped for statistical analysis

        #Bin data to check for statistical correlation in the aquisitions
        #std of G
        stand_dev = sdev(G)
        
        #generate binned copy of G with binned_alpha = alpha/binsize
        binsize = 40
        binned_G = bin(G, binsize)

        #std of binned G
        stand_dev_binned = sdev(binned_G)

        #confront the std of the binned and unbinned data, until binned data error grow with binsize there are correlations between measurements
        print(stand_dev) 
        print(stand_dev_binned)

        #run and extraxt boostrapped statistics
        avgE, sdevE = bootstrap_deltaE(binned_G, a[l], nbstrap=100000)

        plt.title(f'$\epsilon$={eps}, N={N}, N_cor={N_cor}, N_cf={N_cf}, c={c}')
        plt.errorbar(t, avgE, yerr=sdevE, fmt="o", label=f'a = {a[l]}')
        plt.legend()
        plt.axis([-0.1, 3.2, 1, 3])

        print(avgE - 1.933*np.ones(len(t)))

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

    plt.plot(t, 1.933*np.ones(len(t)), label='exact')
    plt.show()

if __name__ == '__main__':
    main()