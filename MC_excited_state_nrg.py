#MOSTLY TAKEN FROM LEPAGE'S PAPER
#https://arxiv.org/abs/hep-lat/0506036

import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit

eps = 1.4
a = 0.5                                                 #lattice spacing
N = 20                                                  #no of lattice sites
N_cor = 20                                              #no of updates before storing a configuration to make samples statistically independent
N_cf = 1000                                               #number of generated random paths (configurations)

max_n = 8                                               #max value of n to show on plot

@njit
def update(x):
    for j in range(N):
        old_x = x[j]                                    # save original value
        old_Sj = S(j, x)
        x[j] = x[j] + np.random.uniform(-eps, eps)      # update x[j]
        dS = S(j, x) - old_Sj                           # change in action
        if dS > 0 and np.exp(-dS) < np.random.uniform(0,1):
            x[j] = old_x                                # restore old value

@njit
def S(j, x):                                            # harm. osc. S
    jp = (j+1)%N                                        # next site
    jm = (j-1)%N                                        # previous site
    return a*x[j]**2/2 + x[j]*(x[j]-x[jp]-x[jm])/a

@njit
def compute_G(x, n):                                    #returns the mean correlation of sites n timesteps (n*a) apart
    g = 0
    for j in range(N):
        g += x[j]*x[(j+n)%N]
    return g/N

@njit
def MCaverage(x, G):
    for j in range(N):                                  # initialize x
        x[j] = 0
    for j in range(0, 10*N_cor):                         # thermalize x   
        update(x)
    for alpha in range(N_cf):                           # loop on random paths
        for j in range(N_cor):
            update(x)
        for n in range(N):
            G[alpha][n] = compute_G(x, n)
    G_avg_over_paths = np.empty(N)        
    for n in range(N):                                  # compute MC averages
        avg_G = 0
        for alpha in range(N_cf):
            avg_G += G[alpha][n]
        avg_G = avg_G/N_cf
        G_avg_over_paths[n] = avg_G
    return G_avg_over_paths                             #returns the mean correlation for each distance (n*a) averaged over N_cf paths (configurations)

@njit
def deltaE(G_avg_over_paths):                                          # Delta E(t)
    adE = np.log(np.abs(G_avg_over_paths[:-1] / G_avg_over_paths[1:]))
    return adE/a

@njit
def avg(G):                                             # MC avg of G
    return np.sum(G)/len(G)

@njit
def avg_over_paths(G):                                             # MC avg of G
    return np.sum(G, axis=0)/len(G)

#@njit
def sdev(G):                                            # std dev of G
    g = np.asarray(G)
    return np.abs(avg_over_paths(g**2)-avg_over_paths(g)**2)**0.5

#generates a boostrapped copy of G[alpha] where alpha indices the different paths
@njit
def bootstrap(G):
    N_cf = len(G)
    G_bootstrap = []                                    # new ensemble
    for i in range(N_cf):
        alpha = int(np.random.uniform(0, N_cf))         # choose random config
        G_bootstrap.append(G[alpha])                    # keep G[alpha]
    return G_bootstrap

#@njit
def bin(G, binsize):
    G_binned = []                                       # binned ensemble
    for i in range(0, len(G), binsize):                 # loop on bins
        G_avg = 0
        for j in range(binsize):                        # loop on bin elements
            G_avg += G[i+j]
        G_binned.append(G_avg/binsize)                  # keep bin avg
    return G_binned

@njit
def bootstrap_deltaE(G, nbstrap=100):                   # Delta E + errors
    avgE = deltaE(G)                                    # avg deltaE
    bsE = []
    for i in range(nbstrap):                            # bs copies of deltaE
        g = bootstrap(G)
        bsE.append(deltaE(g))
    bsE = np.array(bsE)
    sdevE = sdev(bsE)                                   # spread of deltaE’s
    print("t", "Delta E(t)", "error")
    print(26*"-")
    for i in range(len(avgE)/2):
        print(f'{i} {avgE[i]} {sdevE[i]}')

def main():

    time_start = time.perf_counter()

    x = np.zeros(N, dtype=float)
    G = np.zeros((N_cf, N), dtype=float)

    MCaverage(x, G)      #after running MCaverage G is now ready to be binned and bootsrapped for statistical analysis

    G_avg_over_paths = avg_over_paths(G)

    #energy differnece between first excited state and ground state (1 for HO)
    dE = deltaE(G_avg_over_paths)

    #std of G
    stand_dev = sdev(G)
    #print(stand_dev[:max_n])

    #propagation of error assuming independent values of G
    abs_err_deltaE = np.empty(N-1)
    for i in range(N-1):
        abs_err_deltaE[i] = np.abs(((stand_dev[i]/G_avg_over_paths[i])-(stand_dev[i+1]/G_avg_over_paths[i+1]))/a)

    #generate binned copy of G with binned_alpha = alpha/binsize
    binsize = 5
    binned_G = bin(G, binsize)

    #std of binned G
    stand_dev_binned = sdev(binned_G)
    #print(stand_dev_binned[:max_n])

    #plot deltaE
    t = [a*q for q in range(N-1)]
    plt.errorbar(t[:max_n], dE[:max_n], yerr=abs_err_deltaE[:max_n], fmt="o", label='computed')
    plt.plot(t[:max_n], np.ones(len(t))[:max_n], label='exact')
    plt.legend()
    plt.show()


    #average of bootstrapped G
    #print ('avg G (bootstrap)\n', avg(bootstrap(G)))
    
    #energy difference of bootstrapped G
    #print('Delta E (bootstrap)\n', deltaE(bootstrap(G)))

    #bootstrap_deltaE(G)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))

if __name__ == '__main__':
    main()