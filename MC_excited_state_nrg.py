#MOSTLY TAKEN FROM LEPAGE'S PAPER
#https://arxiv.org/abs/hep-lat/0506036

import numpy as np
import time

time_start = time.perf_counter()

eps = 1.4
a = 0.5
N = 20
N_cor = 20
N_cf = 25       #SWEEP ON THIS

x = np.zeros(N)
G = np.zeros((N_cf, N))

def update(x):
    for j in range(0,N):
        
        old_x = x[j] # save original value
        old_Sj = S(j,x)
        x[j] = x[j] + np.random.uniform(-eps,eps) # update x[j]
        dS = S(j,x) - old_Sj # change in action
        
        if dS>0 and np.exp(-dS)<np.random.uniform(0,1):
            x[j] = old_x # restore old value

def S(j,x): # harm. osc. S
    
    jp = (j+1)%N # next site
    jm = (j-1)%N # previous site
   
    return a*x[j]**2/2 + x[j]*(x[j]-x[jp]-x[jm])/a

def compute_G(x,n):
    g = 0
    for j in range(0,N):
        g = g + x[j]*x[(j+n)%N]
    return g/N

def MCaverage(x,G):
    for j in range(0,N): # initialize x
        x[j] = 0
    for j in range(0,5*N_cor): # thermalize x   
        update(x)
    for alpha in range(0,N_cf): # loop on random paths
        for j in range(0,N_cor):
            update(x)
        for n in range(0,N):
            G[alpha][n] = compute_G(x,n)
    for n in range(0,N): # compute MC averages
        avg_G = 0
        for alpha in range(0,N_cf):
            avg_G = avg_G + G[alpha][n]
        avg_G = avg_G/N_cf
        print(f'G({n}) = {avg_G}')


MCaverage(x, G)

def avg(G): # MC avg of G
    return np.sum(G)/len(G)

def sdev(G): # std dev of G
    g = np.asarray(G)
    return np.abs(avg(g**2)-avg(g)**2)**0.5

print('avg G\n', avg(G))
print('avg G (binned)\n', avg(bin(G,4)))
print ('avg G (bootstrap)\n', avg(bootstrap(G)))

def deltaE(G): # Delta E(t)
    avgG = avg(G)
    adE = np.log(np.abs(avgG[:-1]/avgG[1:]))
    return adE/a

print('Delta E\n', deltaE(G))
print('Delta E (bootstrap)\n', deltaE(bootstrap(G)))

def bootstrap_deltaE(G,nbstrap=100): # Delta E + errors
    avgE = deltaE(G) # avg deltaE
    bsE = []
    for i in range(nbstrap): # bs copies of deltaE
        g = bootstrap(G)
        bsE.append(deltaE(g))
    bsE = np.array(bsE)
    sdevE = sdev(bsE) # spread of deltaE’s
    print('{n%2s} {%10s} {%10s}'.format("t","Delta E(t)","error"))
    print(26*"-")
    for i in range(len(avgE)/2):
        print(f'{i} {avgE[i]} {sdevE[i]}')

bootstrap_deltaE(G)

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))