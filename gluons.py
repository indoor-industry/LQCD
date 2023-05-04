import numpy as np

N = 8
eps = 0.24
beta = 5.5      #includes tadpole improvement
a = 0.25        #units are fm
Ncor = 50

#generate a random SU(3) matrix
def SU3():
    #matrix with entries between -1 and 1
    rand = np.random.rand(3, 3)*2 - 1
    #make it symmetric such that it is hermitian
    H = rand + rand.T
    #transform it trough a variation
    M = np.ones((3, 3)) + 1j*eps*H
    #normalize fisrt column to unity
    m1 = M[0]
    m1 /= np.sqrt(sum(m1**2))
    #make second vector orthogonal to m1
    m2 = np.random.randn(3)  # take a random vector
    m2 = m2 -  (m2.dot(m1)*m1/sum(m1**2))       # make it orthogonal to k
    m2 /= np.sqrt(sum(m2**2))  # normalize it
    #make third vector orthogonal to both trough cross product
    m3 = np.cross(m1, m2)
    #make them into a matrix
    U = np.stack((m1, m2, m3))
    #turn it from unitary to special unitary by extracting the phase
    SU = U/(np.linalg.det(U))**(1/3)
    return SU

M = SU3()
print(np.dot(M,M.conj().T)) #must be the unit matrix
