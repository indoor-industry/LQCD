import numpy as np
import time
from tqdm import tqdm
from numba import jit

N = 8
eps = 0.24
beta = 5.5      #includes tadpole improvement
a = 0.25        #units are fm
Ncor = 50
dim = 4
N_mat = 100

@jit(nopython=True)
def factorial(x):
    fact = 1
    for i in np.arange(2, x+1, 1):
        fact *= i
    return fact

#function to calculate adjoint of matrix
@jit(nopython=True)
def dag(M):
    return M.conj().T

#generate a random SU(3) matrix
steps=30
@jit(nopython=True)
def SU3(steps):
    #matrix with entries between -1 and 1 to initialise
    ones = (np.random.rand(3, 3)*2 - 1)*(1+1j)
    #make it hermitian
    H = 0.5*(ones + dag(ones))
    #make it unitary
    U = np.zeros((3, 3), np.complex128)
    for i in range(steps):
        U = U + ((1j*eps)**i/factorial(i))*np.linalg.matrix_power(H, i)
    #make it special
    SU = U/(np.linalg.det(U))**(1/3)
    return SU

#create an array of SU3 matrices OF LENGHT 2*N_mat and store them away, make sure it also contains the hermitian conjugate of each
@jit(nopython=True)
def matrices(N_mat):
    Ms = np.empty((2*N_mat, 3, 3), np.complex128)
    for i in range(N_mat):
        M = SU3(steps)
        Ms[i] = M
        Ms[N_mat+i] = dag(M)
    return Ms

#generate lattice with identity matrices on each node
@jit(nopython=True)
def initialise_lattice(lattice_size, dimensions):
    lat = np.empty((lattice_size, lattice_size, lattice_size, lattice_size, dimensions, 3, 3), np.complex128)
    for t in range(lattice_size):
        for x in range(lattice_size):
            for y in range(lattice_size):
                for z in range(lattice_size):
                    for dim in range(dimensions):
                        lat[t][x][y][z][dim] = np.identity(3, np.complex128)
    return lat

@jit(nopython=True)
def up(coordinate, direction):
    coordinate[direction] = (coordinate[direction] + 1)%N
    return coordinate

@jit(nopython=True)
def down(coordinate, direction):
    coordinate[direction] = (coordinate[direction] - 1)%N
    return coordinate

@jit(nopython=True)
def call_link(point, direction, lattice, dagger:bool):
    if dagger == False:
        return lattice[point[0], point[1], point[2], point[3], direction]
    elif dagger == True:
        return dag(lattice[point[0], point[1], point[2], point[3], direction])

@jit(nopython=True)
def gamma_plaquette(lattice, point, starting_direction):
    up(point, starting_direction)                           #move on initial link

    point_clockwise = point.copy()
    point_anticlockwise = point.copy()

    clockwise = np.zeros((3, 3), np.complex128)
    anticlockwise = np.zeros((3, 3), np.complex128)
    gamma = np.zeros((3, 3), np.complex128)
    for direction in range(dim):                                    #cycle over directions other than the starting_direction
        if direction != starting_direction:
            right = call_link(point_clockwise, direction, lattice, dagger=False)                  #take link pointing "right"
            right = np.ascontiguousarray(right)
            up(point_clockwise, direction)                                    #move "right"
            right_down = call_link(point_clockwise, starting_direction, lattice, dagger=True)    #take link moving "down"
            right_down = np.ascontiguousarray(right_down)
            down(point_clockwise, starting_direction)                         #move "down"
            right_down_left = call_link(point_clockwise, direction, lattice, dagger=True)             #take link moving "left"
            right_down_left = np.ascontiguousarray(right_down_left)

            left = call_link(point_anticlockwise, direction, lattice, dagger=True)
            left = np.ascontiguousarray(left)
            down(point_anticlockwise, direction)
            left_down = call_link(point_anticlockwise, starting_direction, lattice, dagger=True)
            left_down = np.ascontiguousarray(left_down)
            down(point_anticlockwise, starting_direction)
            left_down_right = call_link(point_anticlockwise, direction, lattice, dagger=False)
            left_down_right = np.ascontiguousarray(left_down_right)
            
            clockwise += (right @ right_down) @ right_down_left
            anticlockwise += (left @ left_down) @ left_down_right

            gamma += clockwise + anticlockwise
    
    return gamma

@jit(nopython=True)
def metropolis_update(lattice, matrices, n):
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(dim):
                        point = [t, x, y, z]
                        gamma = gamma_plaquette(lattice, point, mu)
                        for i in range(n):
                            rand = np.random.randint(2*N_mat)
                            M = matrices[rand]
                            old_link = call_link(point, mu, lattice, dagger=False)
                            old_link = np.ascontiguousarray(old_link)
                            new_link = M @ old_link

                            dS= np.real(np.trace((old_link- new_link) @ gamma))
                            if dS < 0 or np.exp(-dS) > np.random.uniform(0, 1):
                                lattice[point[0], point[1], point[2], point[3], mu] = new_link


def main():

    time_start = time.perf_counter()

    Ms = matrices(N_mat)
    lattice = initialise_lattice(N, dim)

    #TEST
    point = np.ones(4, dtype=np.int8)
    direction = 2
    U = call_link(point, direction, lattice, dagger=False)
    print(U)
    print(gamma_plaquette(lattice, point, 2))

    for i in tqdm(range(2*Ncor)):
        metropolis_update(lattice , Ms, n=10)

    #TEST
    point = np.ones(4, dtype=np.int8)
    direction = 2
    U = call_link(point, direction, lattice, dagger=False)
    print(U)
    print(gamma_plaquette(lattice, point, 2))

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))


if __name__ == '__main__':
    main()
