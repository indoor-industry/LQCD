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
Ncf = 10

def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.conj().T @ m)

@jit(nopython=True)
def factorial(x):
    fact: float = 1
    for i in np.arange(2, x+1, 1):
        i = float(i)
        fact *= i
    return fact

#function to calculate adjoint of matrix
@jit(nopython=True)
def dag(M):
    return M.conj().T

#generate a random SU(3) matrix
@jit(nopython=True)
def SU3(steps=30):
    #matrix with entries between -1 and 1 to initialise
    ones = (np.random.rand(3, 3)*2 - 1)*1 + (np.random.rand(3, 3)*2 - 1)*1j
    #make it hermitian
    H = (1/2)*(ones + dag(ones))
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
        M = SU3()
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
    
    point_clockwise = point.copy()
    point_anticlockwise = point.copy()

    up(point_clockwise, starting_direction)                           #move up initial link
    up(point_anticlockwise, starting_direction)                           #move up initial link

    clockwise = np.zeros((3, 3), np.complex128)
    anticlockwise = np.zeros((3, 3), np.complex128)
    gamma = np.zeros((3, 3), np.complex128)
    for direction in range(dim):                                    #cycle over directions other than the starting_direction
        if direction != starting_direction:
            right = call_link(point_clockwise, direction, lattice, dagger=False)                  #take link pointing "right"
            right = np.ascontiguousarray(right)
            up(point_clockwise, direction)
            down(point_clockwise, starting_direction)                                    #move "right"
            right_down = call_link(point_clockwise, starting_direction, lattice, dagger=True)    #take link moving "down"
            right_down = np.ascontiguousarray(right_down)
            down(point_clockwise, direction)                         #move "down"
            right_down_left = call_link(point_clockwise, direction, lattice, dagger=True)             #take link moving "left"
            right_down_left = np.ascontiguousarray(right_down_left)
            up(point_clockwise, starting_direction)

            down(point_anticlockwise, direction)
            left = call_link(point_anticlockwise, direction, lattice, dagger=True)
            left = np.ascontiguousarray(left)
            down(point_anticlockwise, starting_direction)
            left_down = call_link(point_anticlockwise, starting_direction, lattice, dagger=True)
            left_down = np.ascontiguousarray(left_down)
            left_down_right = call_link(point_anticlockwise, direction, lattice, dagger=False)
            left_down_right = np.ascontiguousarray(left_down_right)
            up(point_anticlockwise, direction)
            up(point_anticlockwise, starting_direction)

            clockwise += (right @ right_down) @ right_down_left
            anticlockwise += (left @ left_down) @ left_down_right

    gamma = clockwise + anticlockwise
    
    return gamma

@jit(nopython=True)
def metropolis_update(lattice, matrices, hits=10):
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(dim):
                        point = [t, x, y, z]
                        gamma = gamma_plaquette(lattice, point, mu)
                        for i in range(hits):
                            rand = np.random.randint(2*N_mat)
                            M = matrices[rand]
                            old_link = call_link(point, mu, lattice, dagger=False)
                            old_link = np.ascontiguousarray(old_link)
                            new_link = M @ old_link

                            dS= -(beta/3)*np.real(np.trace((new_link - old_link) @ gamma))
                            if dS < 0 or np.exp(-dS) > np.random.uniform(0, 1):
                                lattice[point[0], point[1], point[2], point[3], mu] = new_link

@jit(nopython=True)
def wilson_plaquette(lattice, starting_point):

    point = starting_point.copy()

    w_plaquette = 0
    for starting_direction in range(dim):
        for direction in range(starting_direction):
            link_up = call_link(point, starting_direction, lattice, dagger=False)
            link_up = np.ascontiguousarray(link_up)
            up(point, starting_direction)                                    #move "up"
            link_right = call_link(point, direction, lattice, dagger=False)    #take link moving "down"
            link_right = np.ascontiguousarray(link_right)
            up(point, direction)                                    #move "up"
            down(point, starting_direction)
            link_down = call_link(point, starting_direction, lattice, dagger=True)    #take link moving "down"
            link_down = np.ascontiguousarray(link_down)
            down(point, direction)
            link_left = call_link(point, direction, lattice, dagger=True)
            link_left = np.ascontiguousarray(link_left)

            w_plaquette += (1/3)*np.real(np.trace(link_up @ link_right @ link_down @ link_left))

    return w_plaquette/6

@jit(nopython=True)
def wilson_rectangle(lattice, starting_point):

    point = starting_point.copy()

    w_rectangle = 0
    for starting_direction in range(dim):
        for direction in range(starting_direction):
            link_up = call_link(point, starting_direction, lattice, dagger=False)
            link_up = np.ascontiguousarray(link_up)
            up(point, starting_direction)                                    #move "up"
            link_right = call_link(point, direction, lattice, dagger=False)    #take link moving "down"
            link_right = np.ascontiguousarray(link_right)
            up(point, direction)                                    #move "up"
            link_right_right = call_link(point, direction, lattice, dagger=False)    #take link moving "down"
            link_right_right = np.ascontiguousarray(link_right_right)
            up(point, direction)
            down(point, starting_direction)
            link_down = call_link(point, starting_direction, lattice, dagger=True)    #take link moving "down"
            link_down = np.ascontiguousarray(link_down)
            down(point, direction)
            link_left = call_link(point, direction, lattice, dagger=True)
            link_left = np.ascontiguousarray(link_left)
            down(point, direction)
            link_left_left = call_link(point, direction, lattice, dagger=True)
            link_left_left = np.ascontiguousarray(link_left_left)

            w_rectangle += (1/3)*np.real(np.trace(link_up @ link_right @ link_right_right @ link_down @ link_left @ link_left_left))

    return w_rectangle/6

@jit(nopython=True)
def wilson_over_lattice(lattice, matrices, shape):
    W_plaquettes = np.zeros(Ncf, dtype=np.float64)
    for alpha in range(Ncf):
        for skip in range(Ncor):
            metropolis_update(lattice, matrices, hits=10)
        for t in range(N):
            for x in range(N):
                for y in range(N):
                    for z in range(N):
                        point = np.array([t, x, y, z])
                        if shape == 'axa':
                            W_plaquettes[alpha] += wilson_plaquette(lattice, point)
                        elif shape == '2axa':
                            W_plaquettes[alpha] += wilson_rectangle(lattice, point)
        print(W_plaquettes[alpha] / N**dim)
    return W_plaquettes/N**dim, shape

def main():

    time_start = time.perf_counter()

    Ms = matrices(N_mat)            #generate SU(3) matrix pool
    lattice = initialise_lattice(N, dim)        #initialize lattice

    for i in tqdm(range(2*Ncor)):
        metropolis_update(lattice , Ms)       #thermalize lattice for 2*Ncor steps

    axa, shape = wilson_over_lattice(lattice, Ms, shape='2axa')
    np.savetxt(f'data/{shape}.csv', axa)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))


if __name__ == '__main__':
    main()
