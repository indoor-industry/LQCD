import numpy as np
import time
from tqdm import tqdm
from numba import jit

#choose 'n' for unimproved antion and 'y' otherwise
improve = 'n'
#atoms per side of lattice
N = 8
#parameter for creation of SU(3) matrices, affects lattice creation and metropolis acceptance ratio
eps_mat = 0.24
#parameter for smearing
eps_smear = 1/12
#action parameter
beta = 5.5      #includes tadpole improvement
#improved action parameter
beta_improved = 1.719
#tadpole improvement for imporved action
u_0 = 0.797
#lattice spacing
a = 0.25
#number of lattice evolutions before acquiring a measurement to avoid correlations
Ncor = 50
#space+time dimensions
dim = 4
#size of pool of SU(3) matrices (includes just as many hermitian conjugates of them)
N_mat = 100
#number of aquisitions performed
Ncf = 10

#fucntion to check if a matrix is unitary
def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.conj().T @ m)

#homemade factorial function for numba
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
        U = U + ((1j*eps_mat)**i/factorial(i))*np.linalg.matrix_power(H, i)
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
    print('Matrices ready')
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

#BOTH UP AND DOWN FUNCTIONS KEEP MEMORY OF THE NEW POSITION OF THE POINT
#move a coordinate point up a direction in the lattice
@jit(nopython=True)
def up(coordinate, direction):
    coordinate[direction] = (coordinate[direction] + 1)%N
    return coordinate

#move a coordinate point down a direction in the lattice
@jit(nopython=True)
def down(coordinate, direction):
    coordinate[direction] = (coordinate[direction] - 1)%N
    return coordinate

#call a link SU(3) at a certain point in the lattice given a direction or its hermitian conjugate if direction is negative
@jit(nopython=True)
def call_link(point, direction, lattice, dagger:bool):
    if dagger == False:
        return lattice[point[0], point[1], point[2], point[3], direction]
    elif dagger == True:
        return dag(lattice[point[0], point[1], point[2], point[3], direction])

#calculate the main part of the variation in action for the unimproved action
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
            link_right = call_link(point_clockwise, direction, lattice, dagger=False)                  #take link pointing "right"
            link_right = np.ascontiguousarray(link_right)
            up(point_clockwise, direction)                                                             #move "up"
            down(point_clockwise, starting_direction)                                                  #move "down"
            link_right_down = call_link(point_clockwise, starting_direction, lattice, dagger=True)     #take link pointing "down"
            link_right_down = np.ascontiguousarray(link_right_down)
            down(point_clockwise, direction)                                                           #move "left"
            link_right_down_left = call_link(point_clockwise, direction, lattice, dagger=True)         #take link pointing "left"
            link_right_down_left = np.ascontiguousarray(link_right_down_left)
            up(point_clockwise, starting_direction)                                                    #back to initial position

            down(point_anticlockwise, direction)
            link_left = call_link(point_anticlockwise, direction, lattice, dagger=True)
            link_left = np.ascontiguousarray(link_left)
            down(point_anticlockwise, starting_direction)
            link_left_down = call_link(point_anticlockwise, starting_direction, lattice, dagger=True)
            link_left_down = np.ascontiguousarray(link_left_down)
            link_left_down_right = call_link(point_anticlockwise, direction, lattice, dagger=False)
            link_left_down_right = np.ascontiguousarray(link_left_down_right)
            up(point_anticlockwise, direction)
            up(point_anticlockwise, starting_direction)

            clockwise += (link_right @ link_right_down) @ link_right_down_left
            anticlockwise += (link_left @ link_left_down) @ link_left_down_right

    gamma = clockwise + anticlockwise
    
    return gamma

#another part in the variation of the action for the imporved case, much longer but same basic reasoning as plaquette
@jit(nopython=True)
def gamma_rectangle(lattice, point, starting_direction):
    
    point_clockwise_vertical_down = point.copy()
    point_anticlockwise_vertical_down = point.copy()
    point_clockwise_vertical_up = point.copy()
    point_anticlockwise_vertical_up = point.copy()
    point_clockwise_horizontal = point.copy()
    point_anticlockwise_horizontal = point.copy()

    up(point_clockwise_vertical_down, starting_direction)                           #move up initial link
    up(point_clockwise_vertical_up, starting_direction)                           #move up initial link
    up(point_anticlockwise_vertical_down, starting_direction)                           #move up initial link
    up(point_anticlockwise_vertical_up, starting_direction)                           #move up initial link
    up(point_clockwise_horizontal, starting_direction)                           #move up initial link
    up(point_anticlockwise_horizontal, starting_direction)                           #move up initial link

    clockwise_vertical_up = np.zeros((3, 3), np.complex128)
    clockwise_vertical_down = np.zeros((3, 3), np.complex128)
    anticlockwise_vertical_up = np.zeros((3, 3), np.complex128)
    anticlockwise_vertical_down = np.zeros((3, 3), np.complex128)
    clockwise_horizonal = np.zeros((3, 3), np.complex128)
    anticlockwise_horizontal = np.zeros((3, 3), np.complex128)
    
    gamma = np.zeros((3, 3), np.complex128)
    for direction in range(dim):                                    #cycle over directions other than the starting_direction
        if direction != starting_direction:
####################################################################################################################
            
            link_up = call_link(point_clockwise_vertical_up, starting_direction, lattice, dagger=False)                  #take link pointing "right"
            link_up = np.ascontiguousarray(link_up)
            
            #clockwise vertical up
            up(point_clockwise_vertical_up, starting_direction)
            link_up_right = call_link(point_clockwise_vertical_up, direction, lattice, dagger=False)
            link_up_right = np.ascontiguousarray(link_up_right)
            up(point_clockwise_vertical_up, direction)                                    #move "right"
            down(point_clockwise_vertical_up, starting_direction)
            link_up_right_down = call_link(point_clockwise_vertical_up, starting_direction, lattice, dagger=True)    #take link moving "down"
            link_up_right_down = np.ascontiguousarray(link_up_right_down)
            down(point_clockwise_vertical_up, starting_direction)                         #move "down"
            link_up_right_down_down = call_link(point_clockwise_vertical_up, starting_direction, lattice, dagger=True)    #take link moving "down"
            link_up_right_down_down = np.ascontiguousarray(link_up_right_down_down)
            down(point_clockwise_vertical_up, direction)
            link_up_right_down_down_left = call_link(point_clockwise_vertical_up, direction, lattice, dagger=True)
            link_up_right_down_down_left = np.ascontiguousarray(link_up_right_down_down_left)
            up(point_clockwise_vertical_up, starting_direction)

            #anticlockwise vertical up
            up(point_anticlockwise_vertical_up, starting_direction)
            down(point_anticlockwise_vertical_up, direction)
            link_up_left = call_link(point_anticlockwise_vertical_up, direction, lattice, dagger=True)
            link_up_left = np.ascontiguousarray(link_up_left)
            down(point_anticlockwise_vertical_up, starting_direction)
            link_up_left_down = call_link(point_anticlockwise_vertical_up, starting_direction, lattice, dagger=True)    #take link moving "down"
            link_up_left_down = np.ascontiguousarray(link_up_left_down)
            down(point_anticlockwise_vertical_up, starting_direction)                         #move "down"
            link_up_left_down_down = call_link(point_anticlockwise_vertical_up, starting_direction, lattice, dagger=True)    #take link moving "down"
            link_up_left_down_down = np.ascontiguousarray(link_up_left_down_down)
            link_up_left_down_down_right = call_link(point_anticlockwise_vertical_up, direction, lattice, dagger=False)
            link_up_left_down_down_right = np.ascontiguousarray(link_up_left_down_down_right)
            up(point_anticlockwise_vertical_up, direction)
            up(point_anticlockwise_vertical_up, starting_direction)
#########################################################################################################################################


#########################################################################################################################################
            link_right = call_link(point_clockwise_vertical_down, direction, lattice, dagger=False)                  #take link pointing "right"
            link_right = np.ascontiguousarray(link_right)

            #clockwise vertical down
            up(point_clockwise_vertical_down, direction)
            down(point_clockwise_vertical_down, starting_direction)                                    #move "right"
            link_right_down = call_link(point_clockwise_vertical_down, starting_direction, lattice, dagger=True)    #take link moving "down"
            link_right_down = np.ascontiguousarray(link_right_down)
            down(point_clockwise_vertical_down, starting_direction)                         #move "down"
            link_right_down_down = call_link(point_clockwise_vertical_down, starting_direction, lattice, dagger=True)             #take link moving "left"
            link_right_down_down = np.ascontiguousarray(link_right_down_down)
            down(point_clockwise_vertical_down, direction)
            link_right_down_down_left = call_link(point_clockwise_vertical_down, direction, lattice, dagger=True)
            link_right_down_down_left = np.ascontiguousarray(link_right_down_down_left)
            link_right_down_down_left_up = call_link(point_clockwise_vertical_down, starting_direction, lattice, dagger=False)
            link_right_down_down_left_up = np.ascontiguousarray(link_right_down_down_left_up)
            up(point_clockwise_vertical_down, starting_direction)
            up(point_clockwise_vertical_down, starting_direction)

            #clockwise horizonal
            up(point_clockwise_horizontal, direction)
            link_right_right = call_link(point_clockwise_horizontal, direction, lattice, dagger=False)                  #take link pointing "right"
            link_right_right = np.ascontiguousarray(link_right_right)
            up(point_clockwise_horizontal, direction)
            down(point_clockwise_horizontal, starting_direction)                                    #move "right"
            link_right_right_down = call_link(point_clockwise_horizontal, starting_direction, lattice, dagger=True)    #take link moving "down"
            link_right_right_down = np.ascontiguousarray(link_right_right_down)
            down(point_clockwise_horizontal, direction)                         #move "down"
            link_right_right_down_left = call_link(point_clockwise_horizontal, direction, lattice, dagger=True)             #take link moving "left"
            link_right_right_down_left = np.ascontiguousarray(link_right_right_down_left)
            down(point_clockwise_horizontal, direction)                         #move "down"
            link_right_right_down_left_left = call_link(point_clockwise_horizontal, direction, lattice, dagger=True)             #take link moving "left"
            link_right_right_down_left_left = np.ascontiguousarray(link_right_right_down_left_left)
            up(point_clockwise_horizontal, starting_direction)
################################################################################################################################


###################################################################################################################################
            down(point_anticlockwise_vertical_down, direction)
            down(point_anticlockwise_horizontal, direction)
            link_left = call_link(point_anticlockwise_vertical_down, direction, lattice, dagger=True)
            link_left = np.ascontiguousarray(link_left)

            #anticlockwise vertical down
            down(point_anticlockwise_vertical_down, starting_direction)
            link_left_down = call_link(point_anticlockwise_vertical_down, starting_direction, lattice, dagger=True)
            link_left_down = np.ascontiguousarray(link_left_down)
            down(point_anticlockwise_vertical_down, starting_direction)
            link_left_down_down = call_link(point_anticlockwise_vertical_down, starting_direction, lattice, dagger=True)
            link_left_down_down = np.ascontiguousarray(link_left_down_down)
            link_left_down_down_right = call_link(point_anticlockwise_vertical_down, direction, lattice, dagger=False)
            link_left_down_down_right = np.ascontiguousarray(link_left_down_down_right)
            up(point_anticlockwise_vertical_down, direction)
            link_left_down_down_right_up = call_link(point_anticlockwise_vertical_down, starting_direction, lattice, dagger=False)
            link_left_down_down_right_up = np.ascontiguousarray(link_left_down_down_right_up)
            up(point_anticlockwise_vertical_down, starting_direction)
            up(point_anticlockwise_vertical_down, starting_direction)

            #anticlockwise horizontal
            down(point_anticlockwise_horizontal, direction)
            link_left_left = call_link(point_anticlockwise_horizontal, direction, lattice, dagger=True)
            link_left_left = np.ascontiguousarray(link_left_left)
            down(point_anticlockwise_horizontal, starting_direction)
            link_left_left_down = call_link(point_anticlockwise_horizontal, starting_direction, lattice, dagger=True)
            link_left_left_down = np.ascontiguousarray(link_left_left_down)
            link_left_left_down_right = call_link(point_anticlockwise_horizontal, direction, lattice, dagger=False)
            link_left_left_down_right = np.ascontiguousarray(link_left_left_down_right)
            up(point_anticlockwise_horizontal, direction)
            link_left_left_down_right_right = call_link(point_anticlockwise_horizontal, direction, lattice, dagger=False)
            link_left_left_down_right_right = np.ascontiguousarray(link_left_left_down_right_right)
            up(point_anticlockwise_horizontal, direction)
            up(point_anticlockwise_horizontal, starting_direction)
###########################################################################################################################################

            clockwise_vertical_up += link_up @ link_up_right @ link_up_right_down @  link_up_right_down_down @ link_up_right_down_down_left
            clockwise_vertical_down += link_right @ link_right_down @ link_right_down_down @ link_right_down_down_left @ link_right_down_down_left_up
            anticlockwise_vertical_up += link_up @ link_up_left @ link_up_left_down @ link_up_left_down_down @ link_up_left_down_down_right
            anticlockwise_vertical_down += link_left @ link_left_down @ link_left_down_down @ link_left_down_down_right @ link_left_down_down_right_up
            clockwise_horizonal += link_right @ link_right_right @ link_right_right_down @ link_right_right_down_left @ link_right_right_down_left_left
            anticlockwise_horizontal += link_left @ link_left_left @ link_left_left_down @ link_left_left_down_right @ link_left_left_down_right_right

    gamma = clockwise_vertical_up + clockwise_vertical_down + anticlockwise_vertical_up + anticlockwise_vertical_down + clockwise_horizonal + anticlockwise_horizontal
    
    return gamma

#metropolis update function
@jit(nopython=True)
def metropolis_update(lattice, matrices, hits=10):
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(dim):
                        point = [t, x, y, z]
                        if improve == 'n':
                            gamma_P = gamma_plaquette(lattice, point, mu)
                            for i in range(hits):                               #update a number of times before acquiring measurements
                                rand = np.random.randint(2*N_mat)
                                M = matrices[rand]
                                old_link = call_link(point, mu, lattice, dagger=False)
                                old_link = np.ascontiguousarray(old_link)
                                new_link = M @ old_link

                                dS = -(beta/3)*np.real(np.trace((new_link - old_link) @ gamma_P))
                                if dS < 0 or np.exp(-dS) > np.random.uniform(0, 1):
                                    lattice[point[0], point[1], point[2], point[3], mu] = new_link

                        elif improve == 'y':
                            gamma_P = gamma_plaquette(lattice, point, mu)
                            gamma_R = gamma_rectangle(lattice, point, mu)
                            for i in range(hits):
                                rand = np.random.randint(2*N_mat)
                                M = matrices[rand]
                                old_link = call_link(point, mu, lattice, dagger=False)
                                old_link = np.ascontiguousarray(old_link)
                                new_link = M @ old_link

                                dS = -(beta_improved/3)*(5/(3*u_0**4)*np.real(np.trace((new_link-old_link) @ gamma_P))-1/(12*u_0**6)*np.real(np.trace((new_link - old_link) @ gamma_R)))
                                if dS < 0 or np.exp(-dS) > np.random.uniform(0, 1):
                                    lattice[point[0], point[1], point[2], point[3], mu] = new_link

@jit(nopython=True)
def planar_loops(lattice, point, length, duration):
    W_planar=0
    for space_direction in range(1, 4, 1):
        loop = np.identity(3, np.complex128)
        for time in range(duration):
            link = call_link(point, 0, lattice, dagger=False)
            link = np.ascontiguousarray(link)
            loop = loop @ link
            up(point, 0)
        for space in range(length):
            link = call_link(point, space_direction, lattice, dagger=False)
            link = np.ascontiguousarray(link)
            loop = loop @ link
            up(point, space_direction)
        for time_reverse in range(duration):
            down(point, 0)
            link = call_link(point, 0, lattice, dagger=True)
            link = np.ascontiguousarray(link)
            loop = loop @ link
        for space_reverse in range(length):
            down(point, space_direction)
            link = call_link(point, space_direction, lattice, dagger=True)
            link = np.ascontiguousarray(link)
            loop = loop @ link
        W_planar += (1/3)*np.real(np.trace(loop))
    return W_planar/3

#calculate wichever shape of wilson loops opver the whole lattice and average
@jit(nopython=True)
def planar_loop_over_lattice(lattice, matrices, length, duration):
    W_planar = np.zeros(Ncf, dtype=np.float64)
    for alpha in range(Ncf):
        for skip in range(Ncor):
            metropolis_update(lattice, matrices, hits=10)
        for t in range(N):
            for x in range(N):
                for y in range(N):
                    for z in range(N):
                        point = np.array([t, x, y, z])
                        W_planar[alpha] += planar_loops(lattice, point, length, duration)
        
        print(W_planar[alpha] / N**dim)
    
    return W_planar/N**dim

@jit(nopython=True)
def Wilson(lattice, Ms, max_r, min_t, max_t):
    W_planar_r_t = np.empty((max_r, max_t-min_t))
    W_planar_r_t_err = np.empty((max_r, max_t-min_t))
    for r in range(1, max_r):
        for t in range(min_t, max_t):
            W_r = planar_loop_over_lattice(lattice, Ms, r, t)
            W_planar_r_t[r-1, t-1] = mean(W_r)
            W_planar_r_t_err[r-1, t-1] = stdev(W_r)

    return W_planar_r_t, W_planar_r_t_err, r

@jit(nopython=True)
def gauge_covariant_derivative(lattice, point, starting_direction):
    link_up = call_link(point, starting_direction, lattice, dagger=False)
    link_up = np.ascontiguousarray(link_up)
    for direction in range(dim):

        link_right = call_link(point, direction, lattice, dagger=False)
        link_right = np.ascontiguousarray(link_right)
        up(point, direction)
        link_right_up = call_link(point, starting_direction, lattice, dagger=False)
        link_right_up = np.ascontiguousarray(link_right_up)
        up(point, starting_direction)
        down(point, direction)
        link_right_up_left = call_link(point, direction, lattice, dagger=True)
        link_right_up_left = np.ascontiguousarray(link_right_up_left)

        down(point, direction)
        link_left_up_right = call_link(point, direction, lattice, dagger=False)
        link_left_up_right = np.ascontiguousarray(link_left_up_right)
        down(point, starting_direction)
        link_left_up = call_link(point, starting_direction, lattice, dagger=False)
        link_left_up = np.ascontiguousarray(link_left_up)
        link_left = call_link(point, direction, lattice, dagger=True)
        link_left = np.ascontiguousarray(link_left)
        up(point, direction)
        up(point, starting_direction)

    loop_right = link_right @ link_right_up @ link_right_up_left
    loop_left = link_left @ link_left_up @ link_left_up_right

    smeared_link = (1/(u_0*a)**2)*(loop_right - 2*(u_0**2)*link_up + loop_left)
    return smeared_link

@jit(nopython=True)        
def smear_lattice(lattice, smearing_eps):
    smeared_lattice = np.empty((N, N, N, N, dim, 3, 3), dtype=np.complex128)
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    point = np.array([t, x, y, z])
                    for direction in range(dim):
                        smeared_lattice[t, x, y, z, direction] = lattice[t, x, y, z, direction] + smearing_eps*(a**2)*gauge_covariant_derivative(lattice, point, direction)

    return smeared_lattice

@jit(nopython=True)
def smearings(lattice, number_of_smears):
    repeatedly_smeared_lattice = lattice.copy()
    for i in number_of_smears:
        repeatedly_smeared_lattice = smear_lattice(repeatedly_smeared_lattice, smearing_eps=1/12)
    return repeatedly_smeared_lattice

@jit(nopython=True)
def mean(data):
    return np.sum(data)/len(data)

@jit(nopython=True)
def stdev(data, ddof=0):
    #calculate mean of data
    mean_data = sum(data) / len(data)
    # Calculate squared differences for each data point and mean
    squared_diffs = [(x - mean_data) ** 2 for x in data]
    # Calculate the average of the squared differences
    variance = sum(squared_diffs) / (len(squared_diffs) - ddof)
    # Calculate the square root of the variance
    stddev = variance ** 0.5
    return stddev

def main():

    time_start = time.perf_counter()

    Ms = matrices(N_mat)            #generate SU(3) matrix pool
    lattice = initialise_lattice(N, dim)        #initialize lattice

    for i in tqdm(range(2*Ncor)):
        metropolis_update(lattice , Ms)       #thermalize lattice for 2*Ncor steps

    #lattice = np.load('data/lattice imp=n.npy')

    max_r = 3
    min_t = 4
    max_t = 6
    W_planar_r_t, W_planar_r_t_err, radius = Wilson(lattice, Ms, max_r, min_t, max_t)

    np.save('data/W_rt.npy', W_planar_r_t)
    np.save('data/W_rt_err.npy', W_planar_r_t_err)

    radius = range(1, max_r)
    np.save('data/radius.npy', radius)

    time_elapsed = (time.perf_counter() - time_start)
    print ("checkpoint %5.1f secs" % (time_elapsed))


if __name__ == '__main__':
    main()