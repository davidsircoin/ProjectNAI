import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

######DATA######

# Velocity of water for different distances
distance_from_plume = [0, 250, 500, 750, 1000]
measured_velocity = [0.03, 0.032, 0.031, 0.027, 0.029]

t_0 = -1.1 # temperature at the plume
t_1000 = 1 # temperature 1000m away from the plume

psu_0 = 34.1 # salinity at the plume
psu_1000 = 35 # salinity 1000m away from the plume

turb = 100 # turbulence constant


################
# EQUATION TO SOLVE:
# k*y'' - u(x)*y' - u'(x)*y = 0


# We use picewise cubic interpolation for the velocity data-points
# Since we also need the derivative of the velocity later on linear interpolation isn't an option.
velocity = sp.interpolate.CubicSpline(distance_from_plume, measured_velocity)
velocity_p = velocity.derivative()
x = 0



def main():
    # uncomment these to test what happens if the velocity is constant
    #velocity = lambda _: 0.03
    #velocity_p = lambda _: 0

    N = 100
    xs = np.linspace(0, 1000, N)

    velocities = list(map(velocity, xs))
    velocity_ps = list(map(velocity_p, xs))

    _, ys_temperature_D, _ = ComputeNumericalSolution(velocity, velocity_p, N-2, t_0, t_1000) # Temperature
    _, ys_salinity_D, _ = ComputeNumericalSolution(velocity, velocity_p, N-2, psu_0, psu_1000) # Salinity

    ys_temperature_E = bvp_solve(velocity, velocity_p, t_0, t_1000, N)
    ys_salinity_E = bvp_solve(velocity, velocity_p, psu_0, psu_1000, N)

    fig, ax = plt.subplots(3,2)
    ax[0,0].plot(xs, velocities, label=f'v')
    ax[0,0].scatter(distance_from_plume, measured_velocity)
    ax[0,1].plot(xs, velocity_ps, label=f'v\'')
    ax[1,0].plot(xs, ys_temperature_D, label=f'Temperature N={N}')
    ax[1,1].plot(xs, ys_salinity_D, label=f'Salinity (PSU) N={N}')
    ax[2,0].plot(xs, ys_temperature_E, label=f'Temperature N={N}')
    ax[2,1].plot(xs, ys_salinity_E, label=f'Salinity (PSU) N={N}')
    fig.legend()
    plt.show()



# Numerical solver of ODE. Still need to check if the entries are completely correct.
def ComputeNumericalSolution(velocity, velocity_p, N , leftbc , rightbc ):
    h = ( 1000 ) / ( N + 1 )
    x = np . linspace (0 , 1000, N + 2 )
    A = np . zeros (( N , N ) )
    B = np.zeros((N,N))
    F = np . zeros ( N )
    # Assembly
    A [0 , 0 ] =-2*turb
    A [0 , 1 ] = turb
    F [ 0 ] = 0 - leftbc*turb/h**2

    for i in range (1 , N - 1 ) :
        A [i , i - 1 ] = turb
        A [i , i ] = -2*turb
        A [i , i + 1 ] = turb
        F [ i ] = 0
    A [ N -1 , N - 2 ] = turb
    A [ N -1 , N - 1 ] = -2*turb

    for i in range(N):
        B[i,i] =  velocity(x[i])
        try: B[i,i+1] = velocity_p(x[i])*h - velocity(x[0])
        except: pass
    F [ N - 1 ] = - rightbc*(turb/h**2 +velocity(x[-1])/h)
    A = (1/h**2)*A
    B = (1/h)*B
    print(F)
    print(B)
    y_h_int = np . linalg . solve ((A - B), F )
    y_h = np . zeros ( N + 2 )
    y_h [ 0 ] = leftbc
    y_h [ 1 : N + 1 ] = y_h_int
    y_h [ - 1 ] = rightbc

    return x , y_h , h



def bvp_solve(velocity_f, velocity_fp, leftbc, rightbc, N):
    A = assemble_A(velocity_f, velocity_fp, N)
    F = assemble_F(leftbc, rightbc, N)

    print("v: ", velocity_f(300), velocity(699))
    print("A:", A)
    print("F", F)

    return np.linalg.solve(A, F)



def assemble_A(velocity_f, velocity_fp, N):
    xs, h = np.linspace(0, 1000, N, retstep=True)
    h2 = h**2

    A = np.zeros((N, N))
    A[0,0] = 1
    A[N-1, N-1] = 1
    for i in range(1, N-1):
        vel = velocity_f(xs[i])*h
        velp = velocity_fp(xs[i])

        A[i, i-1] = (turb / h2) + (vel / (2*h))
        A[i, i] =  (-2*turb / h2) - velp
        A[i, i+1] = (turb / h2) - (vel / (2*h))

    return A



def assemble_F(leftbc, rightbc, N):
    B = np.zeros(N)
    B[0] = leftbc
    B[N-1] = rightbc
    return B



#def show_ODE_solution():
#    x1, y_h1, h1 = ComputeNumericalSolution(100, -1.1, 1.0) # Temperature
#    x2, y_h2, h2 = ComputeNumericalSolution(7, 34.1, 35) # Salinity
#    fig, ax = plt.subplots(1,2)
#    ax[0].plot(x1,y_h1, color = 'red', label = f'Temperature h={h1}')
#    ax[1].plot(x2, y_h2, color = 'pink', label = f'Salinity (PSU) h={h2}')
#    fig.legend()
#    plt.show()




if __name__ == "__main__":
    main()
