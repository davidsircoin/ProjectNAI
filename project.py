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



# Numerical solver of ODE. Still need to check if the entries are completely correct.
def ComputeNumericalSolution(N , leftbc , rightbc ):
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

def show_ODE_solution():
    x1, y_h1, h1 = ComputeNumericalSolution(100, -1.1, 1.0) # Temperature
    x2, y_h2, h2 = ComputeNumericalSolution(7, 34.1, 35) # Salinity
    fig, ax = plt.subplots(1,2)
    ax[0].plot(x1,y_h1, color = 'red', label = f'Temperature h={h1}')
    ax[1].plot(x2, y_h2, color = 'pink', label = f'Salinity (PSU) h={h2}')
    fig.legend()
    plt.show()

