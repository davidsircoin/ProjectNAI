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
    xs, h = np.linspace(0, 1000, N, retstep=True)

    velocities = list(map(velocity, xs))
    accelerations = list(map(velocity_p, xs))

    _, ys_temperature_v1, _ = solve_ODE_v1(velocity, velocity_p, N-2, t_0, t_1000) # Temperature
    _, ys_salinity_v1, _ = solve_ODE_v1(velocity, velocity_p, N-2, psu_0, psu_1000) # Salinity

    ys_temperature_v2 = solve_ODE_v2(velocities, accelerations, N, h, t_0, t_1000)
    ys_salinity_v2 = solve_ODE_v2(velocities, accelerations, N, h, psu_0, psu_1000)

    ys_temperature_v3 = solve_ODE_v3(velocities, accelerations, N, h, t_0, t_1000)
    ys_salinity_v3 = solve_ODE_v3(velocities, accelerations, N, h, psu_0, psu_1000)


    fig, ax = plt.subplots(4,2)
    ax[0,0].plot(xs, velocities, label=f'v')
    ax[0,0].scatter(distance_from_plume, measured_velocity)
    ax[0,1].plot(xs, accelerations, label=f'v\'')
    ax[1,0].plot(xs, ys_temperature_v1, label = f'Temperature N={N} v1', color = "green")
    ax[1,1].plot(xs, ys_salinity_v1, label = f'Salinity (PSU) N={N} v1', color = "blue")
    ax[2,0].plot(xs, ys_temperature_v2, label = f'Temperature N={N} v2', color = "purple")
    ax[2,1].plot(xs, ys_salinity_v2, label = f'Salinity (PSU) N={N} v2', color = "orange")
    ax[3,0].plot(xs, ys_temperature_v3, label = f'Temperature N={N} v3', color = 'red',)
    ax[3,1].plot(xs, ys_salinity_v3, label = f'Salinity (PSU) N={N} v3', color ="black")

    fig.legend()
    plt.show()

    fig, ax = plt.subplots(2)
    ys_density_D = density_approximation(ys_temperature_D, ys_salinity_D, xs)
    ys_density_E = density_approximation(ys_temperature_E, ys_salinity_E, xs)

    print(ys_density_D)

    fig, ax = plt.subplots(2)
    ax[0].plot(xs, ys_density_D, label = f'Density N={N}')
    ax[1].plot(xs, ys_density_E, label = f'Density N={N}')
    fig.legend()
    plt.show()

    
# Numerical solver of ODE. Still need to check if the entries are completely correct.
def solve_ODE_v1(velocity, velocity_p, N , leftbc , rightbc ):
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



def solve_ODE_v2(velocities, accelerations, N, h, leftbc, rightbc):
    A = assemble_A(velocities, accelerations, N, h)
    F = assemble_F(leftbc, rightbc, N)
    return np.linalg.solve(A, F)



def solve_ODE_v3(velocities, velocity_ps, N, h, leftbc, rightbc):
    A = assembly_of_A(N, h, turb, velocities, velocity_ps)
    F = assembly_of_F(N, h, leftbc, rightbc)
    return np.linalg.solve(A, F)



def assemble_A(velocities, accelerations, N, h):
    h2 = h**2

    A = np.zeros((N, N))
    A[0,0] = 1
    A[N-1, N-1] = 1
    for i in range(1, N-1):
        #vel = velocity_f(xs[i])*h
        #velp = velocity_fp(xs[i])*h

        #A[i, i-1] = (turb / h2) + (vel / (2*h))
        #A[i, i] =  (-2*turb / h2) - velp
        #A[i, i+1] = (turb / h2) - (vel / (2*h))

        A [i , i - 1 ] = (1/2)*(2*turb + h*velocities[i])
        A [i , i ] =  -2*turb + h2*accelerations[i]
        A [i , i + 1 ] = (1/2)*(2*turb - h*velocities[i])

    return (1/h2)*A



def assemble_F(leftbc, rightbc, N):
    B = np.zeros(N)
    B[0] = leftbc
    B[N-1] = rightbc
    return B



def density_approximation(temp, sal, xs):
    # Parameters
    rho_ref = 1027.51 #kg/m³            in situ density
    alpha_lin = 3.733*10**(-5) #◦C      Thermal expansion coefficient
    beta_lin = 7.843*10**(-4) #PSU      Salinity contraction coefficient
    T_ref = -1 # ◦C                     Reference temperature
    S_ref = 34.2 #PSU                   Reference salinity

    density = []
    def equation(x):
        return rho_ref*(1 - alpha_lin*(temp[x] - T_ref) + beta_lin*(sal[x] - S_ref))
    
    for x in range(len(xs)):
        print(equation(x))
        density.append(equation(x))
        
    return np.array(density)
        


def assembly_of_A(N, h, turb, v, vp): # Finite Difference Scheme for second order d
    h2 = h**2
    A = np . zeros (( N , N ) )

    A [0 , 0 ] = -2*turb + h2*vp[0]
    A [0 , 1 ] = (1/2)*(2*turb - h*v[0])

    for i in range (1 , N - 1 ) :
        A [i , i - 1 ] = (1/2)*(2*turb + h*v[i])
        A [i , i ] =  -2*turb + h2*vp[i]
        A [i , i + 1 ] = (1/2)*(2*turb - h*v[i])


    A [ N - 1 , N - 2 ] = (1/2)*(2*turb + h*v[N-1])
    A [ N - 1 , N - 1 ] = -2*turb + h2*vp[N-1]

    A = (1/h**2)*A
    return A

def assembly_of_F(N, h, leftbc, rightbc):
    h2 = h**2
    F = np.zeros(N)
    F[0] = 0 - leftbc/h2
    for i in range(1, N-1):
        F[i] = 0
    F[N-1] = 0 - rightbc/h2
    return F


if __name__ == "__main__":
    main()
