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
velocity_interpolation= sp.interpolate.CubicSpline(distance_from_plume, measured_velocity)

def main():
    N = 100
    xs, h = np.linspace(0, 1000, N, retstep=True)

    # Descretize the velocity interpolation
    velocities = list(map(velocity_interpolation, xs))

    # Aproximate the derivative of the interpolated velocity function wrt x
    velocities_p = approximate_derivative(velocity_interpolation, N)

    ys_temperature = [t_0, *solve_ODE(velocities, velocities_p, N-2, h, t_0, t_1000), t_1000]
    ys_salinity = [psu_0, *solve_ODE(velocities, velocities_p, N-2, h, psu_0, psu_1000), psu_1000]
    ys_density = density_approximation(ys_temperature, ys_salinity, xs)

    fig, ax = plt.subplots(3,2)
    fig.delaxes(ax[2,1])
    ax[0,0].plot(xs, velocities, label=f'v')
    ax[0,0].scatter(distance_from_plume, measured_velocity)
    ax[0,1].plot(xs, velocities_p, label=f'v\'')
    ax[1,0].plot(xs, ys_temperature, label = f'Temperature N={N}', color = 'red')
    ax[1,1].plot(xs, ys_salinity, label = f'Salinity (PSU) N={N}', color = 'blue')
    ax[2,0].plot(xs, ys_density, label = f'Density N={N}', color="pink")
    fig.legend(loc = 'lower right')
    plt.show()


def solve_ODE(velocities, velocities_p, N, h, leftbc, rightbc):
    A = assembly_of_A(N, h, velocities, velocities_p)
    F = assembly_of_F(N, h, velocities, leftbc, rightbc)
    return np.linalg.solve(A, F)


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
        density.append(equation(x))
        
    return np.array(density)
        

def assembly_of_A(N, h, v, vp): # Finite Difference Scheme for second order d
    h2 = h**2
    A = np . zeros (( N , N ) )

    A [0 , 0 ] = -2*turb - h2*vp[0]
    A [0 , 1 ] = (1/2)*(2*turb - h*v[0])

    for i in range (1 , N - 1 ) :
        A [i , i - 1 ] = (1/2)*(2*turb + h*v[i])
        A [i , i ] =  -2*turb - h2*vp[i]
        A [i , i + 1 ] = (1/2)*(2*turb - h*v[i])


    A [ N - 1 , N - 2 ] = (1/2)*(2*turb + h*v[N-1])
    A [ N - 1 , N - 1 ] = -2*turb - h2*vp[N-1]

    A = (1/h2)*A
    return A


def assembly_of_F(N, h, velocity, leftbc, rightbc):
    h2 = h**2
    F = np.zeros(N)
    F[0] = 0 - leftbc*(1/(2*h2))*(2*turb + h * velocity[1])
    for i in range(1, N-1):
        F[i] = 0
    F[N-1] = 0 - rightbc*(1/(2*h2))*(2*turb - h * velocity[N-2])
    return F


def approximate_derivative(vel, N):
    dx = 1/N
    x = np.linspace(0, 1000, N)
    xpdx = np.array([n + dx for n in x])
    return (vel(xpdx) - vel(x))/dx

 
if __name__ == "__main__":
    main()
