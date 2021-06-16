"""Code for problem 2 c)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from integrators import RK4_step
def initial(x):
    return np.exp(-400*(x - 1/2)**2)

def num_sol(x, t, method):  
    """Solve the ODE \dot{v} = f(t,v) with a specified method."""
    M = len(x)-2
    N = len(t)-1
    h = x[1] - x[0]
    k = t[1] - t[0]
    
    v_list = np.zeros((N+1,M+2))
    
    # Solve for internal grid points.
    v_list[0,1:-1] = initial(x[1:-1])
    F = lambda t, v : np.concatenate(([-v[0]*v[1]], -v[1:-1]*(v[2:] - v[0:-2]), [v[-1]*v[-2]]))/(2*h)
    
    for i in range(N):
        v_list[i+1,1:-1] = method(k, t[i], v_list[i,1:-1], F)  
    
    # Add B.C.
    zeros = np.zeros_like(t)
    v_list[:,0] = v_list[:,-1] = zeros
    return v_list

def plot_tail(n, interval, sol, x, t):
    """Plot the n last iterations of the solution."""
    for i in range(1, n*interval, interval):
        plt.plot(x, sol[-i, :], label = f"$t = {t[-i]}$")
    
    plt.xlabel('$x$')
    plt.ylabel('$u(x,t)$')
    plt.legend()
    plt.show()

# Plot breaking point of solution.
M = 1000
N = 1000
T = 0.06
x = np.linspace(0, 1, M+2)
t = np.linspace(0, T, N+1)

sol = num_sol(x, t, RK4_step)

plot_tail(4, 50, sol, x, t)

# Solve with scipy RK45.
def scipy_solution(x,t0,t_bound):
    """Solve the ODE with scipy's integrated function RK45 and plot the solution at t_bound."""
    
    h = x[1] - x[0]
    F = lambda t, v : np.concatenate(([-v[0]*v[1]], -v[1:-1]*(v[2:] - v[0:-2]), [v[-1]*v[-2]]))/(2*h)
    
    scipySol = RK45(F, t0, initial(x[1:-1]), t_bound)

    while scipySol.status != "finished":
        scipySol.step()

    y = scipySol.y
    zeros = np.zeros(1)
    y = np.hstack((zeros, y))
    y = np.hstack((y, zeros))
    plt.plot(x, y, label = f"R45 t = {t_bound}", linestyle = "dotted")
    print(scipySol.t)
    print(scipySol.status)

    plt.legend()
    plt.show()
    
#scipy_solution(x, 0, t[-1]) 
