"""Code for problem 4.

Numerical solution to the linearized Korteweg-De Vries equation solved on x [-1,1] and t [0,T].
The implementation using the Euler method is shown in the bottom of the file and 
is only for visualization purposes (numerically unstable).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags,identity
from scipy.sparse.linalg import spsolve
import numpy.linalg as la
from utilities import e_l, plot_order
from plotting_utilities import plot3d_sol_time

def analytical_solution(x,t):
    """Analytical solution to the problem."""
    return np.sin(np.pi*(x-t))

def theta_method_kdv(M,N,T,theta,init):
    """Solve the KdV equation.
    
    M, N: measure for number of gridpoints along the x- and t-dirction
    T: end time, run the calculations until t=T.
    theta: parameter to choose method, 0 for forward Euler and 1/2 for Crank-Nicolson
    init: function setting the initial condition u(x,0)
    """
    U = np.zeros((N+1,M+1))
    U[0,:] = init(np.linspace(-1,1,M+1))

    h = 2/M
    k = T/N

    c = 1 + np.pi**2
    a = 1/(8*h**3)
    b = c/(2*h) - 3/(8*h**3)

    # Periodic BC.
    down_corner_b = np.concatenate((np.array([0,b]), np.zeros(M-1)))
    down_corner_a = np.concatenate((np.array([0,a,a,a]), np.zeros(M-3)))
    
    data = np.array([-down_corner_b,-down_corner_a, np.full(M+1,a), np.full(M+1,b), 
                np.full(M+1,-b), np.full(M+1,-a), np.flip(down_corner_a), np.flip(down_corner_b)])
    diags = np.array([-(M-1),-(M-3),-3,-1,1,3,M-3,M-1])
    Q = spdiags(data, diags, M+1, M+1)
    
    lhs = identity(M+1) - theta*k*Q
    matrix = identity(M+1) + (1-theta)*k*Q
    for n in range(N):
        rhs = matrix @ U[n,:]
        U[n+1,:] = spsolve(lhs,rhs)
    return U

def disc_convergence_plot(M,theta,init,savename=False):
    """Plot relative l_2 norm at t=1 while the discretization number M increases expnentially."""
    T = 1
    N = 1000
    t = np.linspace(0,T,N+1)
    disc_err = np.zeros(len(M))
    for i, m in enumerate(M):
        x = np.linspace(-1,1,m+1)
        u = analytical_solution(x,T)
        U = theta_method_kdv(m,N,T,theta,init)
        disc_err[i] = e_l(U[-1,:],u)

    Ndof = M*N
    plt.plot(Ndof,disc_err,label=r"$e^r_{\ell}$ CN",color="red",marker='o')
    plot_order(Ndof,disc_err[0],2,r"$\mathcal{O}(N_{dof}^{-2})$","red")
    
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.xlabel(r"$M\cdot N$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename + ".pdf")
    plt.show()

def plot_l2_norm(M,N,T,theta,init,skip,savename=False):
    """Plot l2 norm of the numerical solution as a function of time.
    
    M, N: measure for number of gridpoints along the x- and t-dirction
    T: end time, run the calculations until t=T.
    theta: parameter to choose method, 0 for forward Euler and 1/2 for Crank-Nicolson
    init: function setting the initial condition u(x,0)
    skip: calculate and plot the norm of the solution at every 'skip' time step.
    """
    t_skipped = np.linspace(0, T, N+1)[::skip]
    U = theta_method_kdv(M,N,T,theta,init)
    l2_norm = np.zeros(len(t_skipped))
    l2_norm[:] = la.norm(U[::skip,:],axis=1)/np.sqrt(M+1)

    plt.plot(t_skipped, l2_norm, label=r"$||\mathbf{U}||_{\ell_2}$", color='red')
    plt.ylim(0.705,0.709)
    plt.ylabel(r"$||\cdot||_{\ell_2}$")
    plt.xlabel("$t$")
    plt.legend()
    if savename:
        plt.savefig(savename + ".pdf")
    plt.show()

initial_sine = lambda x : np.sin(np.pi*x)
initial_sine_2 = lambda x : np.sin(2*np.pi*x)  # Our own choice of initial condition (only used for 4c).

# ---| Plot solution |--- #
M=50; N=50; T=1
x = np.linspace(-1,1,M+1)
t = np.linspace(0,T,N+1)
U = theta_method_kdv(M,N,T,1/2,initial_sine)
#plot3d_sol_time(U,x,t,-100,10,analytical_solution)

# ---| Convergence plot |---#
M = np.array([16,32,64,128,256,512,850])
#disc_convergence_plot(M,1/2,initial_sine)

# ---| Plot l2 norm |---#
M = 800; N=1000; T=10
#plot_l2_norm(M,N,T,1/2,initial_sine,5)
#plot_l2_norm(M,N,T,1/2,initial_sine_2,5)


# ---| Euler method (only for visualization) |--- #
#plot3d_sol(M,N,T,0,initial_sine)
#disc_convergence_plot(M,0,initial_sine)
#plot_l2_norm(M,N,T,0,initial_sine,5)  
#plot_l2_norm(M,N,T,0,initial_sine_2,5)
