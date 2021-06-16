"""Code for problem 2 d) and e) in part 2 (Sine-Gordon)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from integrators import RK2_step, RK3_step, RK4_step, RKN12_step, RKN34_step
from utilities import e_l, plot_order
from plotting_utilities import plot3d_sol_time

c = 0.5  
def analytical_solution(x,t):
    """Analytical solution, choosing c=1/2 and the minus-sign as solution."""
    b = (x - c*t)/np.sqrt(1 - c**2)
    return 4*np.arctan(np.exp(-b))

def num_solution(x, t, method):
    """Numerical solution using Runge-Kutta or Runge-Kutta-Nystrøm schemes, solved with 'method' step.
    
    Input parameters:
    x: x-grid
    t: time-grid
    method: callable step function for RK or RKN loop. 
    """
    assert(callable(method))
    
    f_1 = lambda t : analytical_solution(x[0],t)
    f_2 = lambda t : analytical_solution(x[-1],t)
    u_0 = lambda x : analytical_solution(x,0)
    u_1 = lambda x : 4*c*np.exp(-x/np.sqrt(1 - c**2)) / (( np.sqrt(1 - c**2) * (np.exp(-2*x/np.sqrt(1 - c**2)) + 1)))

    N = len(t)-1
    M = len(x)-2
    k = t[1] - t[0]
    h = x[1] - x[0]

    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])/h**2
    diags = np.array([-1, 0, 1])
    Ah = spdiags(data, diags, M, M)
    
    g = lambda t, v : np.concatenate(([-np.sin(v[0])+f_1(t)/h**2],-np.sin(v[1:-1]),[-np.sin(v[-1])+f_2(t)/h**2]))
    F = None
    if (method==RK2_step) or (method==RK3_step) or (method==RK4_step):
        F = lambda t, y : np.array([y[1], Ah.dot(y[0]) + g(t,y[0])])
    elif (method==RKN12_step) or (method==RKN34_step):
        F = lambda t, v : Ah.dot(v) + g(t,v)
    assert(callable(F))
    
    Y = np.zeros((N+1,2,M))
    Y[0, 0, :] = u_0(x[1:-1])
    Y[0, 1, :] = u_1(x[1:-1])
    
    for i in range(N):
        Y[i+1,:,:] = method(k, t[i], Y[i,:,:], F)
    
    Usol = Y[:,0,:]
    
    # Insert B.C.
    Usol = np.insert(Usol,0,f_1(t),axis=1)
    Usol = np.column_stack((Usol,f_2(t)))

    return Usol

def refinement(M,N,solvers,colors,labels,savename=False):
    """Perform h- or t-refinement and plots the result."""
    T = 5
    Ndof = M*N
    err_start = np.zeros(len(solvers))
    
    if np.ndim(M) == 0: # t-refinement.
        N_ref = 10000
        x = np.linspace(-5,5,M+2)
        t_ref = np.linspace(0,T,N_ref+1)
        assert(N[-1]<N_ref)
        for i, method in enumerate(solvers):
            U_ref = num_solution(x,t_ref,method)
            err = np.zeros(len(N))
            for j, n in enumerate(N):
                t = np.linspace(0,T,n+1)
                U = num_solution(x,t,method)
                err[j] = e_l(U[-1,:],U_ref[-1,:])
            plt.plot(Ndof, err, label=labels[i], color=colors[i], marker = 'o')
            err_start[i] = err[0]
        
    elif np.ndim(N) == 0:
        t = np.linspace(0,T,N+1)
        for i, method in enumerate(solvers):
            err = np.zeros(len(M))
            for j, m in enumerate(M):
                x = np.linspace(-5,5,m+2)
                U = num_solution(x,t,method)
                u = analytical_solution(x,t[-1])
                err[j] = e_l(U[-1,:],u)
            plt.plot(Ndof, err, label=labels[i], color=colors[i], marker = 'o')
            err_start[i] = err[0]
    
    # Change these manually.
    plot_order(Ndof, err_start[0], 2, label=r"$\mathcal{O}(N_{dof}^{-2})$", color=colors[0])
    #plot_order(Ndof, err_start[1], 3, label=r"$\mathcal{O}(N_{dof}^{-3})$", color=colors[1])
    #plot_order(Ndof, err_start[2], 4, label=r"$\mathcal{O}(N_{dof}^{-4})$", color=colors[2])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$M \cdot N$")
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename+".pdf")
    plt.show()


# ===| Run code below. |=== #

# Plot solution.
M = 20; N=20; T=5
x = np.linspace(-5,5,M+2)
t = np.linspace(0,T,N+1)
U = num_solution(x, t, RK4_step)
#plot3d_sol_time(U,x,t,-55,20,analytical_solution)

# RK h-refinement.
N = 1000
M = np.array([32,64,128,256,512])
solvers = [RK2_step,RK3_step,RK4_step]
colors = ['red','green', 'blue']
labels = [r'$e^r_{\ell}$ (RK2)', r'$e^r_{\ell}$ (RK3)',r'$e^r_{\ell}$ (RK4)']
#refinement(M,N,solvers,colors,labels)

# RK t-refinement.
M_ref = 400
N = np.array([1000,1500,2000,2500,3000,3500])
#refinement(M_ref,N,solvers,colors,labels)

# RKN h-refinement.
N = 15000  
M = np.array([32,64,128,256,512])
solvers = [RKN12_step,RKN34_step]
colors = ['red', 'blue']
labels = [r'$e^r_{\ell}$ (RKN12)', r'$e^r_{\ell}$ (RKN34)']
#refinement(M,N,solvers,colors,labels)

# RKN t-refinement.
M_ref = 400
N = np.array([2000,2500,3000,3500,4000,4500])
#refinement(M_ref,N,solvers,colors,labels)
