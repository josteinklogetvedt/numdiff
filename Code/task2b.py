"""Code for problem 2 b).

Solve u_t = u_xx on x[0,1], t[0,T] with reference to manufactured solution, with UMR with r-refinement.

Dirichlet BC u(0,t)=u(1,t)=0, and initial value f(x)=3*sin(2*pi*x)
First order method; Backward Euler
Second order; Crank Nicolson
"""
from utilities import *
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import numpy as np
from scipy.interpolate import interp1d 
from plotting_utilities import plot3d_sol_time
initial = (lambda x: 3*np.sin(2*np.pi*x))

def analytic_solution(x,t):
    return 3*np.exp(-4*(np.pi**2)*t)*np.sin(2*np.pi*x)

def theta_method(x, t, theta):
    """Theta-method.
    x: x-grid 
    t: t-grid  
    theta: Parameter to change how the method works. 
    
    Returns a new matrix U with the solution of the problem in the gridpoints. 
    """
    M = len(x)-2
    N = len(t)-1
    h = x[1] - x[0]
    k = t[1] - t[0]
    r = k/h**2
    
    # Insert boundaries. 
    U = np.zeros((N+1,M+2))
    U[0,:] = initial(x)
    U[:,0] = 0
    U[:,-1] = 0
    
    # Set up the matrices.
    data = np.array([np.full(M, (1-theta)*r), np.full(M, 1-2*(1-theta)*r), np.full(M, (1-theta)*r)])
    diags = np.array([-1, 0, 1])
    b = spdiags(data,diags,M,M,format='csr')

    data = np.array([np.full(M, -theta*r), np.full(M, 1+2*theta*r), np.full(M, -theta*r)])
    lhs = spdiags(data,diags,M,M,format='csr')

    for n in range(N):
        U[n+1,1:-1] = spsolve(lhs,b @ U[n,1:-1])

    return U

def plot_UMR(M,N,savename=False):
    """Calculate and plot refinement plots with L_2 and l_2 norm."""
    T = 0.2
    time_index = -1
    if np.ndim(N)==0:
        N = np.ones_like(M)*N
    elif np.ndim(M)==0:
        M = np.ones_like(N)*M
    else:
        assert(len(M)==len(N))
    disc_err_first = np.zeros(len(M))
    disc_err_second = np.zeros(len(M))
    cont_err_first = np.zeros(len(M))
    cont_err_second = np.zeros(len(M))
    for i in range(len(M)):
        x = np.linspace(0,1,M[i]+2) 
        t = np.linspace(0,T,N[i]+1)
        
        U_BE = theta_method(x,t,1)
        U_CN = theta_method(x,t,1/2)
        u = analytic_solution(x,t[time_index])
        
        disc_err_first[i] = e_l(U_BE[time_index,:],u)
        disc_err_second[i] = e_l(U_CN[time_index,:],u)

        analytic_solution_time = lambda x : analytic_solution(x,t[time_index])
        cont_err_first[i] = e_L(interp1d(x, U_BE[time_index,:], kind = 'cubic'), analytic_solution_time, x[0], x[-1])
        cont_err_second[i] = e_L(interp1d(x, U_CN[time_index,:], kind = 'cubic'), analytic_solution_time, x[0], x[-1])
    
    MN = M*N    
    # These need to be changed manually.
    plot_order(MN, cont_err_first[0], 2/3, label = r"$\mathcal{O}(N_{dof}^{-2/3})$", color = "lime")
    plot_order(MN, cont_err_second[0], 2/3, label = r"$\mathcal{O}(N_{dof}^{-2/3})$", color = "blue")

    plt.plot(MN,cont_err_first, label=r"$e^r_{L_2}$ (BE)", color='green',marker='o')
    plt.plot(MN,cont_err_second, label=r"$e^r_{L_2}$ (CN)",color='purple',marker='o')
    plt.plot(MN, disc_err_first, label=r"$e^r_{\ell}$ (BE)", color='red',marker='o',linestyle="--")
    plt.plot(MN, disc_err_second, label=r"$e^r_{\ell}$ (CN)",color='orange',marker='o',linestyle="--")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M \cdot N$')
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename+'.pdf')
    plt.show()


# ====| Run code below. |==== #

#Plot of analytical and numerical solution
M=20; N=20; T=0.2
x = np.linspace(0,1,M+2)
t = np.linspace(0,T,N+1)
U = theta_method(x,t,1/2)
#plot3d_sol_time(U,x,t,55,15,analytic_solution)


# ---| h-refinement. |--- # 
N = 1000
M = np.array([8,16,32,64,128,256]) 
#plot_UMR(M,N)  # h-refinement, both methods are second order but BE flattens out because of big error in t

# ---| k-refinement. |--- # 
M = 1000
N = np.array([8,16,32,64,128,256])
#plot_UMR(M,N) #t-refinement, BE går som O(h), CN som O(h^2)

# ---| k = h-refinement. Doubling k and h for each iteration. |--- # 
M = np.array([8,16,32,64,128,256])
N = np.array([8,16,32,64,128,256])
#plot_UMR(M,N)  #BE går som O(h^1/2), CN som O(h^1)

# r = k/h^2 constant-refinement. Here r=1024=k/h^2=M^2/N.
M = np.array([64,128,256,512,1024,2048])
N = np.array([4,16,64,256,1024,4096])
#plot_UMR(M,N)  #BE og CN går som O(h^2/3) (etterhvert i hvert fall)
