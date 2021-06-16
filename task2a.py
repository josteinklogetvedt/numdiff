"""Code for problem 2 a)."""
from scipy.sparse import spdiags, identity
from scipy.sparse.linalg import spsolve
import numpy as np
import pickle # To save the reference solution.
from utilities import *
from plotting_utilities import plot3d_sol_time
initial = (lambda x: 2*np.pi*x - np.sin(2*np.pi*x))

def calc_sol(x, t, order, theta, plot = False):
    """ 
    order = 1: Use first oder disc. on BC.
    order = 2: Use second order disc. on BC.
    theta = 1: Backward Euler.
    theta = 1/2: Trapezoidal rule (CN).
    """
    M = len(x)-1
    N = len(t)-1
    
    # Construct Q.
    data = np.array([np.full(M+1, 1), np.full(M+1, -2), np.full(M+1, 1)])
    diags = np.array([-1, 0, 1])
    Q = spdiags(data, diags, M+1, M+1,format='lil')
    if order == 1:
        Q[0,:3] = [-1,0,1]
        Q[-1,-3:] = [1,0,-1]
    elif order == 2:
        Q[0, 1] = Q[-1, -2] =  2
    Q = Q.tocsr()
    
    sol = np.zeros((N+1,M+1))  
    sol[0,:] = initial(x)
    
    k = t[1]-t[0]
    h = x[1]-x[0]
    r = k/h**2
    
    lhs = identity(M+1) - theta*r*Q
    b = identity(M+1) + (1-theta)*r*Q
    for n in range(N):
        rhs = b @ sol[n,:]
        sol[n+1,:] = spsolve(lhs,rhs)
    
    return sol

def save_ref_sol(Mstar, Nstar, order, theta, filename):
    """Save the reference solution to file."""
    T = 0.2
    x = np.linspace(0,1,Mstar+1)
    t = np.linspace(0,T,Nstar+1)
    ref_sol = calc_sol(x, t, order, theta) 
    with open(filename, 'wb') as fi:
        pickle.dump(ref_sol, fi)

def calc_error(M, N, filename, savename=False): 
    """Calculate the relative error with the reference solution.

    Input:
    M: List or scalar depending on the type of refinement.
    N: List or scalar depending on the type of refinement.
    filename: Name of the file where the reference solution has been saved. 
    """
    ref_sol = None
    Mstar = Nstar = 1000   # Change values here if you change it in the file.
    with open(filename, 'rb') as fi: 
        ref_sol = pickle.load(fi)

    if np.ndim(N)==0:
        N = np.ones_like(M)*N
    elif np.ndim(M)==0:
        M = np.ones_like(N)*M
    else:
        assert(len(N)==len(M))
    modulus = Mstar % M  # Controls that M are divisible by Mstar. 
    # This does not apply to N because we look at error in T which is similar for both sol and ref_sol
    if len(modulus[np.nonzero(modulus)]) != 0:
        print('Wrong M values.')
        return 1
        
    disc_err_first = np.zeros(len(M))
    disc_err_second = np.zeros(len(M))

    T = 0.2
    for i in range(len(M)):
        x = np.linspace(0,1,M[i]+1)
        t = np.linspace(0,T,N[i]+1)
        sol_1 = calc_sol(x, t, 2, 1)
        sol_2 = calc_sol(x, t, 2, 1/2)
        u_star = ref_sol[-1,0::(Mstar//M[i])]
       
        disc_err_first[i] = e_l(sol_1[-1,:], u_star)
        disc_err_second[i] = e_l(sol_2[-1,:], u_star)      
    
    MN = M*N

    plt.plot(MN, disc_err_first, label = r"$e^r_{\ell}$ (BE)",color='red',marker='o')
    plt.plot(MN, disc_err_second, label = r"$e^r_{\ell}$ (CN)",color='blue',marker='o')

    # These need to be changed manually.
    plot_order(MN, disc_err_first[0], 1/2, label = r"$\mathcal{O}(N_{dof}^{-1/2})$", color = 'red')
    plot_order(MN, disc_err_second[0], 1, label = r"$\mathcal{O}(N_{dof}^{-1})}$", color='blue')

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r'$M \cdot N$')
    plt.ylabel(r"Error $e^r_{\ell}$")
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename+'.pdf')
    plt.show()

def compare_discr(M, N, filename, savename=False):
    """Compare the first and second order discretizations of the BCs by making a convergence plot."""
    ref_sol = None
    with open(filename, 'rb') as fi: 
        ref_sol = pickle.load(fi)

    disc_err_first = np.zeros(len(M))
    disc_err_second = np.zeros(len(M))
    T = 0.2
    t = np.linspace(0,T,N+1)
    for i in range(len(M)):
        x = np.linspace(0,1,M[i]+1)
        sol_1 = calc_sol(x, t, 1, 1/2)
        sol_2 = calc_sol(x, t, 2, 1/2)
        u_star = ref_sol[-1,0::(Mstar//M[i])]
       
        disc_err_first[i] = e_l(sol_1[-1,:], u_star)
        disc_err_second[i] = e_l(sol_2[-1,:], u_star)      
    
    MN = M*N

    plt.plot(MN, disc_err_first, label = r"$e^r_{\ell}$ (1st order)",color='red',marker='o')
    plt.plot(MN, disc_err_second, label = r"$e^r_{\ell}$ (2nd order)",color='blue',marker='o')

    # These need to be changed manually.
    plot_order(MN, disc_err_first[0], 1, label = r"$\mathcal{O}(N_{dof}^{-1})$", color = 'red')
    plot_order(MN, disc_err_second[0], 2, label = r"$\mathcal{O}(N_{dof}^{-2})}$", color='blue')

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r'$M \cdot N$')
    plt.ylabel(r"Error $e^r_{\ell}$")
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename+'.pdf')
    plt.show()


# ====| Run code below. |==== #

Mstar = Nstar = 1000
filename = 'ref_sol.pk'
# We use 2 order disc. of BC and CN for the reference solution.
#save_ref_sol(Mstar, Nstar, 2, 1/2, filename) # Only needs to be run once, or if you change Mstar

# ---| Plot solution. |--- #
M=50; N=50; T=0.2
x = np.linspace(0,1,M+1)
t = np.linspace(0,T,N+1)
sol = calc_sol(x,t,2,1/2)
#plot3d_sol_time(sol,x,t,-130,15)

# ---| Compare firts and second order disc. of BCs. |--- # 
N = 1000
M = np.array([8,10,20,25,40,50,100,125,200,250,500])
#compare_discr(M,N,filename)

# ---| h-refinement. |--- # 
N = 1000
M = np.array([8,10,20,25,40,50,100,125,200,250,500])
#calc_error(M, N, filename)

# ---| k-refinement. |--- #
M = 1000
#N = np.array([8,10,20,25,40,50,100,125,200,250,500])
N = np.array([4,8,16,32,64,128,256])  #does not have to be divisible by Nstar
#calc_error(M,N,filename)  

# ---| h=ck -refinement, here both M and N increases. |--- #
M = np.array([8,10,20,25,40,50,100,125,200,250,500])
N = np.array([8,10,20,25,40,50,100,125,200,250,500])
#calc_error(M,N,filename)  #gives BE; Ndof^(-1/2) and CN; Ndof^(-1)

### Delete this?? - don't need such refinement
# r -refinement, keeping r fixed, r=40=M^2/N. Difficult to choose appropriate values
M = np.array([20,25,40,50,100,125,200])
N = np.array([10,16,40,63,250,391,1000]) 
#calc_error(M,N,filename) 
