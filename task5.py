"""Code for problem 5."""
from scipy.sparse import spdiags # Make sparse matrices with scipy.
from scipy.sparse.linalg import spsolve 
import numpy as np
import numpy.linalg as la
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt
from utilities import *

class FEM_sol:
     """A class which represents the FEM solution.
     This is necessary in order to incorporate the coefficients
     in the solution function u_h(x).
     """
     def __init__(self, coeff, x_grid):
         self.coeff = coeff
         self.x_grid = x_grid

     def uh(self, x): # An alternative to this function is to interpolate the ui values linearly.
         index = 0
         for i in range(len(self.x_grid) - 1): # Probably more effective with a bisection method.
             if x <= self.x_grid[i + 1]:
                 break
             else:
                 index += 1

         left = self.coeff[index] * (self.x_grid[index + 1] - x)/(self.x_grid[index + 1] - self.x_grid[index])
         right = self.coeff[index + 1] * (x - self.x_grid[index])/(self.x_grid[index + 1] - self.x_grid[index]) 
         return left + right
        
def FEM_solver_Dirichlet(BC, f, x):
    """General FEM solver with Dirichlet BC."""
    N = len(x)
    # Construct A
    A = np.zeros((N,N))
    for i in range(N - 1):
        A[i:(i + 2), i:(i + 2)] +=  1/(x[i+1] - x[i]) * np.array([[1, -1], [-1, 1]])

    rhs = np.zeros(N)
    for i in range(N - 1):
        # Functions to do Gaussian quadrature on:
        v1 = lambda y: (x[i + 1] - y) * f(y)
        v2 = lambda z: (z -x[i]) * f(z)
        rhs[i:(i+2)] += 1/(x[i + 1] - x[i]) * np.array([gauss(5, v1, x[i], x[i +1]), gauss(5, v2, x[i], x[i + 1])])

    BC_vec = np.zeros(N)
    BC_vec[0] = BC[0]
    BC_vec[-1] = BC[1]

    rhs = rhs - A @ BC_vec

    # Implementing BC.
    A = A[1:-1,1:-1]
    rhs = rhs[1:-1]

    # Solve system.
    u = la.solve(A, rhs) 
    u = np.hstack((np.array(BC[0]), u))
    u = np.hstack((u, np.array(BC[1])))

    return u

def UFEM(N_list, BC, f, anal_sol, x_interval, savename = False):
    """Conduct FEM with uniform refinement in 1D."""
    err_list = []

    for N in N_list:
        x = np.linspace(x_interval[0], x_interval[1], N)
        U = FEM_solver_Dirichlet(BC, f, x)
        num_sol = FEM_sol(U, x)
    
        # Calculating error.
        diff = lambda x : num_sol.uh(x) - anal_sol(x)
        err = cont_L2_norm(diff, x[0], x[-1])
        err_list.append(err)

    plt.plot(N_list, err_list, marker = 'o', label = "$||u - u_h||_{L_2}$")
    plot_order(np.array(N_list), err_list[0], 2, "$O(h^{2})$", color = "red")
    plot_order(np.array(N_list), err_list[0], 1, "$O(h)$", color = "green")

    plt.title("UFEM")
    plt.xlabel("$M$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename + ".pdf")
    plt.show()

# AFEM.
def calc_cell_errors(U, u, x):
    """Calculates the error for each cell by taking the L_2 norm. Used in AFEM."""
    n = len(x) - 1 # Number of cells.
    cell_errors = np.zeros(n)

    for i in range(n):
        diff = lambda x: U(x) - u(x)
        cell_errors[i] = cont_L2_norm(diff, x[i], x[i + 1])

    return cell_errors

def AFEM(N0, steps, alpha, type, f, anal_sol, x_interval, savename = False):
    """Conduct FEM with adaptive refinement in 1D steps times."""
    err_list = []
    N_list = []
    x = np.linspace(x_interval[0], x_interval[1], N0)

    for i in range(steps):
        U = FEM_solver_Dirichlet(BC, f, x)
        U_interp = interp1d(x, U, kind = 'linear')
        num_sol = FEM_sol(U, x)

        cell_errors = calc_cell_errors(num_sol.uh, anal_sol, x)
    
        # Calculating error.
        diff = lambda x : num_sol.uh(x) - anal_sol(x)
        err = cont_L2_norm(diff, x[0], x[-1])
        
        N_list.append(len(x) - 1)
        err_list.append(err)

        if type == 'avg':
            err = 1/N_list[i] * err
        elif type == 'max':
            err = max(cell_errors)
        else:
            raise Exception("Invalid type.")
        
        x = list(x)
        k = 0 # index for x in case we insert points.
        for j in range(len(cell_errors)):
            if cell_errors[j] > alpha * err:
                x.insert(k+1, x[k] + 0.5 * (x[k+1] - x[k]))
                k += 1
            k += 1
        x = np.array(x)

    plt.plot(N_list, err_list, marker = 'o', label = "$||u - u_h||_{L_2}$")
    plot_order(np.array(N_list), err_list[0], 2, "$O(h^{2})$", color = "red")

    plt.title(str(steps) + '-step AFEM with ' + type + '-error and ' + r'$\alpha =$' + str(alpha))
    plt.xlabel("$M$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename + ".pdf")
    plt.show()

def plot_and_save(anal_sol, f, BC, N_list, N0, steps, task_name):
    """General method for performing UFEM and AFEM and saving the resulting plots for all tasks."""
    # UFEM
    UFEM(N_list, BC, f, anal_sol, x_interval, task_name + "_UFEM")

    # Average AFEM
    alpha = 1
    AFEM(N0, steps, alpha, 'avg', f, anal_sol, x_interval, task_name + "_avgAFEM")

    # Maximum AFEM 
    alpha = 0.7
    AFEM(N0, steps, alpha, 'max', f, anal_sol, x_interval, task_name + "_maxAFEM")



# ====| Run code below. Uncomment the FEM method you want to run.|==== #

## b)
anal_sol = lambda x: x**2
f = lambda x: -2
x_interval = [0,1]
BC = [0, 1] 
N_list = [2**i for i in range(3, 12)]
# For AFEM:
steps = 7
N0 = 20
#plot_and_save(anal_sol, f, BC, N_list, N0, steps, "5b")


## c)
anal_sol = lambda x: np.exp(-100 * x**2)
f = lambda x: - (40000*x**2 - 200) * np.exp(-100 * x**2)
x_interval = [-1, 1]
BC = [np.exp(-100), np.exp(-100)]
N_list = [2**i for i in range(3, 12)]
# For AFEM:
steps = 7
N0 = 20
#plot_and_save(anal_sol, f, BC, N_list, N0, steps, "5c")


## d) 
anal_sol = lambda x: np.exp(-1000 * x**2)
f = lambda x: - (4000000 * x**2 - 2000) * np.exp(-1000 * x**2)
x_interval = [-1, 1]
BC = [np.exp(-1000), np.exp(-1000)]
N_list = [2**i for i in range(3, 12)]
# For AFEM:
steps = 7
N0 = 20
#plot_and_save(anal_sol, f, BC, N_list, N0, steps, "5d")


## e)
anal_sol = lambda x: x**(2/3) 
f = lambda x:  2/9 * x**(-4/3)
x_interval = [0, 1]
BC = [0, 1]
N_list = [2**i for i in range(3, 12)]
# For AFEM:
steps = 7
N0 = 20
#plot_and_save(anal_sol, f, BC, N_list, N0, steps, "5e")
