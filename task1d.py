"""Code for problem 1 d)."""
import numpy as np
from numpy.lib.function_base import interp
import numpy.linalg as la
import matplotlib.pyplot as plt
from utilities import *
from scipy.interpolate import interp1d 
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from matplotlib.cm import get_cmap
eps = 0.01
def f(x):
    """Right hand side of 1D Poisson equation."""
    return eps**(-2)*np.exp(-(1/eps)*(x - 0.5)**2)*(4*x**2 - 4*x + 1 - 2*eps)

def anal_solution(x):
    """Manufactured solution of the Poisson equation with Dirichlet BC."""
    return np.exp(-(1/eps)*(x-0.5)**2)

def num_sol_UMR(x,order): # order = 1 or 2.
    """First order numerical solution of the Possion equation with Dirichlet B.C.,
    given by the manufactured solution. Using a forward difference scheme. 

    The parameter 'order' can take values 1 or 2. 
    """
    M = len(x)-2
    h = np.diff(x)[0]
    assert(order == 1 or order == 2)

    # Construct Dirichlet boundary condition from manuactured solution.
    alpha = anal_solution(x[0])
    beta = anal_solution(x[-1])
    
    # Construct Ah. 
    data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])*1/h**2
    diags = np.array([-1, 0, 1])
    Ah = sp.diags(data, diags, shape = [M, M], format = "csc")
    
    if order == 1:
        f_vec = np.full(M,f(x[:-2]))
    else: # order = 2.
        f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0]- alpha/h**2
    f_vec[-1] = f_vec[-1] - beta/h**2

    # Solve linear system. 
    Usol = sla.spsolve(Ah, f_vec) 
    Usol = np.insert(Usol, 0, alpha)
    Usol = np.append(Usol,beta)
    
    return Usol

#----------UMR--------------
def plot_UMR_errors(save = False, barplot = False):
    """Convergence plot from UMR."""
    M_list = 2**np.arange(3,12,dtype=int)
    X = []
    U_2 = []
    cell_error_list = []
    tol_list = []

    e_1_disc = np.zeros(len(M_list))
    e_2_disc = np.zeros(len(M_list))
    e_1_cont = np.zeros(len(M_list))
    e_2_cont = np.zeros(len(M_list))


    for i, m in enumerate(M_list):
        x = np.linspace(0,1,m+2)
        X.append(x)
        u = anal_solution(x)
        first_order_num = num_sol_UMR(x,1)
        second_order_num = num_sol_UMR(x,2)
        U_2.append(second_order_num)
        cell_errors = calc_cell_errors(x, anal_solution, interp1d(x, U_2[-1], kind = 'cubic'))
        cell_error_list.append(cell_errors)
        tol_list.append(np.average(cell_errors))

        # Discrete norms. 
        e_1_disc[i] = e_l(first_order_num, u)
        e_2_disc[i] = e_l(second_order_num, u)
        
        # Continuous norms. 
        interpU_first = interp1d(x, first_order_num, kind = 'cubic')
        interpU_second = interp1d(x, second_order_num, kind = 'cubic')

        e_1_cont[i] = e_L(interpU_first, anal_solution, x[0], x[-1])
        e_2_cont[i] = e_L(interpU_second, anal_solution, x[0], x[-1])
        
    plt.plot(M_list,e_2_cont,label=r"$e^r_{L_2}$ second", color = "black",marker='o')
    plt.plot(M_list,e_1_cont,label=r"$e^r_{L_2}$ first", color = "green",marker='o')
    plt.plot(M_list,e_1_disc,label=r"$e^r_{\ell}$ first", color = "red",marker='o',linestyle = "--")
    plt.plot(M_list,e_2_disc,label=r"$e^r_{\ell}$ second", color = "blue",marker='o',linestyle = "--")
    plot_order(M_list, e_1_disc[0], 1, r"$\mathcal{O}(h)$", 'red')
    plot_order(M_list, e_2_disc[0], 2, r"$\mathcal{O}(h^2)$", 'blue')
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.xlabel("$M$")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("loglogtask1dUMR.pdf")
    plt.show()

    if barplot:
        plot_bar_error(X, U_2, 0, len(U_2), cell_error_list, tol_list)


def plot_UMR_solution(save = False):
    """Plot analytical solution and numerical solution using AMR."""
    M = 40
    x = np.linspace(0,1,M+2)
    x_an = np.linspace(0,1,500) # Plot the analytical solution on a smooth grid. 
    u = anal_solution(x_an) 
    first_order_num = num_sol_UMR(x,1)
    second_order_num = num_sol_UMR(x,2)
    plt.plot(x_an,u,label="Analytical", color = "blue")
    plt.plot(x, first_order_num, label = "First", linestyle = "dotted", color = "green")
    plt.plot(x, second_order_num, label = "Second", linestyle = "dotted", linewidth = 2, color = "red")
    plt.xlabel(r"$x$")
    plt.legend()
    if save:
        plt.savefig("solutionTask1dUMR.pdf")
    plt.show()

#----------AMR--------------
def coeff_stencil(i,h):
    """Calculate the coefficients in the four-point stencil for a given index i and list h."""
    #d_p, d_m, d_2m = d_i+1, d_i-1, d_i-2 (notation from the article)
    if i==1:    # Use a 3-point stencil with equal spacing at first iteration, that means h[0]=h[1].
        b = c = 1/h[0]**2
        a = 0
    else:
        d_2m = h[i-2] + h[i-1]
        d_p = h[i]
        d_m = h[i-1]
        a = 2 * (d_p - d_m) / (d_2m*(d_2m + d_p)*(d_2m - d_m)) 
        b = 2 * (d_2m - d_p) / (d_m*(d_2m - d_m)*(d_m + d_p))
        c = 2 * (d_2m + d_m) / (d_p*(d_m + d_p)*(d_2m + d_p))
    return a, b, c

def num_sol_AMR_second(x):
    """Second order discretization of U_xx with nonuniform grid, using the four-point stencil."""
    M = len(x)-2
    a, b, c = np.zeros(M),np.zeros(M),np.zeros(M)
    h = np.diff(x)
    
    for i in range(M):
        a[i],b[i],c[i] = coeff_stencil(i+1,h)  # a[0] = a_1 in the scheme.
    
    data = [a[2:], b[1:], -(a+b+c), c[:-1]]
    diagonals = np.array([-2,-1, 0, 1])  
    Ah = sp.diags(data, diagonals, format = "csc")
    
    alpha = anal_solution(0)
    beta = anal_solution(1)
    
    f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0] - b[0]*alpha
    f_vec[1] = f_vec[1] - a[1]*alpha
    f_vec[-1] = f_vec[-1] - c[-1]*beta

    Usol = sla.spsolve(Ah, f_vec) 
    Usol = np.insert(Usol, 0, alpha)
    Usol = np.append(Usol,beta)
    
    return Usol  

def num_sol_AMR_first(x):
    """First order discretization of U_xx with nonuniform grid, using the three-point stencil."""
    M = len(x)-2
    h = np.diff(x)
 
    b = 2/(h[:-1]*(h[:-1]+h[1:]))    
    c = 2/(h[1:]*(h[1:] + h[:-1]))
    
    data = [b[1:], -(b+c), c[:-1]]   
    diagonals = np.array([-1, 0, 1])
    Ah = sp.diags(data, diagonals, format = "csc")
        
    alpha = anal_solution(x[0])
    beta = anal_solution(x[-1])
    
    f_vec = np.full(M, f(x[1:-1]))
    f_vec[0] = f_vec[0] - alpha*b[0]
    f_vec[-1] = f_vec[-1] - beta*c[-1]

    Usol = sla.spsolve(Ah,f_vec)
    Usol = np.insert(Usol, 0, alpha)
    Usol = np.append(Usol, beta)
    
    return Usol

def calc_cell_errors(x, u, U):
    """Calculates an error for each cell by interpolation and numeric integration."""
    n = len(x) - 1 # Number of cells.
    cell_errors = np.zeros(n)
    v =  lambda x: np.abs(u(x) - U(x))
    for i in range(n):
        cell_errors[i] = cont_L2_norm(v, x[i], x[i + 1])
    return cell_errors

def AMR(x0, steps, num_solver, type): 
    """Mesh refinement 'steps' amount of times. Find the error, x-grid and numerical solution for each step."""
    assert(callable(num_solver))
    disc_error = np.zeros(steps+1)
    cont_error = np.zeros(steps+1)
    M_list = np.zeros(steps+1)
    U_list = [num_solver(x0)]
    x_list = [x0]
    cell_error_list = []
    tol_list = []

    for k in range(steps):
        M_list[k] = len(x_list[-1])-2

        # Calc. discrete l2 error:
        u = anal_solution(x_list[-1])        
        disc_error[k] = e_l(U_list[-1],u)
         
        # Calc. continous L2 error:
        U_interp = interp1d(x_list[-1], U_list[-1], kind = 'cubic')
        cont_error[k] = e_L(U_interp, anal_solution, 0, 1)

        # Refinement: 
        x = list(np.copy(x_list[-1]))
        cell_errors = calc_cell_errors(x, anal_solution, U_interp)
        cell_error_list.append(cell_errors)
        tol = None
        if type == 'avg1':
            diff = lambda x: anal_solution(x) - U_interp(x)
            tol = 1/len(cell_errors) * cont_L2_norm(diff, x[0], x[-1])
        elif type == 'avg2':
            tol = np.average(cell_errors) 
        elif type == 'max':
            tol = 0.7*np.max(cell_errors)
        else:
            raise Exception("Unknown type.")
        tol_list.append(tol)
    
        j = 0 # Index for x in case we insert points.
        for i in range(len(cell_errors)):
            if cell_errors[i] > tol:
                x.insert(j + 1, x[j] + 0.5*(x[j+1] - x[j]))
                j += 1 
            j += 1

        # Tests to check if first two cells have same length.
        if (x[1] - x[0]) != (x[2] - x[1]):
            x.insert(1, x[0] + 0.5*(x[1] - x[0]))
    
        x = np.array(x) 
        x_list.append(x)
        U_list.append(num_solver(x))
        
    # Add last elements.
    u = anal_solution(x_list[-1])
    disc_error[-1] = e_l(U_list[-1],u)
    interpU = interp1d(x_list[-1], U_list[-1], kind = 'cubic')
    cont_error[-1] = e_L(interpU, anal_solution, 0, 1)
    M_list[-1] = len(x_list[-1])-2
    return U_list, x_list, disc_error, cont_error, M_list, cell_error_list, tol_list

def plot_AMR_solution(num_solver, type, save = False):
    """Plot analytical solution and numerical solution using AMR."""
    assert(callable(num_solver))
    M = 3
    x = np.linspace(0, 1, M+2)
    steps = 6
    U, X, _, _, _, _, _= AMR(x, steps, num_solver, type)

    # For colors in plot. 
    name = "tab10"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=colors)
    x_an = np.linspace(0, 1, 1000)
    ax.plot(x_an,anal_solution(x_an),label="Analytical",color = "black", linewidth = 2.0)

    for i in range(0,steps+1):
        ax.plot(X[i],U[i],label=str(i), linestyle = "dashed", linewidth = 1)

    plt.legend()
    plt.xlabel(r"$x$")
    if save:
        if "first" in num_solver.__name__:
            plt.savefig("solutionTask1dAMRFirst.pdf")
        elif "second" in num_solver.__name__:
            plt.savefig("solutionTask1dAMRSecond.pdf")
    plt.show()

def plot_bar_error(X, U, start, stop, cell_error_list, tol_list):
    """Plot error at each cell as bar-plot."""

    plt.rcParams.update({'font.size': 7})
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=(15,15))
    for i in range(start,stop):   
        
        axs.flatten()[i].plot(X[i][:-1], [tol_list[i] for j in range(len(cell_error_list[i]))],label='tolerance',linestyle='dashed', color = "black") # Average AMR.
        axs.flatten()[i].bar(X[i][:-1], cell_error_list[i], align='edge',label=str(i), width = np.diff(X[i]), fill = False, edgecolor = "royalblue")
    
    plt.show()

def plot_AMR_errors(M, x0, steps, type, barplot = False, save=False):
    """Convergence plot from AMR."""
    U_1, X_1, disc_error_1, cont_error_1, M_1, tol_list_1, cell_error_list1 = AMR(x0,steps,num_sol_AMR_first, type)
    U_2, X_2, disc_error_2, cont_error_2, M_2, tol_list_2, cell_error_list2 = AMR(x0,steps,num_sol_AMR_second, type)

    plt.plot(M_1, disc_error_1, label="$e_\ell^r$ (3 point stencil)",color='red',marker='o',linewidth=2)
    plt.plot(M_1, cont_error_1, label="$e_{L_2}^r$ (3 point stencil)",color='red',linestyle="--",linewidth=2)
    plt.plot(M_2, disc_error_2, label="$e_\ell^r$ (4 point stencil)",color='blue',marker='o',linewidth=2)
    plt.plot(M_2, cont_error_2, label="$e_{L_2}^r$ (4 point stencil)",color='blue',linestyle="--",linewidth=2)
    plot_order(M_1, disc_error_1[0], 1, r"$\mathcal{O}(h)$", 'green')
    plot_order(M_2, disc_error_2[0], 2, r"$\mathcal{O}(h^2)$", 'black')
    plt.ylabel(r"Error $e^r_{(\cdot)}$")
    plt.xlabel("$M$")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("loglogtask1dAMR.pdf")
    plt.show()

    if barplot:
        plot_bar_error(X_2, U_2, 0, steps, tol_list_2, cell_error_list2)

#====|-----------------|====#
#====| Run code below. |====#
#====|-----------------|====#

#plot_UMR_solution()
#plot_UMR_errors(barplot = True)

type = "max"
#plot_AMR_solution(num_sol_AMR_first, type)
#plot_AMR_solution(num_sol_AMR_second, type)

#---plot errors---#
M = 4
x0 = np.linspace(0, 1, M+2)
steps = 16

#plot_AMR_errors(M, x0, steps, 'avg2', barplot = True)
