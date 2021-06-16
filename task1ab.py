"""Code for problem 1 a) and b). 

a) Numerical solution using the given difference method in task 1a.

b) Modify code from a) with different boundary conditions.
In this case we have Dirichlet on both sides, which corresponds to 
the simple case example in 3.1 in BO's note.
"""
from scipy.interpolate import interp1d 
from scipy.integrate import quad
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from utilities import *

class Task1ab:
    """Solution to 1a) and 1b)."""
    def __init__(self, problem, M):
        """Constructor.

        Input:
        problem: Specify which problem is desired to solve.
        M: Points on x-axis.
        """
        assert(problem == "a" or problem == "b")
        if problem == "a":
            self._problem = "a"
            self._alpha = 0
            self._sigma = 0
            self._num_solution = self.num_solution_a # Alias function for numerical solution. 
        else:
            self._problem = "b"
            self._alpha = 1
            self._sigma = 1/3
            self._num_solution = self.num_solution_b # Alias function for numerical solution. 

        self.M = 40 
        self._x = np.linspace(0, 1, M+2) # Make 1D grid.

    def plot_solution(self, save = False):
        """Plot analytical and numerical solution to the problem.
        
        Input: 
        save: Dictate if a pdf of the plot is saved or not. 
        """
        plt.plot(self._x, self.anal_solution(self._x), label="An", color = "black", marker = "o", linewidth = 2)
        plt.plot(self._x, self._num_solution(self._x, self.M), label="Num", color = "red", linestyle = "dotted", marker = "o", linewidth = 3)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$u$")
        plt.legend()
        if save:
            plt.savefig("solutionsTask1"+self._problem+".pdf") 
        plt.show()

    def plot_convergence_plot(self, save = False):
        """Construct and plot convergence plot.
    
        Input: 
        save: Dictate if a pdf of the plot is saved or not.
        """
        M = 2**np.arange(2, 11, dtype = int)
        discrete_error = np.zeros(len(M))
        cont_error = np.zeros(len(M))

        for i, m in enumerate(M):
            x = np.linspace(0, 1, m+2)
            Usol = self._num_solution(x, M = m)
            analsol = self.anal_solution(x)
            interpU = interp1d(x, Usol, kind = 'cubic')
            cont_error[i] = self.e_L(interpU, self.anal_solution, x)
            discrete_error[i] = e_l(Usol, analsol)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(M, discrete_error, label=r"$e^r_\ell$", color = "blue", marker = "o", linewidth = 3)
        ax.plot(M, cont_error, label = r"$e^r_{L_2}$", color = "red", linestyle = "--", marker = "o", linewidth = 2)
        plot_order(M, discrete_error[0], order = 2, label=r"$\mathcal{O}$($h^2$)", color = "green")
        ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
        ax.set_xlabel("Number of points M")
        plt.legend()
        plt.grid() 
        if save:
            plt.savefig("loglogtask1"+self._problem+".pdf")
        plt.show() 

    def f(self, x):
        """Right hand side of 1D Poisson equation."""
        return np.cos(2*np.pi*x) + x

    def anal_solution(self, x):
        """Analytical solution of the Possion equation with given Neumann BC.
    
        alpha = sigma = 0 gives solution from 1a).
        alpha = 1, sigma = 1/3 gives solution from 1b).
        """
        return -1/(4*np.pi**2)*np.cos(2*np.pi*x) + 1/6*x**3 + (self._sigma - 1/2)*x + self._alpha + 1/(4*np.pi**2)

    #----Utility methods to calculate numerical solutions follow.---- 
    def num_solution_a(self, x, M):
        """Numerical solution of the Possion equation with given Neumann BC in 1a)."""
        assert(M >= 3)
        h = 1/(M+1)

        # Construct Ah. 
        data = np.array([np.full(M+1, 1), np.full(M+1, -2), np.full(M+1, 1)])
        diags = np.array([-1, 0, 1])
        Ah = sp.spdiags(data*1/h**2, diags, M+1, M+1, format = "lil")
        
        Ah[-1, -3] = h/2
        Ah[-1, -2] = -2*h
        Ah[-1, -1] = 3*h/2

        Ah = Ah.tocsr() # Change format in order to implement sparse solver of linear equations. 

        # Construct f.
        f_vec = np.full(M+1, self.f(x[1:]))
        f_vec[0] = f_vec[0] - 0/h**2
        f_vec[-1] = 0

        # Solve linear system. 
        Usol = sla.spsolve(Ah, f_vec) # Use sparse solver. 

        # Add left Dirichlet condition to solution.
        Usol = np.insert(Usol, 0, 0)
        return Usol

    def num_solution_b(self, x, M):
        """Numerical solution of the Possion equation with given Neumann BC in 1b)."""
        h = 1/(M+1)

        # Construct Ah. 
        data = np.array([np.full(M, 1), np.full(M, -2), np.full(M, 1)])
        diags = np.array([-1, 0, 1])
        Ah = sp.spdiags(data*1/h**2, diags, M, M, format = "lil")

        # Construct f.
        f_vec = np.full(M, self.f(x[1:-1]))
        f_vec[0] = f_vec[0] - 1/h**2
        f_vec[-1] = f_vec[-1] - 1/h**2

        Ah = Ah.tocsr() # Change format in order to implement sparse solver of linear equations.

        # Solve linear system. 
        Usol = sla.spsolve(Ah, f_vec) # Use sparse solver. 


        # Add left Dirichlet condition to solution.
        Usol = np.insert(Usol, 0, 1)

        # Add right Dirichlet condition to solution.
        Usol = np.append(Usol, 1)
        return Usol

    #----Utility methods to make convergence plots for both norms follow.---- 
    def cont_L2_norm(self, v, x):
        """Continuous L2 norm of v(x) between left and right.
        
        This is a bit different to the function in utilities.py, because of alpha and sigma.
        """
        assert(callable(v))
        integrand = lambda x: v(x)**2
        return np.sqrt(quad(integrand, x[0], x[-1])[0])

    def e_L(self, U, u, x):
        """Relative error e_L.
    
        U: Approximate numerical solution.
        u: Function returning analytical solution. 
        x: x-axis to find relative error on.

        This is a bit different to the function in utilities.py, because of alpha and sigma.
        """
        assert(callable(u) and callable(U))
        f = lambda x: u(x) - U(x)
        numer = self.cont_L2_norm(f, x)
        denom = self.cont_L2_norm(u, x)
        return numer/denom

# Make objects. Plot solutions and plot convergence plots.
taskonea = Task1ab(problem = "a", M = 40)
#taskonea.plot_solution()
#taskonea.plot_convergence_plot()

taskoneb = Task1ab(problem = "b", M = 40)
#taskoneb.plot_solution()
#askoneb.plot_convergence_plot()
