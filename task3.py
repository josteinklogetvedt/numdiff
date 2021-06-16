"""Code for problem 3.

Numerical solution of the 2D Laplace equation, with the given boundary conditions, on the unit square.
h and k, i.e. step sizes in x and y respectively, DO NOT have to be equal. 
"""
from scipy.sparse import diags # Make sparse matrices with scipy.
import scipy.sparse.linalg as sla
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from plotting_utilities import plot3d_sol
from utilities import *

class Task3:
    def plot_solution(self, Mx = 50, My = 50):
        """Calculate numerical solution on specified grid, and plot together with analytical solution."""
        U, xv, yv = self.num_solution_Mx_My(Mx = Mx, My = My)
        plot3d_sol(U, xv, yv, Uan = self.analytic_solution)

    def convergence_plot(self, varying, savename = False):
        """Make convergence plot specified by which quantity is varying."""
        assert(varying == "Mx" or varying == "My" or varying == "Both") 
        self._colors = ["red", "green", "black", "orange"]
        self._powers = [2] # Power used in the convergence plot. 

        # Assert that the savename variable is of the correct format.
        if (varying == "Mx" or varying == "My") and savename:
            assert(type(savename) is list and len(savename) == 4)
        elif savename:
            assert(isinstance(savename, str))

        if varying == "Mx":
            self._constant_list = [10, 100, pow(10, 3), pow(10, 4)] # Constant values in plots. 
            maximum = 2**7 # Maximum limit of Mx.
        elif varying == "My":
            self._constant_list = [10, 100, pow(10, 3), pow(10, 4)] # Constant values in plots. 
            maximum = 2**7 # Maximum limit of My.
        elif varying == "Both":
            maximum = 2**10 # Maximum limit of My and Mx. 
            self._powers = [1] # Power used in the convergence plot. 

        varying_list = 2 ** np.arange(1, np.log(maximum)/np.log(2)+1, dtype = int)
        if varying == "Both":
            self._discrete_error = np.zeros(len(varying_list))
            for i, m in enumerate(varying_list):
                Usol, xv, yv = self.num_solution_Mx_My(Mx = m, My = m)
                analsol = self.analytic_solution(xv, yv)
                self._discrete_error[i] = e_l(Usol, analsol)
            if savename:
                self.plot_plots(varying_list, varying_list, savename=savename)
            else: 
                self.plot_plots(varying_list, varying_list)
        elif varying:
            for j, constant in enumerate(self._constant_list):
                self._discrete_error = np.zeros(len(varying_list))
                for i, m in enumerate(varying_list):
                    if varying == "Mx":
                        Usol, xv, yv = self.num_solution_Mx_My(Mx = m, My = constant)
                    elif varying == "My":
                        Usol, xv, yv = self.num_solution_Mx_My(Mx = constant, My = m)

                    analsol = self.analytic_solution(xv, yv)
                    self._discrete_error[i] = e_l(Usol, analsol)
                if savename:
                    self.plot_plots(varying_list, constant, savename=savename[j])
                else: 
                    self.plot_plots(varying_list, constant)

    def analytic_solution(self, x, y):
        """Analytical solution to the 2D Laplace equation."""
        return (1/np.sinh(2*np.pi))*np.sinh(2*np.pi*y)*np.sin(2*np.pi*x)

    def num_solution_Mx_My(self, Mx, My):
        """Numerical solution of 2D Laplace.
        
        Input: 
        Mx: Number of internal points in x-direction. 
        My: Number of internal points in y-direction. 
        """
        M2 = Mx*My
        h = 1/(Mx+1)
        k = 1/(My+1)

        # Construct A. 
        data = np.array([np.full(M2, 1/k**2)[:-1], np.full(M2, -2*(1/h**2+1/k**2)), np.full(M2, 1/k**2)[:-1], 
                            np.full(M2, 1/h**2)[:-My], np.full(M2, 1/h**2)[:-My]])
        diag = np.array([-1, 0, 1, -My, My])
        A = diags(data, diag, format = "csc")
        
        # Construct F.
        F = np.zeros(M2)
        point_counter = 1

        # Change some elements to match the correct linear system + construct F. 
        for i in range(My, My*Mx, My):
            A[i-1, i] = A[i, i-1] = 0
            F[i-1] = -(1/k**2)*np.sin(2*np.pi*point_counter*h)
            point_counter += 1

        # Add values out of loop-bound.  
        F[M2-1] = -(1/k**2)*np.sin(2*np.pi*point_counter*h)

        # Solve linear system.  
        Usol = sla.spsolve(A, F) 
        
        # Next, want to unpack into grids, for plotting.
        x = np.linspace(0, 1, Mx+2)
        y = np.linspace(0, 1, My+2) 
        xv, yv = np.meshgrid(x, y)
        U = np.zeros_like(xv) # Grid for solution. Need to insert solution into this, including boundary conditions. 

        # Insert upper boundary condition last in U, since y increases "downwards" in yv. 
        U[-1, :] = np.sin(2*np.pi*x)

        # Need to unpack the solution vector with the correct coordinates. 
        for i in range(int(len(Usol)/My)): # This gives the rows (x-values).
            for j in range(My): # This gives the columns (y-values).
                U[j+1, i+1] = Usol[j + (My*i)]

        return U, xv, yv

    def plot_plots(self, Mx, My, savename = False):
            """Helper function used when plotting convergence plots."""
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(Mx*My, self._discrete_error, label=r"$e^r_\ell$", color = "blue", marker = "o", linewidth = 3)
            for i, p in enumerate(self._powers): # If one wants to plot several orders. 
                plot_order(Mx*My, self._discrete_error[0], p, r"$\mathcal{O}(N_{dof}^{-%s})$)" % str(p), self._colors[i])
            ax.set_ylabel(r"Error $e^r_{(\cdot)}$")
            ax.set_xlabel(r"$M_x \cdot M_y$")
            plt.legend()
            plt.grid() 
            if savename:
                plt.savefig(savename+".pdf")
            plt.show() 
    
solution = Task3()
#solution.plot_solution()
#solution.convergence_plot("Mx", savename=["task3bMy20", "task3bMy50", "task3bMy100", "task3bMy500"])
#solution.convergence_plot("My", savename=["task3bMx20", "task3bMx50", "task3bMx100", "task3bMx500"])
#solution.convergence_plot("Both", savename="task3bBothVary")
