"""Utility functions that are imported when in need of plotting solutions."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3d_sol(U, xv, yv, Uan = False, savename = False):
    """Plot numerical solution (and optionally analytical) for task 3: 2D Laplace."""
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.view_init(azim=55, elev=15)
    surface = ax.plot_surface(xv, yv, U, cmap="seismic") 
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("Intensity")
    if callable(Uan):
        x = y = np.linspace(0, 1, 1000) 
        xv, yv = np.meshgrid(x, y)
        surface2 = ax.plot_surface(xv, yv, Uan(xv, yv), cmap="Greys", alpha = 0.7)
    if savename:
        plt.savefig(savename+".pdf")
    plt.show()

def plot3d_sol_time(U, x, t, angle, elevation, Uan = False, savename = False):
    """Plot numerical solution (and optionally analytical) along x- and t-axis."""
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    ax.view_init(azim=angle, elev=elevation)
    xv, tv = np.meshgrid(x,t)
    surface = ax.plot_surface(xv, tv, U, cmap="seismic") 
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("Intensity")

    if callable(Uan):
        x = np.linspace(x[0],x[-1],1000)
        t = np.linspace(t[0],t[-1],1000)
        xv, tv = np.meshgrid(x, t)
        surface2 = ax.plot_surface(xv, tv, Uan(xv, tv), cmap="Greys", alpha = 0.7)
    if savename:
        plt.savefig(savename+".pdf")
    plt.show()
