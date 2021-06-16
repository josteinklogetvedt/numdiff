"""Runge-Kutta and Runge-Kutta-Nystrøm integrators."""
import numpy as np 

def RK2_step(k, t_i, y, f):
    """One step in RK2-method."""
    s1 = f(t_i, y)
    s2 = f(t_i +k, y + k*s1)
    return y + (k/2)*(s1+s2)

def RK3_step(k, t_i, y, f):
    """One step in RK3-method."""
    s1 = f(t_i,y)
    s2 = f(t_i+k/2, y + (k/2)*s1)
    s3 = f(t_i+k, y - k*s1 + 2*k*s2)
    return y + (k/6)*(s1 + 4*s2 + s3)

def RK4_step(k, t_i, y, f):
    """One step in RK4-method.""" 
    s1 = f(t_i, y)
    s2 = f(t_i + k/2, y + (k/2)*s1)  
    s3 = f(t_i + k/2, y + (k/2)*s2) 
    s4 = f(t_i + k, y + k*s3)
    return y + (k/6)*(s1 + 2*s2 + 2*s3 + s4)

def RKN12_step(k, t_i, y, f):
    """One step in RKN-12 method."""
    s1 = f(t_i, y[0])
    
    y_der_new = y[1] + k*s1
    y_new = y[0] + k*y[1] + k**2*(s1/2)
    return np.array([y_new, y_der_new])

def RKN34_step(k, t_i, y, f):
    """One step in RKN-34 method."""
    delta = (1/12)*(2 - 4**(1/3)-16**(1/3))
    s1 = f(t_i+(1/2-delta)*k, y[0] + (1/2-delta)*k*y[1])
    s2 = f(t_i + (1/2)*k, y[0] + (1/2)*k*y[1]+k**2/(24*delta)*s1)
    s3 = f(t_i + (1/2 + delta)*k, y[0] + (1/2+delta)*k*y[1] + k**2*(1/(12*delta)*s1 + 
                            (delta-1/(12*delta))*s2))
    
    y_der_new = y[1] + k*((s1+s3)/(24*delta**2) + (1-1/(12*delta**2))*s2)
    y_new = y[0] + k*y[1] + k**2*((1-1/2+delta)/(24*delta**2)*s1 + 
                    ((1-1/(12*delta**2))*(1-1/2))*s2 + (1-1/2-delta)/(24*delta**2)*s3)
    return np.array([y_new, y_der_new])
