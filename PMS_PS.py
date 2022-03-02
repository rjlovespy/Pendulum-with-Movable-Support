"""
Importing Necessary Libraries to Simulate the Motion of a Simple Pendulum with a Movable Support
"""
import matplotlib.pyplot as plt  
import numpy as np


"""
Defining Constants, Step-size, Initial Conditions, Independent Variable
"""
m1, m2 = 5, 8
l, k, g = 2, 50, 9.8
t0, tf, n = 0, 5, 10000
h = (tf -t0)/n
S0 = np.array([2.25, 0.21, 0, 0])
t = np.linspace(t0,tf,n+1)
S= np.array((n+1)*[S0])


""""
Defining the coupled system of 4 first order ODEs
which is equivalent to the coupled system of 2 second order ODEs(Euler-Lagrange Equations)
"""
def dS_dt(S,t):
    x, phi, v, w = S 
    dx_dt = v
    dphi_dt = w
    dv_dt = ((m2*g*np.sin(phi)*np.cos(phi)) + (m2*l*w*w*np.sin(phi)) - (k*x)) / (m1 + m2*(np.sin(phi)**2))
    dw_dt = np.cos(phi)*(((g*np.tan(phi)*(m1+m2)) + (m2*l*w*w*np.sin(phi)) - (k*x)) / ((m2*l*np.cos(phi)**2) - ((m1+m2)*l)))
    return np.array([dx_dt, dphi_dt, dv_dt, dw_dt])


"""
Applying Runge-Kutta Fourth Order Method because
1. Local Truncation Error in RK45/EM = h^5/h^2 = 10^(-15)/10^(-6)
2. Global Truncation Error in RK45/EM = h^4/h = 10^(-12)/10^(-3)
"""
def rk45(f, n):
    for i in range(n):
        k1 = h * f(S[i], t[i])    
        k2 = h * f(S[i] + 0.5 * k1, t[i] + 0.5*h)
        k3 = h * f(S[i] + 0.5 * k2, t[i] + 0.5*h)
        k4 = h * f(S[i] + k3, t[i] + h)
        S[i+1] = S[i] + (k1 + 2*(k2 + k3 ) + k4) / 6
    return S, t


"""
Calling rk45 to get dependent variables
"""
S,t=rk45(dS_dt,n)
x, phi, v, w = S.T[0],S.T[1],S.T[2],S.T[3]
x1, y1 = x, np.zeros(n+1)  
x2, y2 = x + l*np.sin(phi), -l*np.cos(phi)

"""
Plotting the Position Space
"""
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x1, y1, label="Support")
ax.plot(x2, y2, label="Pendulum")
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_title("Position Space")
ax.grid(which="major")
ax.grid(which="minor", linestyle="--")
ax.minorticks_on()
ax.legend()
fig.suptitle("A Pendulum with a Movable Support")
fig.tight_layout()
plt.show()