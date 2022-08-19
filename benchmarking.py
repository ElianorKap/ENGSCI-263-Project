# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
from matplotlib import pyplot as plt

def ode_model(t, p, q, a, B, p0):

    dpdt = -a*q-B*(p-p0)

    return dpdt

def solve_ode(f, t0, t1, dt, p0, pars):
    tspan = t1 - t0
    k = int(tspan / dt)
    t = np.linspace(t0, t1, k + 1)
    p = [p0]
    for i in range(1,len(t)):
        f0 = f(t[i-1], p[i-1], *pars)
        f1 = f(t[i-1] + dt, p[i-1] + dt * f0, *pars)
        p.append(p[i-1] + dt * ((f0 / 2) + (f1 / 2)))
    return t, p

def benchmarking():
    t = []
    for i in np.arange(0, 10, 0.5):
        t.append(i)
    a = 1
    B = 1
    q = -1
    p0 = 0
    t0 = t[0]
    t1 = t[-1]
    dt = 0.1
    fun = ode_model
    parm = [q, a, B, p0]
    x1, y1 = solve_ode(fun, t0, t1, dt, p0, parm)
    y2 = np.zeros(len(x1))
    for i in range(len(x1)):
        y2[i] = -((a*q)*(1-math.e**(-B*x1[i])))/B+p0
    f, ax = plt.subplots(1, 1)
    ax.plot(x1, y1, 'x', label='numerical solution')
    ax.plot(x1, y2, 'r-', label='analytical solution')
    ax.set_ylabel('Pressure,P')
    ax.set_xlabel('time,t')
    ax.set_title("benchmark")
    ax.legend(loc=2)
    f1,ax1 = plt.subplots(1, 1)
    error = np.zeros(len(x1))
    for i in range(len(x1)):
        error[i] = abs(y2[i]-y1[i])/abs(y1[i])
    ax1.plot(x1, error, 'k.')
    ax1.set_ylabel('relative error against benchmark')
    ax1.set_xlabel('t')
    ax1.set_title("error analysis")
    f2, ax2 = plt.subplots(1, 1)
    con = []
    thetat = []
    for i in np.arange(0.1,1,0.05):
        x3,y3 = solve_ode(fun, 0, 10, i, p0, parm)
        con.append(y3[-1])
        thetat.append(1/i)
    print(con)
    ax2.plot(thetat, con, 'k.')
    ax2.set_ylabel('P(t=10)')
    ax2.set_xlabel('1/theta(t)')
    ax2.set_title("timestep convergence")
    plt.show()



if __name__ == '__main__':
    benchmarking()
