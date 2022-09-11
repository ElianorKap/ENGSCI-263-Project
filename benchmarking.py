# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
from matplotlib import pyplot as plt

def ode_model(t, p, q, a, B, p0):
    ''' Return the derivative dp/dt at time, t, for given parameters.

            Parameters:
            -----------
            t : float
                Independent variable.
            p : float
                Dependent variable.
            q : float
                mass flow rate.
            a : float
                Lumped parameter.
            B : float
                Binary control variable.
            p0 : float
                Ambient value of dependent variable.

            Returns:
            --------
            dpdt : float
                Derivative of dependent variable with respect to independent variable.

            Notes:
            ------
            This is a simplified version for dp/dt
        '''
    dpdt = -a*q-B*(p-p0)

    return dpdt

def solve_ode(f, t0, t1, dt, p0, pars):
    ''' Solve an ODE numerically.

            Parameters:
            -----------
            f : callable
                Function that returns dpdt given variable and parameter inputs.
            t0 : float
                Initial time of solution.
            t1 : float
                Final time of solution.
            dt : float
                Time step length.
            p0 : float
                Initial value of solution.
            pars : array-like
                List of parameters passed to ODE function f.

            Returns:
            --------
            t : array-like
                Independent variable solution vector.
            p : array-like
                Dependent variable solution vector.

            Notes:
            ------
            using the Improved Euler Method to solve ode
            '''
    # calculate the time span
    tspan = t1 - t0
    k = int(tspan / dt)
    t = np.linspace(t0, t1, k + 1)
    p = [p0]
    # using the Improved Euler Method to solve ode
    for i in range(1,len(t)):
        f0 = f(t[i-1], p[i-1], *pars)
        f1 = f(t[i-1] + dt, p[i-1] + dt * f0, *pars)
        p.append(p[i-1] + dt * ((f0 / 2) + (f1 / 2)))
    # return the time and pressure for numerical solution
    return t, p

def benchmarking(a, B, q, p0, plot = True):
    ''' Compare analytical and numerical solutions.

            Parameters:
            -----------
            a : float
                Lumped parameter.
            B : float
                Binary control variable.
            q : float
                mass flow rate.
            p0 : float
                Initial value of solution.
            plot: Bool
                If True, creates plots of timestep convergence and error analysis

            Returns:
            --------
            analytical_solution : float
                                Analytical solution at final time.
            numerical_solution : float
                                Numerical solution at final time.

            Notes:
            ------
            This function called within if __name__ == "__main__":

            It should contain commands to obtain analytical and numerical solutions,
            plot the benchmarking, relative error against benchmarking and timestep convergence

        '''
    # get the value of time for two solution
    t = []
    for i in np.arange(0, 100, 2):
        t.append(i)
    t0 = t[0]
    t1 = t[-1]
    dt = 0.1
    fun = ode_model
    parm = [q, a, B, p0]
    # use solve_ode function get the numerical solutions
    x1, y1 = solve_ode(fun, t0, t1, dt, p0, parm)
    # get the analytical solutions
    y2 = np.zeros(len(x1))
    for i in range(len(x1)):
        y2[i] = -((a*q)*(1-math.e**(-B*x1[i])))/B+p0
    # plot the benchmarking
    if plot == True:
        f, ax = plt.subplots(1, 1, figsize=(12,6))
        ax.plot(x1, y1, 'x', label='numerical solution')
        ax.plot(x1, y2, 'r-', label='analytical solution')
        ax.set_ylabel('Pressure, MPa')
        ax.set_xlabel('Time, s')
        ax.set_title("Benchmark, a = 18.1, d = 0.026, p0 = 10.87")
        ax.legend(loc=2)
        plt.savefig('Benchmark plot 1')

        # plot the relative error against the benchmarking
        f3,ax3 = plt.subplots(1, 1, figsize=(12,6))
        error = np.zeros(len(x1))
        for i in range(len(x1)):
            error[i] = abs(y2[i]-y1[i])/abs(y1[i])
        ax3.plot(x1, error, 'k.')
        ax3.set_ylabel('Relative error against benchmark')
        ax3.set_xlabel('Time, s')
        ax3.set_title("Error Analysis")
        # plot the timestep convergence
        plt.savefig('Benchmark plot 3')

        f2, ax2 = plt.subplots(1, 1, figsize=(12,6))
        con = []
        thetat = []
        for i in np.arange(0.0001,1,0.0001):
            x3,y3 = solve_ode(fun, 0, 10, i, p0, parm)
            # record the pressure when (t = 10)
            con.append(y3[-1])
            # record the 1/theta(t)
            thetat.append(1/i)
        # plot the graph
        ax2.plot(thetat, con, 'k.')
        ax2.set_ylabel('P(t=10)')
        ax2.set_xlabel('1/theta(t)')
        ax2.set_title("timestep convergence")
        plt.savefig('Benchmark plot 2')

        plt.show()
    analytical_solution = y1[-1]
    numerical_solution = y2[-1]
    return analytical_solution, numerical_solution


if __name__ == '__main__':
    benchmarking(18.1,0.026,1,10.87)
