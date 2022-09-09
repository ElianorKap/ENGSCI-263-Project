# ENGSCI263: Lab Exercise 2
# lab2.py

# PURPOSE:
# IMPLEMENT a lumped parameter model and CALIBRATE it to data.

# PREPARATION:
# Review the lumped parameter model notes and obtain data from the kettle experiment.

# SUBMISSION:
# - Show your calibrated LPM to the instructor.

# imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def ode_model(t, p, q, p0 ,a, b):
    overpressure = 25.6
    if p >= overpressure:
        #print(p)
        return (a * q - b*(p - p0)**2)
    
    return (a * q)


def interpolate_kettle_heatsource(t, scale=1.):

    time , mass = np.genfromtxt( 'gs_mass.txt' , delimiter=',', skip_header = 1 ).T
    secondsPerMonth = 2628288 #average month
    q = scale * mass / secondsPerMonth
    #q = np.interp(t,time,q)
    res = []
    
    for stamp in t:
        found = 0
        for i in range(len(time)):
            if not found and stamp <= time[i]:
                res.append( q[i] )
                found = 1
    return res

def solve_kettle_ode(f, t, x0, pars, scale=1.):
    ''' Solve the kettle ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method. 

        Function q(t) should be hard coded within this method. Create duplicates of 
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''

    dt = round(t[1]-t[0] , 2)
    x = 0.*t
    x[-1] = x0
    q = interpolate_kettle_heatsource(t, scale)
    for i in range(0, len(t)):
        dxdtp = ode_model(t[i-1], x[i-1], q[i-1],  *pars)
        xp = x[i-1] + dt*dxdtp
        dxdtc = ode_model(t[i], xp, q[i],  *pars)
        x[i] = x[i-1] + dt*(dxdtp+dxdtc)/2.
    return t, x

def objective(pars):
    time , pressure = np.genfromtxt( 'gs_pres.txt' , delimiter=',', skip_header = 1 ).T
    t = np.linspace(2009.0,2019.0,201)
    t , p = solve_kettle_ode(ode_model, t, 25.16 , pars)
    p_model = [p[i] for i in range(0,len(t)+1, 5 )] 


    # calculate the objective function
    obj = 0.	# initiate

    for i in range(len(time)):	# runs through time list
        obj += (pressure[i]-p_model[i])**2		# add weighted squared enthalpy difference

    return obj

def plot_kettle_model():
    ''' Plot the kettle LPM over top of the data.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to read and plot the experimental data, run and 
        plot the kettle LPM for hard coded parameters, and then either display the 
        plot to the screen or save it to the disk.

    '''
    to , To = np.genfromtxt( 'gs_pres.txt' , delimiter=',', skip_header = 1 ).T

    timeModel , mass = np.genfromtxt( 'gs_mass.txt' , delimiter=',', skip_header = 1 ).T
    #pars=[5.2e-4,0.7e-3] - best solution
    
    pars=[0.1, 12.1, 0.006]
    x0=25.16
    tm,Tm = solve_kettle_ode(ode_model, timeModel, x0, pars)
    print("obj was ",objective(pars))
    def Tmodel(t, *pars):
        x0=25.16
        tm,Tm = solve_kettle_ode(ode_model, t, x0, pars)
        return Tm

    #p0 = [5.2e-4,0.7e-3]
    # our guess
    #p0=[10 , 24, 0.1]

    constants=curve_fit(Tmodel, to, To, pars, check_finite= True)
    a_const=constants[0][1]
    b_const=constants[0][2]
    c_const= constants[0][0]

    print("a = ", a_const)
    print("b = ",b_const)
    print("p0 = ", c_const)
 
    pars=[c_const, a_const , b_const]
    x0=25.16
    tmi,Tmi = solve_kettle_ode(ode_model, timeModel , x0, pars)
    print("obj is now",objective(pars))

    f1,ax1 = plt.subplots(1, 1, figsize=(12,6))
    ax1.plot(to,To, 'k' ,label='observations')
    ax1.plot(tm,Tm, 'r-', label='model guess')

    f2,ax2 = plt.subplots(1, 1, figsize=(12,6))

    ax1.plot(to,To, 'k' ,label='observations')
    ax2.plot(tmi,Tmi, 'b-', label='model improved')
    
    ax.set_xlabel('time, $t$ [s]')

    ax.set_ylabel('temperature, $T$ [$^{\circ}$C]')
    ax.set_title("Our Model initial parameter estimate improvement using a curve fitting function")
    ax.legend()
    plt.show()

if __name__ == "__main__":

    #plot_benchmark()
    plot_kettle_model()

