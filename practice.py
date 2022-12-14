import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def ode_model(t, p, q, p0 ,  a, d ):
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
        d : float
            Lumped parameter.
        p0 : float
            Ambient value of dependent variable.
        Returns:
        --------
        dpdt : float
            Derivative of dependent variable with respect to independent variable.
    '''
    overpressure = 25.6
    #if greater than overpressure inlcude leakage
    if p >= overpressure:
        #print(p)
        return (a * q - d*(p - p0)**2)
    #otherwise NO LEAKAGE
    return (a * q)


def solve_ode_kettle(f,t0, t1, dt, x0, pars, scale=1.):
    """ Compute solution of initial value ODE problem using explicit RK method.
    Parameters
    ----------
    f : callable
    Derivative function.

    t0 : float
    Initial value of independent variable.

    t1 : float
    Final value of independent variable.

    y0 : float
    Initial value of solution.

    h : float
    Step size.

    pars : iterable
    Optional parameters to pass into derivative function.

    Returns
    -------
    t : array-like
    Independent variable at solution.

    y : array-like
    Solution
    """
    def step_ieuler(f, tk, yk, h, args=None, scale=1.):
        ''' Finds solution after 1 improved euler step
        Parameters
        ----------
        f : callable
        Derivative function.
        tk : float
        current time step value.
        yk : float
        value of solution at time tk.
        h : float
        Step size.
        args : iterable
        Optional parameters to pass into derivative function.

        Returns
        -------
        yk_1 : float
        Value of solution at step size h from time tk.
        '''
        #Solve q first
        def qsolve1(tk, scale=1.):
            q = interpolate_kettle_heatsource(tk, scale)
            return q

        if args is None:
            args = []
        #Find k1
        q = qsolve1(tk, scale)
        k1 = f(tk,yk,q,*args)
        #Take a forward step
        fake_step = yk + h*k1
        #find k2
        q = qsolve1(tk + h, scale)
        k2 = f(tk+h, fake_step,q,*args)
        #approximate solution value for next step
        yk_1 = yk + 0.5*h*(k1+k2)
        return yk_1

    if pars is None:
        pars = []
    
    t = np.linspace(t0, t1, int( (t1-t0)/dt)+1 )

    x = [x0]
    t[0] = t0
    t[-1] = t1
    xstart = x0
    for ind,i in enumerate(t) :
        #initial value already stored so can skip
        if ind == 0:
            continue
        else:
            #Uses improved euler
            xk_1 = step_ieuler(f, i-dt, x0, dt, pars, scale)
            x0 = xk_1
            x.append( xk_1 )

    return t,x


def interpolate_kettle_heatsource(t, scale=1.):
    '''Finds mass values on a given month and scales them according scale factor
    
    Parameters
    ---------
    t : int
        time value at which to find q value

    scale : integer
        Scales the injection/extraction flows

    Return
    --------
    q[i] : Integer
        Value of q at relevant time stamp
    
    '''
    #Read relevant values
    time , mass = np.genfromtxt( 'gs_mass.txt' , delimiter=',', skip_header = 1 ).T
    secondsPerMonth = 2628288 #average month 
    q = scale*mass / secondsPerMonth #convert to Kg/s

    #find correct time from which to read the q value and return it
    for i in range(len(time)):
        if t <= time[i]:
            return q[i]

# def objective(pars):
#     time , pressure = np.genfromtxt( 'gs_pres.txt' , delimiter=',', skip_header = 1 ).T
#     t = np.linspace(2009.0,2019.0,201)
#     t , p = solve_kettle_ode(ode_model, t, 25.16 , pars)
#     p_model = [p[i] for i in range(0,len(t)+1, 5 )]


#     # calculate the objective function
#     obj = 0.	# initiate

#     for i in range(len(time)):	# runs through time list
#         obj += (pressure[i]-p_model[i])**2		# add weighted squared enthalpy difference

#     return obj

def misfit(pars):
    ''' Find and return misfit vector and prints squared sum error

    Parameters
    ----------
    pars : iterable
        Optional parameters to pass into derivative function.

    Returns
    ----------
    misfitVector : array
        contains misfit values evaluated at every data point we have been given.
    
    '''
    time , pressure = np.genfromtxt( 'gs_pres.txt' , delimiter=',', skip_header = 1 ).T

    t , p = solve_ode_kettle(ode_model, 2009 ,2019, 0.05 , 25.16 , pars)

    p_model = [p[i] for i in range(0,len(t), 5 )] 


    # calculate the misfit vectore
    misfitVector= []	# initiate

    for i in range(len(time)):	# runs through time list
        misfitVector.append(pressure[i]-p_model[i])	 	# add the calculated misfit
    tot = 0 #initiate total
    for i in misfitVector:
        tot += i**2 #Add on square of each misfit
    print("squared sum error: ",tot) #print that total value
    
    return misfitVector #return misfitvector


def plot_kettle_model():
    '''Plots initial and best fit models with their misfits, and also prints the errors

    Inputs
    --------
    None

    Returns
    --------
    None
    
    '''
    time , pressure = np.genfromtxt( 'gs_pres.txt' , delimiter=',', skip_header = 1 ).T

    #plt.plot( time, pressure ) 
    a =  18.1
    b =  0.026
    p0 = 10.87  
    #0.57 obj

    t , p = solve_ode_kettle(ode_model, 2009 ,2019, 0.1 , 25.16 , [p0,a,b] )
    
    misfitVector  = misfit([p0,a,b])
    
    #tb , pb = solve_ode_kettle(ode_model, 2009 ,2019, 0.1 , 25.16 , [ 0.1, 12.1,0.009] )
    
    tb , pb = solve_ode_kettle(ode_model, 2009 ,2019, 0.1 , 25.16 , [0.1, 12.1, 0.006] )

    # plt.plot( t,p , 'r')
    # plt.plot(tb,pb, "b")

    f,ax = plt.subplots(1, 2, figsize=(18,6))
    ax[0].set_title(f"a = {a} , d = {b} , overpressure = 25.6 , p0 ={p0}")
    ax[0].plot(time,pressure, 'k' ,label='observations')
    #ax.plot(tb,pb, 'r-', label='model guess')

    ax[0].plot(t,p, 'b-', label='model improved')
    #obj is now 0.996

    ax[1].plot( time, misfitVector , 'x')
    ax[1].set_ylim(-0.4,0.4)
    ax[1].plot( t, np.zeros(len(t)) , '.')
    ax[0].set_xlabel('time, $t$ [year]')

    ax[1].set_xlabel('time, $t$ [year]')
    ax[1].set_ylabel('Pressure misfit, $MPa$')

    ax[0].set_ylabel('Pressure, $MPa$')
    f.suptitle("Improved Best-Fit LPM Model")
    ax[0].legend()

    # p_model = [p[i] for i in range(0,len(t)+1, 5 )]
    #constants=curve_fit(Tmodel,  p_model , pressure, [a,b])
    
    #a_const=constants[0][0]
    #b_const=constants[0][1]
    plt.show()
    #plt.savefig("modelFitImproved")
    a =  12.1
    b =  0.006
    p0 = 0.1
    pars=[p0, a,b ]
    t , p = solve_ode_kettle(ode_model, 2009 ,2019, 0.1 , 25.16 , pars )
    misfitVector  = misfit(pars)

    f,ax = plt.subplots(1, 2, figsize=(18,6))
    ax[0].set_title(f"a = {a} , d = {b} , overpressure = 25.6 , p0 ={p0}")
    ax[0].plot(time,pressure, 'k' ,label='observations')
    #ax.plot(tb,pb, 'r-', label='model guess')

    ax[0].plot(t,p, 'b-', label='model improved')
    #obj is now 0.996

    ax[1].plot( time, misfitVector , 'x')
    ax[1].plot( t, np.zeros(len(t)) , '.')
    ax[1].set_ylim(-0.4,0.4)
    ax[0].set_xlabel('time, $t$ [year]')
    ax[0].set_ylabel('Pressure, $MPa$')
    f.suptitle("Initial Best Fit LPM Model")
    ax[0].legend()

    ax[1].set_xlabel('time, $t$ [year]')
    ax[1].set_ylabel('Pressure misfit, $MPa$')

    plt.show()

if __name__ == "__main__":
    
    
    plot_kettle_model()


#BEST FOR 10.87 p0 is a = 18.1, b = 0.026S

# STARTED FORM pars=[0.1, 12.1, 0.006]
#obj was  2.7452737435723424

# tb , pb = solve_ode_kettle(ode_model, 2009 ,2019, 0.1 , 25.16 , [ 0.001, 29.86,0.009] )
#^ for gradient descent 

