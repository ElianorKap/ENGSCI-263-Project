from cProfile import label
import numpy as np
from matplotlib import pyplot as plt

def ode_model(t, p, q, a, b):
    p0 = 9
    
    overpressure = 25.3
    
    if p >= overpressure:   
        print("was here")     
        return a * q - b*(p - p0)**2
    else:               

        return a * q

def solve_ode_kettle(f,t0, t1, dt, x0, pars):

    def step_ieuler(f, tk, yk, h, args=None):

        #Solve q first
        def qsolve1(tk):
            q = interpolate_kettle_heatsource(tk)
            return q

        if args is None:
            args = []
        #Find k1
        q = qsolve1(tk)
        k1 = f(tk,yk,q,*args)
        #Take a forward step
        fake_step = yk + h*k1
        #find k2
        q = qsolve1(tk + h)
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
            xk_1 = step_ieuler(f, i-dt, x0, dt, pars)
            x0 = xk_1
            x.append( xk_1 )

    return t,x


def interpolate_kettle_heatsource(t):

    time , mass = np.genfromtxt( 'gs_mass.txt' , delimiter=',', skip_header = 1 ).T
    secondsPerMonth = 2628288 #average month
    q = mass / secondsPerMonth
    q = np.interp(t,time,q)

    return q

def plot_kettle_model():
    
    time , pressure = np.genfromtxt( 'gs_pres.txt' , delimiter=',', skip_header = 1 ).T
    
    plt.plot( time, pressure ,color='black' ,label="experimental data")

    a = 14
    b = 9*10**(-3)
    t , p = solve_ode_kettle(ode_model, 2009 ,2019, 0.1  , 25.3 , [a ,b])
    plt.plot( t,p ,color="blue",label="model")
    plt.legend(loc="lower left")
    plt.show()
    

if __name__ == "__main__":
    
    plot_kettle_model()