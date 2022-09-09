
#########################################################################################
#
# Function library for plotting lumped parameter model
#
# 	Functions:
#		lmp_2D: lumped parameter model with 2 variables
#		lmp_3D: lumped parameter model with 3 variables
#		obj: objective function for a lmp
#
#########################################################################################

# import modules and functions
import numpy as np

# global variables - observations
tq,q = np.genfromtxt('gs_pres.txt', delimiter = ',', skip_header = 1).T
tp,p = np.genfromtxt('gs_mass.txt', delimiter = ',', skip_header = 1).T
dqdt = 0.*q                 # allocate derivative vector
dqdt[1:-1] = (q[2:]-q[:-2])/(tq[2:]-tq[:-2])    # central differences
dqdt[0] = (q[1]-q[0])/(tq[1]-tq[0])             # forward difference
dqdt[-1] = (q[-1]-q[-2])/(tq[-1]-tq[-2])        # backward difference

# define derivative function
def lpm(t, p, q, p0, a, b):
    overpressure = 25.6
    if p >= overpressure:
        # print(p)
        return (a * q - b * (p - p0) ** 2)

    return (a * q)

# implement an improved Euler step to solve the ODE
def solve_lpm(f, t0, t1, dt, x0, pars, scale=1.):

    def step_ieuler(f, tk, yk, h, args=None, scale=1.):


        # Solve q first
        def qsolve1(tk, scale=1.):
            q = interpolate_kettle_heatsource(tk, scale)
            return q

        if args is None:
            args = []
        # Find k1
        q = qsolve1(tk, scale)
        k1 = f(tk, yk, q, *args)
        # Take a forward step
        fake_step = yk + h * k1
        # find k2
        q = qsolve1(tk + h, scale)
        k2 = f(tk + h, fake_step, q, *args)
        # approximate solution value for next step
        yk_1 = yk + 0.5 * h * (k1 + k2)
        return yk_1

    if pars is None:
        pars = []

    t = np.linspace(t0, t1, int((t1 - t0) / dt) + 1)

    x = [x0]
    t[0] = t0
    t[-1] = t1
    xstart = x0
    for ind, i in enumerate(t):
        # initial value already stored so can skip
        if ind == 0:
            continue
        else:
            # Uses improved euler
            xk_1 = step_ieuler(f, i - dt, x0, dt, pars, scale)
            x0 = xk_1
            x.append(xk_1)

    return t, x            # interp onto requested times

def interpolate_kettle_heatsource(t, scale=1.):

    time , mass = np.genfromtxt( 'gs_mass.txt' , delimiter=',', skip_header = 1 ).T
    secondsPerMonth = 2628288 #average month
    q = scale * mass / secondsPerMonth

    q = q.tolist()
    time = time.tolist()
    q.extend(q)
    time2 = [x + 10 for x in time]
    time.extend(time2)


    for i in range(len(time)):
        if t <= time[i]:
            return q[i]

def fit_mvn(parspace, dist):
    """Finds the parameters of a multivariate normal distribution that best fits the data
    Parameters:
    -----------
        parspace : array-like
            list of meshgrid arrays spanning parameter space
        dist : array-like
            PDF over parameter space
    Returns:
    --------
        mean : array-like
            distribution mean
        cov : array-like
            covariance matrix
    """

    # dimensionality of parameter space
    N = len(parspace)

    # flatten arrays
    parspace = [p.flatten() for p in parspace]
    dist = dist.flatten()

    # compute means
    mean = [np.sum(dist*par)/np.sum(dist) for par in parspace]

    # compute covariance matrix
        # empty matrix
    cov = np.zeros((N,N))
        # loop over rows
    for i in range(0,N):
            # loop over upper triangle
        for j in range(i,N):
                # compute covariance
            cov[i,j] = np.sum(dist*(parspace[i] - mean[i])*(parspace[j] - mean[j]))/np.sum(dist)
                # assign to lower triangle
            if i != j: cov[j,i] = cov[i,j]

    return np.array(mean), np.array(cov)