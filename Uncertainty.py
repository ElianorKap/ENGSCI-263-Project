import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def ode_model(t, p, q, p0, a, b):
    overpressure = 25.6
    if p >= overpressure:
        # print(p)
        return (a * q - b * (p - p0) ** 2)

    return (a * q)

def solve_ode(f, t0, t1, dt, x0, pars, scale=1.):
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

    return t, x

def interpolate_kettle_heatsource(t, scale=1.):
    time, mass = np.genfromtxt('gs_mass.txt', delimiter=',', skip_header=1).T
    secondsPerMonth = 2628288  # average month
    q = scale * mass / secondsPerMonth

    for i in range(len(time)):
        if t <= time[i]:
            return q[i]


def plot_kettle_model():
    time, pressure = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T

    # plt.plot( time, pressure )
    a = 18.1
    b = 0.026
    p0 = 10.87

    t, p = solve_ode(ode_model, 2009, 2019, 0.1, 25.16, [p0, a, b])

    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title(f"a = {a} , b = {b} , overpressure = 25.6 , p0 ={p0}")
    ax.plot(time, pressure, 'k', label='observations')

    ax.plot(t, p, 'b-', label='model improved')
    ax.set_xlabel('time, $t$ [s]')
    ax.set_ylabel('Pressure, $MPa$')
    ax.legend()

    plt.show()

    sigma = [0.25]*len(pressure)  # uncertainty on observations
    p,cov = curve_fit(solve_ode, time, pressure, sigma=sigma)   # second output is covariance matrix


    ps = np.random.multivariate_normal(p, cov, 100)   # samples from posterior
    for pi in ps:
        ax.plot(t, f(t, *pi), 'k-', alpha=0.2, lw=0.5)
    ax.plot([], [], 'k-', lw=0.5, label='posterior samples')
    ax.legend()

if __name__ == "__main__":
    plot_kettle_model()