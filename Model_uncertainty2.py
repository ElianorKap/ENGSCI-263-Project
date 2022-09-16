
# INSTRUCTIONS:
# Jump to section "if __name__ == "__main__":" at bottom of this file.


from plotting2 import *
import matplotlib.pyplot as plt
import pandas as pd
from JuneUncertainty import *

def grid_search():
    ''' This function implements a grid search to compute the posterior over a and b.

        Returns:
        --------
        a : array-like
            Vector of 'a' parameter values.
        b : array-like
            Vector of 'b' parameter values.
        P : array-like
            Posterior probability distribution.
    '''

    # Define parameter ranges for the grid search
    a_best = 18.1
    b_best = 0.026

    # Number of values considered for each parameter within a given interval
    N = 10

    # Vectors of parameter values
    a = np.linspace(a_best /2 ,a_best *1.5, N)
    b = np.linspace(b_best /2 ,b_best *1.5, N)

    # Grid of parameter values: returns every possible combination of parameters in a and b
    A, B = np.meshgrid(a, b, indexing='ij')

    # Empty 2D matrix for objective function
    S = np.zeros(A.shape)

    # Data for calibration
    tp ,po = np.genfromtxt('gs_pres.txt', delimiter = ',', skip_header = 1).T

    # Error variance
    v = 0.15

    # Grid search algorithm
    for i in range(len(a)):
        for j in range(len(b)):
            # Compute the sum of squares objective function at each value
            t, pm = solve_lpm(lpm, 2009, 2019, 0.25, 25.16, [10.87, a[i], b[j]])
            S[i ,j] = np.sum((po -pm )**2 ) /v

    # Compute the posterior
    P = np.exp(- S /2.)

    # Normalize to a probability density function
    Pint = np.sum(P ) *(a[1 ] -a[0] ) *(b[1 ] -b[0])
    P = P/ Pint

    # Plot posterior parameter distribution
    # plot_posterior(a, b, P=P)
    # plt.savefig('posterior meshgrid without samples')
    # plt.show()

    return a, b, P


def construct_samples(a, b, P, N_samples):
    ''' Constructs samples from a multivariate normal distribution fitted to the data.

        Parameters:
        -----------
        a : array-like
            Vector of 'a' parameter values.
        b : array-like
            Vector of 'b' parameter values.
        P : array-like
            Posterior probability distribution.
        N_samples : int
            Number of samples to take.

        Returns:
        --------
        samples : array-like
            parameter samples from the multivariate normal
    '''

    # Compute properties (fitting) of multivariate normal distribution
    # Mean = a vector of parameter means
    # Covariance = a matrix of parameter variances and correlations
    A, B = np.meshgrid(a, b, indexing='ij')
    mean, covariance = fit_mvn([A, B], P)

    # Create samples using numpy function multivariate_normal
    samples = np.random.multivariate_normal(mean, covariance, N_samples)

    # Plot samples and predictions
    # plot_samples(a, b, P=P, samples=samples)
    # plt.savefig('meshgrid with samples')
    # plt.show()
    return samples


def model_ensemble(samples):
    ''' Runs the model for given parameter samples and plots the results.

        Parameters:
        -----------
        samples : array-like
            parameter samples from the multivariate normal

        Returns:
        --------
        None
    '''

    # Setting time vector to evaluate model between
    t = np.linspace(2009, 2019, 101)

    f, ax = plt.subplots(1, 1, figsize=(12, 6))

    # For each sample, the model is solved and plotted
    for a, b in samples:
        t, pm = solve_lpm(lpm, 2009, 2019, 0.1, 25.16, [10.87, a, b])
        ax.plot(t, pm, 'k-', lw=0.25, alpha=0.2)
    ax.plot([], [], 'k-', lw=0.5, alpha=0.4, label='model ensemble')

    # Extracting historical data
    tp, po = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T
    ax.axhline(25.16, color='g', linestyle=':', label='overpressure')

    # Plotting data as error bars
    v = 0.15
    ax.errorbar(tp, po, yerr=v, fmt='ro', label='data')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_title(' Pressure Variation in Ahuroa Resevoir')
    ax.legend(loc='upper left', prop={'size': 15})
    plt.savefig('Uncertainty plot 1')
    plt.show()


def model_ensemble_with_forecasts(samples):
    ''' Finds the pressure at different capacities for given parameter samples, and plots the results

        Parameters:
        -----------
        samples : array-like
            parameter samples from the multivariate normal
        Returns:
        --------
        None

    '''

    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Extracting historical data
    tp, po = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T

    # For each sample, solving and plotting model at current capacity
    for a, b in samples:
        t1, pm = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b])
        t1, pm3 = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b], scale=2.)
        t1, pm4 = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b], scale=1.2)
        t1, pm5 = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b], scale=1.5)

        timelength = len(t1)
        middle_index = int(timelength / 2)

        ax1.plot(t1, pm, 'k-', lw=0.25, alpha=0.2)
        ax1.plot(t1[middle_index:], pm3[middle_index:], 'b-', lw=0.25, alpha=0.2)
        ax1.plot(t1[middle_index:], pm4[middle_index:], 'm-', lw=0.25, alpha=0.2)
        ax1.plot(t1[middle_index:], pm5[middle_index:], 'y-', lw=0.25, alpha=0.2)

    ax1.plot([], [], 'k-', lw=0.5, alpha=0.4, label='model ensemble')
    ax1.plot([], [], 'b-', lw=0.5, alpha=0.4, label='scale: 2')
    ax1.plot([], [], 'm-', lw=0.5, alpha=0.4, label='scale: 1.2')
    ax1.plot([], [], 'y-', lw=0.5, alpha=0.4, label='scale: 1.5')
    ax1.axvline(2019, color='b', linestyle=':', label='calibration/forecast')
    ax1.axhline(25.16, color='g', linestyle=':', label='overpressure')

    v = 0.15
    ax1.errorbar(tp, po, yerr=v, fmt='ro', markersize=2, label='data')
    ax1.set_title('Forecasted Pressure Variation in Ahuroa Resevoir')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.legend(prop={'size': 15})

    plt.savefig('Uncertainty plot 2')
    plt.show()


def leakage_forecasting(samples):
    ''' Finds the cumulative pressure change at different capacities for given parameter samples,
    and plots forecasted results.

        Parameters:
        -----------
        samples : array-like
            parameter samples from the multivariate normal

        Returns:
        --------
        None
    '''

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    overpressure = 25.16

    cumulLeak1 = list()
    count = 0

    # For each sample, solving model and plotting cumulative pressure change at current capacity

    for a, b in samples:
        t, pm = solve_lpm(lpm, 2009, 2029, 0.1, 25.16, [10.87, a, b])
        t = t.tolist()
        last_index = t.index(2019)

        # Solving cumulative pressure change
        dleakage1 = gasLeakage(t, pm, overpressure, b)
        listvals = integralFunc(t, dleakage1)
        listvals = list(listvals)

        cumulLeak1.append(listvals)
        ax2.plot(t, cumulLeak1[count], 'k-', lw=0.25, alpha=0.2)

        count += 1

    # Finding minimum possible pressure at the end of historical period
    minval = 0
    for i in range(count):
        if cumulLeak1[i][last_index] < minval:
            minval = cumulLeak1[i][last_index]

    # Finding maximum possible pressure at the end of historical period
    maxval = -100
    for i in range(count):
        if cumulLeak1[i][last_index] > maxval:
            maxval = cumulLeak1[i][last_index]

    # Finding expected pressure at the end of historical period
    expected_p = (minval + minval) / 2

    # For each sample, solving model and plotting cumulative pressure change at different capacities

    # Initialising ranges for each leakage scenario after 20 years since the beginning of the time domain
    s1min = 100
    s1max = -100

    s2min = 100
    s2max = -100

    s3min = 100
    s3max = -100

    s4min = 100
    s4max = -100

    for a, b in samples:
        x0 = 25.16
        pars = [p0, a, b]

        # Solving model at different scales
        modelATime, model1P = solve_ode_kettle(ode_model, 2009., 2019., 0.1, x0, pars, scale=1.)
        modelATime, model2P = solve_ode_kettle(ode_model, 2009., 2019., 0.1, x0, pars, scale=2.)
        modelATime, modelAP = solve_ode_kettle(ode_model, 2009., 2019., 0.1, x0, pars, scale=1.2)
        modelATime, modelBP = solve_ode_kettle(ode_model, 2009., 2019., 0.1, x0, pars, scale=1.5)

        # Calculating cumulative pressure change
        dleakage1 = gasLeakage(modelATime, model1P, overpressure, b)
        dleakage2 = gasLeakage(modelATime, model2P, overpressure, b)

        cumulLeak1 = integralFunc(modelATime, dleakage1)
        cumulLeak2 = integralFunc(modelATime, dleakage2)

        dleakageA = gasLeakage(modelATime, modelAP, overpressure, b)
        cumulLeakA = integralFunc(modelATime, dleakageA)
        cumulLeakA = cumulLeakA + cumulLeak1[-1]
        dleakageB = gasLeakage(modelATime, modelBP, overpressure, b)
        cumulLeakB = integralFunc(modelATime, dleakageB)
        cumulLeakB = cumulLeakB + cumulLeak1[-1]
        cumulLeakF1 = [*cumulLeak1, *(np.array(cumulLeak1) + cumulLeak1[-1])]
        cumulLeakF2 = [*cumulLeak1, *(np.array(cumulLeakA))]
        cumulLeakF3 = [*cumulLeak1, *(np.array(cumulLeakB))]
        cumulLeakF4 = [*cumulLeak1, *(np.array(cumulLeak2) + cumulLeak1[-1])]
        modelFTime = [*modelATime, *(np.array(modelATime) + modelATime[-1] - modelATime[0])]

        # Plotting cumulative pressure change
        ax2.plot(modelFTime, cumulLeakF1, 'k-', lw=0.25, alpha=0.2)
        ax2.plot(modelFTime, cumulLeakF2, 'b-', lw=0.25, alpha=0.2)
        ax2.plot(modelFTime, cumulLeakF3, 'r-', lw=0.25, alpha=0.2)
        ax2.plot(modelFTime, cumulLeakF4, 'g-', lw=0.25, alpha=0.2)


        # Calculating minimum and maximum final leakage value for each operation capacity scenario
        if cumulLeakF1[-1]  < s1min:
            s1min = cumulLeakF1[-1]

        if cumulLeakF1[-1]  > s1max:
            s1max = cumulLeakF1[-1]

        if cumulLeakF2[-1]  < s2min:
            s2min = cumulLeakF2[-1]

        if cumulLeakF2[-1]  > s2max:
            s2max = cumulLeakF2[-1]

        if cumulLeakF3[-1]  < s3min:
            s3min = cumulLeakF3[-1]

        if cumulLeakF3[-1]  > s3max:
            s3max = cumulLeakF3[-1]

        if cumulLeakF4[-1]  < s4min:
            s4min = cumulLeakF4[-1]

        if cumulLeakF4[-1] > s4max:
            s4max = cumulLeakF4[-1]


    # Plotting cumulative pressure change
    ax2.plot([], [], 'k-', lw=0.5, alpha=0.4, label='model ensemble')
    ax2.plot([], [], 'b-', lw=0.5, alpha=0.4, label='scale: 1.2')
    ax2.plot([], [], 'r-', lw=0.5, alpha=0.4, label='scale: 1.5')
    ax2.plot([], [], 'g-', lw=0.5, alpha=0.4, label='scale: 2')
    ax2.axvline(2019, color='b', linestyle=':', label='calibration/forecast')

    # Printing minimum and maximum final leakage value for each operation capacity scenario

    print("Minimum and maximum leakage value for scenario 1")
    print(s1min)
    print(s1max)
    print("Minimum and maximum leakage value for scenario 2")
    print(s2min)
    print(s2max)
    print("Minimum and maximum leakage value for scenario 3")
    print(s3min)
    print(s3max)
    print("Minimum and maximum leakage value for scenario 4")
    print(s4min)
    print(s4max)

    # Setting axes title and labels
    ax2.set_title('What-if Scenarios With Uncertainty')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Pressure (MPa)')
    ax2.legend(prop={'size': 15})

    plt.savefig('Uncertainty plot 3')
    plt.show()


def create_histograms(samples):
    ''' Creates frequency density histograms for sampled values of parameters a and d.

        Parameters:
        -----------
        samples : array-like
            parameter samples from the multivariate normal

        Returns:
        --------
        None

    '''

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig3, ax3 = plt.subplots()

    # Extracting parameter a samples from the multivariate normal
    x1 = samples[:, 0]
    x1 = x1.tolist()

    # Converting list of parameter values into dataframe
    df1 = pd.DataFrame(x1)

    # Creating empty histogram for plotting frequency density
    num_bins = 30
    n, bins, patches = plt.hist(x1, num_bins)

    # Finding and plotting 95% confidence interval marker for parameter a
    lower = df1.quantile(0.05)
    upper = df1.quantile(0.95)
    mid = df1.quantile(0.5)

    ax3.axvline(lower[0], color='r', linestyle=':')
    ax3.axvline(upper[0], color='r', linestyle=':')

    # PLotting axes title and labels
    ax3.set_title("Frequency Density plot for Parameter a")
    ax3.set_xlabel('Parameter a')
    ax3.set_ylabel('Frequency density')
    plt.savefig('Histogram a')
    plt.show()

    fig4, ax4 = plt.subplots()
    x2 = samples[:, 1]
    x2 = x2.tolist()
    df2 = pd.DataFrame(x2)

    # Creating empty histogram for plotting frequency density
    num_bins = 30
    n, bins, patches = plt.hist(x2, num_bins)

    # Finding and plotting 95% confidence interval marker for parameter d
    lower = df2.quantile(0.05)
    upper = df2.quantile(0.95)
    mid = df2.quantile(0.5)
    ax4.axvline(lower[0], color='r', linestyle=':')
    ax4.axvline(upper[0], color='r', linestyle=':')

    # PLotting axes title and legend
    ax4.set_title("Frequency Density plot for Parameter d")
    ax4.set_xlabel('Parameter d')
    ax4.set_ylabel('Frequency density')
    plt.savefig('Histogram b')
    plt.show()


def histogram_plots():
    ''' Plots frequency density histograms for 10000 samples of parameters a and d.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

    '''

    # Setting number of samples
    N = 10000

    a, b, posterior = grid_search()

    # Collecting sampled values for parameters
    samples = construct_samples(a, b, posterior, N)
    create_histograms(samples)


def present_plots():
    ''' Combines histogram plots for parameter space, and model ensembles for current and forecasted leakage rates.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

    '''
    histogram_plots()
    a, b, posterior = grid_search()

    # Setting number of samples
    N = 10

    # Collecting sampled values for parameters
    samples = construct_samples(a, b, posterior, N)

    # Creating model ensemble over the time period of [2009, 2019]
    model_ensemble(samples)

    # Creating model ensemble plot with predicted forecasts
    model_ensemble_with_forecasts(samples)

    # Creating leakage model ensemble plot with predicted forecasts
    leakage_forecasting(samples)


if __name__ == "__main__":
    present_plots()


