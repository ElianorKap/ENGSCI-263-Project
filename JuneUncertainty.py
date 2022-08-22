
#imports
import numpy as np
import matplotlib.pyplot as plt
from June_sdlab_functions import *
from lab2_curve_fitting import *

#params:
overpressure = 25.6
a = 16.88353712877018
b = 0.057656759102383694
p0 = 18.72192256351264



def integralFunc(xj,yj):
    # creates 1D array of zeros, same length as input array of data
    integralA = np.zeros(len(xj))
    # loops over each data point
    for i in range(len(xj)):
        # skips first data point (assumes is zero)
        if i == 0:
            continue
        # uses trapezium approximation to estimate area under curve between current and
        # last data point, and adds it to running sum of area
        integralA[i]=((yj[i]+yj[i-1])/2)*(xj[i]-xj[i-1]) + integralA[i-1]
    # returns array of cumulative function of input data
    return integralA


def derivativeFunc(xj,yj):
    yi = yj.copy()
    for i in range(len(xj)-1):
        yi[i-1] = (yi[i]-yi[i-1])/(xj[i]-xj[i-1])
    yi[-1] = yi[-2]
    return yi

def main(Plot1,Plot2):
    PresTime, PresHist = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T
    MassTime1, MassHist = np.genfromtxt('gs_mass.txt', delimiter=',', skip_header=1).T
    MassTime2 = [2009.]
    while MassTime2[-1]+1/12<2018.95:
        MassTime2 = [*MassTime2,MassTime2[-1]+1/12]
    MassTime2 = [*MassTime2,2019.000001]
    # Pressure is quarterly, Mass is monthly
    MassCumul1 = integralFunc(MassTime2, MassHist)
    for i in range(len(MassHist)):
        if i == 0:
            MassCumul2 = np.array([MassHist[i]])
        else:
            MassCumul2 = [*MassCumul2,MassCumul2[-1]+MassHist[i]]
    # plot preliminary analysis
    if Plot1:
        fig1, ax1 = plt.subplots()
        P11 = a*MassCumul1/10.**6.
        P12 = a*np.array(MassCumul2)/10.**8.
        P2 = PresHist-PresHist[0]
        spline1 = np.linalg.solve(spline_coefficient_matrix(MassTime2), spline_rhs(MassTime2, P12))
        MassInter = spline_interpolate(PresTime, MassTime2, spline1)
        P3 = P2-MassInter
        P4 = derivativeFunc(PresTime,P3)
        ax1.plot(MassTime2,P12)
        ax1.plot(PresTime,P2)
        ax1.plot(PresTime,P3)
        ax1.plot(PresTime,P4)
        # plot shows positive derivative in gas leakage
        # implying gas leaks bag into reservoir - impossible
        plt.show()

    # add q scaling and analysis using model parameters
    # model plots
    if Plot2:
        fig2, ax2 = plt.subplots()
        pars = [p0, a, b]
        x0 = 25.16
        model1Time, model1P = solve_kettle_ode(ode_model, MassTime1, x0, pars, scale=1.)
        model2Time, model2P = solve_kettle_ode(ode_model, MassTime1, x0, pars, scale=2.)
        ax2.plot(model1Time, model1P)
        ax2.plot(model2Time, model2P)
        plt.show()

    return

if __name__ == '__main__':
    main(True,True)

