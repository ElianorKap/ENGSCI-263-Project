
#imports
import numpy as np
import matplotlib.pyplot as plt
from June_sdlab_functions import *

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

def main():
    PresTime, PresHist = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T
    MassTime1, MassHist = np.genfromtxt('gs_mass.txt', delimiter=',', skip_header=1).T
    MassTime2 = [2009.]
    while MassTime2[-1]+1/12<2018.95:
        MassTime2 = [*MassTime2,MassTime2[-1]+1/12]
    MassTime2 = [*MassTime2,2019.000001]
    # Pressure is quarterly, Mass is monthly
    MassCumul = integralFunc(MassTime2, MassHist)

    # plot preliminary analysis
    if True:
        fig, ax = plt.subplots()
        P1 = a*MassCumul/10**6
        P2 = PresHist-PresHist[0]
        spline1 = np.linalg.solve(spline_coefficient_matrix(MassTime2), spline_rhs(MassTime2, P1))
        MassInter = spline_interpolate(PresTime, MassTime2, spline1)
        P3 = P2-MassInter
        P4 = derivativeFunc(PresTime,P3)
        ax.plot(MassTime2,P1)
        ax.plot(PresTime,P2)
        ax.plot(PresTime,P3)
        ax.plot(PresTime,P4)
        # plot shows positive derivative in gas leakage
        # implying gas leaks bag into reservoir - impossible
        plt.show()

    # add q scaling and analysis using model parameters

    return

if __name__ == '__main__':
    main()

