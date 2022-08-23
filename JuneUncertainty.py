
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
    # Calculates forwards derivative at each point
    # copies input array of independent variable
    yi = yj.copy()
    # loops over each value but last
    for i in range(len(xj)-1):
        # calculates forwards derivative
        yi[i-1] = (yi[i]-yi[i-1])/(xj[i]-xj[i-1])
    # last derivative value is same as second to last
    yi[-1] = yi[-2]
    # returns array of derivatives
    return yi

def gasLeakage(model1Time, model1P):
    # initialises array holding gas leakage values
    dleakage = []
    # loops over every element in time array
    for i in range(len(model1Time)):
        # if model pressure is above overpressure, calculates gas leakage rate
        if model1P[i] > overpressure:
            dleakage = [*dleakage, -b * (model1P[i] - overpressure) ** 2]
        # if model pressure is below overpressure gas leakage is 0
        else:
            dleakage = [*dleakage, 0.]
    # converts list to numpy array
    dleakage = np.array(dleakage)
    # returns array of gas leakage values
    return dleakage

def main(Plot1,Plot2, Plot3, Plot4):
    # inputs, 3 boolean, each matches to ordered section, section 1 is worthless
    # section 2 is model pressure
    # obtains historical data for pressure and mass flow time series from file
    PresTime, PresHist = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T
    MassTime1, MassHist = np.genfromtxt('gs_mass.txt', delimiter=',', skip_header=1).T
    # creates new time array for mass flow to account for monthly values
    MassTime2 = [2009.]
    # loops until end time is reached
    while MassTime2[-1]+1/12<2018.95:
        # adds 1/12 of a year to the previous time value to be next time value
        MassTime2 = [*MassTime2,MassTime2[-1]+1/12]
    # adds final time value slightly larger so integration function can work properly
    MassTime2 = [*MassTime2,2019.000001]
    # Pressure is quarterly, Mass is monthly
    # numerically integrations mass flow to calculate cumulative mass flow assumes rate
    MassCumul1 = integralFunc(MassTime2, MassHist)
    # loops over every value in mass flow data
    for i in range(len(MassHist)):
        if i == 0:
            # initialises numpy array for alternative cumulative mass flow variable
            MassCumul2 = np.array([MassHist[i]])
        else:
            # adds previous mass flow value to running total to fill current array value
            MassCumul2 = [*MassCumul2,MassCumul2[-1]+MassHist[i]]

    # plot preliminary analysis
    if Plot1:
        # creates figure and axes objects
        fig1, ax1 = plt.subplots()
        # converts mass to pressure
        P11 = a*MassCumul1/10.**6.
        # converts mass to pressure
        P12 = a*np.array(MassCumul2)/(1.*10.**8.)
        # converts from pressure to change in pressure from initial value
        P2 = PresHist-PresHist[0]
        P1x = P11
        # creates spline matrix for interpolation from alternate mass-pressure series
        spline1 = np.linalg.solve(spline_coefficient_matrix(MassTime2), spline_rhs(MassTime2, P1x))
        # interpolates alternate mass-pressure series
        MassInter = spline_interpolate(PresTime, MassTime2, spline1)
        # calculates theoretical cumulative gas leakage
        P3 = P2-MassInter
        # calculates derivative of gas leakage series
        P4 = derivativeFunc(PresTime,P3)
        # adds series to plot
        ax1.plot(MassTime2,P1x, label='Cumul Pressure~Mass')
        ax1.plot(PresTime,P2, label='Pressure changes')
        ax1.plot(PresTime,P3, label='Cumul Pressure~Leak')
        ax1.plot(PresTime,P4, label='Leak derivative')
        ax1.legend()
        # shows plot
        plt.show()
        # notes on plot
        # plot shows positive derivative in gas leakage
        # implying gas leaks bag into reservoir - impossible

    # creates ode parameter list
    pars = [p0, a, b]
    # creates variable for initial pressure
    x0 = 25.16
    # models pressure at q scale 1
    model1Time, model1P = solve_kettle_ode(ode_model, MassTime1, x0, pars, scale=1.)
    # models pressure at q scale 2
    model2Time, model2P = solve_kettle_ode(ode_model, MassTime1, x0, pars, scale=2.)
    # add q scaling and analysis using model parameters
    # model plots
    if Plot2:
        # creates figure and axes objects
        fig2, ax2 = plt.subplots()
        # adds scaled model pressures to plot
        ax2.plot(model1Time, model1P, label="Model s=1")
        ax2.plot(model2Time, model2P, label="Model s=2")
        ax2.plot(PresTime,PresHist, label='Historical')
        ax2.legend()
        # shows plot
        plt.show()
        # notes on plot

    # Scaled model gas leakage plots:
    if Plot3:
        # calculates gas leakage rates for scaled model series'
        dleakage1 = gasLeakage(model1Time, model1P)
        dleakage2 = gasLeakage(model2Time, model2P)
        # creates figure and axes objects
        fig3, ax3 = plt.subplots()
        ax3.plot(model1Time, model1P-model1P[0], label='Pressure Model')
        ax3.plot(PresTime, PresHist-PresHist[0], label='Pressure Historical')
        ax3.plot(model1Time, dleakage1, label="Leakage rate s=1")
        ax3.plot(model2Time, dleakage2, label="Leakage rate s=2")
        ax3.legend()
        plt.show()
        # calculates cumulate gas leakage
        cumulLeak1 = integralFunc(model1Time, dleakage1)
        cumulLeak2 = integralFunc(model2Time, dleakage2)
        # prints total gas leakage and increase
        print('Total gas leakage is (s=1): {}'.format(cumulLeak1[-1]))
        print('Total gas leakage is (s=2): {}'.format(cumulLeak2[-1]))
        print('Increase in leakage is: {}%'.format(cumulLeak2[-1]*100/cumulLeak1[-1]))

    # finds maximum scale factor to have 0 gas leakage
    if Plot4:
        # initialises loop variables
        loopBool = True
        loopScale = 1.
        # loops while boolean variable is True
        while loopBool:
            # models pressure at variable scale
            modelxTime, modelxP = solve_kettle_ode(ode_model, MassTime1, x0, pars, scale=loopScale)
            # calculates leakage rates for model
            dleakagex = gasLeakage(modelxTime, modelxP)
            # calculates cumulative gas leakage
            cumulLeakx = integralFunc(modelxTime, dleakagex)
            # if there is no gas leakage ends loop
            if cumulLeakx[-1] == 0.:
                loopBool = False
            #loopScale *= 0.9
            loopScale = (loopScale - 0.01)*0.95
            #loopScale = loopScale - 0.01
        # prints maximum scale at which 0 leakage is modelled
        print('Zero leakage scale is: {}'.format(loopScale))

    return

if __name__ == '__main__':
    main(False, False, True, True)

