# ENGSCI233: Lab - Sampled Data
# sdlab_functions.py

# PURPOSE:
# To IMPLEMENT cubic spline interpolation.

# PREPARATION:
# Notebook sampling.ipynb, ESPECIALLY Section 1.3.1 theory of cubic splines.

# SUBMISSION:
# - YOU MUST submit this file to complete the lab. 
# - DO NOT change the file name.

# TO DO:
# - COMPLETE the functions spline_coefficient_matrix(), spline_rhs() and spline_interpolation().
# - COMPLETE the docstrings for each of these functions.
# - TEST each method is working correctly by passing the asserts in sdlab_practice.py.
# - DO NOT modify the other functions.

import numpy as np


# **this function is incomplete**
#					 ----------
def spline_coefficient_matrix(xi):
    ''' **complete the docstring**
    Takes input 1D array xi, of sampled data points
    returns matrix A, 2D array of polynomial coefficients to be used for calculating coefficients
    '''

    # create an array of zeros with the correct dimensions
    #   **what are the correct dimensions? how to determine this from xi?**
    #   **use np.zeros() to create the array**

    # calculates length of sampled data array to get number at data points
    n = len(xi)
    # calculates amount of rows for matrix from required rows of equations
    lengthCoMatrix = 2*(n-1)+2*(n-2)+2
    # creates matrix of zero values
    matrix = np.zeros((lengthCoMatrix, 4 * (n - 1)))

    # Loop over the subintervals, add matrix coefficients for equations:
    # - polynomial passes through lefthand point of subinterval
    # - polynomial passes through righthand point of subinterval
    #   **how many subintervals should there be (in terms of length of xi)?**
    #   **how to modify loop index so it jumps along a row in increments of 4?**
    #   **how to define width of the subinterval in terms of indices of xi?**
    #   **what values go into matrix A and how do they relate to subinterval width?**

    # loops over each polynomial, filling values for fitting cubic to sampled data
    for i in range(n-1):
        # inputs values for appropriate polynomial matching first value of subinterval
        matrix[2*i,4*i] = 1
        # inputs values for appropriate polynomial matching second value of subinterval
        matrix[2*i+1,4*i:4*(i+1)] = [1, (xi[i+1]-xi[i]), (xi[i+1]-xi[i])**2,(xi[i+1]-xi[i])**3]

    # Loop over neighbouring subintervals, add matrix coefficients for equations:
    # - polynomial gradient continuous at shared point
    # - polynomial second derivative continuous at shared point
    #   **how many shared points should there be (in terms of length of xi)?**
    #   **what values go into matrix A and how do they relate to subinterval width?**

    # loops over each shared derivatives between polynomials
    for i in range(n-2):
        # inputs values for appropriate polynomial matching first derivative
        matrix[2*i+2*(n-1),4*i:4*(i+2)] = [0,1,2*(xi[i+1]-xi[i]),3*(xi[i+1]-xi[i])**2,0,-1,0,0]
        # inputs values for appropriate polynomial matching second derivative
        matrix[2*i+1+2*(n-1),4*i:4*(i+2)] = [0,0,2,6*(xi[i+1]-xi[i]),0,0,-2,0]

    # For the beginning and end points, add matrix coefficients for equations:
    # - the polynomial second derivative is zero

    # fills second to last row with making second derivative at first data point 0
    matrix[-2,0:4] = [0,0,2,0]
    # fills last row with making second derivative at last data point 0
    matrix[-1,-2] = 2
    matrix[-1,-1] = 6*(xi[-1]-xi[-2])

    # returns coefficient matrix for solving
    return matrix


# **this function is incomplete**
#					 ----------
def spline_rhs(xi, yi):
    ''' **complete the docstring**
    inputs:
    xi - 1D array of x values for sampled data
    yi - 1D array of y values for sampled data
    output - 1D array of right hand side matrix for solving polynomial coefficients
    '''
    # **use structure of spline_coefficient_matrix() as a guide for
    #   completing this function**

    # calculates length of sampled data array to get number at data points
    n = len(xi)
    # calculates amount of rows for matrix from required rows of equations
    lengthCoMatrix = 2 * (n - 1) + 2 * (n - 2) + 2
    # creates empty matrix of appropriate length
    b = np.zeros(lengthCoMatrix)
    # loops over each polynomial spline
    for i in range(n-1):
        # sets value to y value for first subinterval value
        b[2*i] = yi[i]
        # sets value to y value for second subinterval value
        b[2*i+1] = yi[i+1]

    # delete this command once you have written your code

    # returns 1D matrix of right hand side values
    return b


# **this function is incomplete**
#					 ----------
def spline_interpolate(xj, xi, ak):
    ''' **complete the docstring**
        inputs:
        xj - 1D array of x values for interpolation points
        xi - 1D array of x values for sampled data
        ak - matrix of solved polynomial coefficients
        output - 1D array of y values for interpolation points
        Notes
        -----
        You may assume that the interpolation points XJ are in ascending order.
        Evaluate polynomial using polyval function DEFINED below.
    '''

    # Suggested strategy (you could devise another).
    # 1. Initialise FIRST subinterval (and polynomial) as CURRENT subinterval (and polynomial).
    # 2. FOR each interpolation point.
    # 3. WHILE interpolation point NOT inside CURRENT subinterval, iterate
    #    to NEXT subinterval (and polynomial).
    # 4. Evaluate CURRENT polynomial at interpolation point.
    # 5. RETURN when all interpolation points evaluated.

    # creates tracking variable for subinterval used
    i = 0
    # creates tracking variable for what index to place values into output array
    j = 0
    # creates array of zeros same length as input interpolation point array
    yj = np.zeros(len(xj))
    # loops over each interpolation point x value
    for z in xj:
        # if interpolation point is outside of current subinterval, updates until is in
        # correct subinterval
        while z >= xi[i+1]:
            i = i+1
        # obtains evaluation of cubic spline at interpolation point, and stores in array
        yj[j] = polyval(ak[4*i:4*(i+1)],(z-xi[i]))
        # increments counter for input index
        j = j+1

    # returns y value array for interpolation points
    return yj


# this function is complete
def display_matrix_equation(A, b):
    ''' Prints the matrix equation Ax=b to the screen.

        Parameters
        ----------
        A : np.array
            Matrix.
        b : np.array
            RHS vector.

        Notes
        -----
        This will look horrendous for anything more than two subintervals.
    '''

    # problem dimension
    n = A.shape[0]

    # warning
    if n > 8:
        print('this will not format well...')

    print(' _' + ' ' * (9 * n - 1) + '_  _       _   _        _')
    gap = ' '
    for i in range(n):
        if i == n - 1:
            gap = '_'
        str = '|{}'.format(gap)
        str += ('{:+2.1e} ' * n)[:-1].format(*A[i, :])
        str += '{}||{}a_{:d}^({:d})'.format(gap, gap, i % 4, i // 4 + 1) + '{}|'.format(gap)
        if i == n // 2 and i % 2 == 0:
            str += '='
        else:
            str += ' '
        if b is None:  # spline_rhs has not been implemented
            str += '|{}{}{}|'.format(gap, 'None', gap)
        else:
            str += '|{}{:+2.1e}{}|'.format(gap, b[i], gap)
        print(str)


# this function is complete
def get_data():
    # returns a data vector used during this lab
    xi = np.array([2.5, 3.5, 4.5, 5.6, 8.6, 9.9, 13.0, 13.5])
    yi = np.array([24.7, 21.5, 21.6, 22.2, 28.2, 26.3, 41.7, 54.8])
    return xi, yi


# this function is complete
def ak_check():
    # returns a vector of predetermined values
    out = np.array([2.47e+01, -4.075886048665986e+00, 0., 8.758860486659859e-01, 2.15e+01,
                    -1.448227902668027e+00, 2.627658145997958e+00, -1.079430243329928e+00, 2.16e+01,
                    5.687976593381042e-01, -6.106325839918264e-01, 5.358287012458253e-01, 2.22e+01,
                    1.170464160078432e+00, 1.157602130119396e+00, -2.936967278262911e-01, 2.82e+01,
                    1.862652894849505e-01, -1.485668420317224e+00, 1.677900564431842e-01, 2.63e+01,
                    -2.825777017172887e+00, -8.312872001888050e-01, 1.079137281294699e+00, 4.17e+01,
                    2.313177016138269e+01, 9.204689515851896e+00, -6.136459677234598e+00])
    return out


# this function is complete
def polyval(a, xi):
    ''' Evaluates a polynomial.

        Parameters
        ----------
        a : np.array
            Vector of polynomial coefficients.
        xi : np.array
            Points at which to evaluate polynomial.

        Returns
        -------
        yi : np.array
            Evaluated polynomial.

        Notes
        -----
        Polynomial coefficients assumed to be increasing order, i.e.,

        yi = Sum_(i=0)^len(a) a[i]*xi**i

    '''
    # initialise output at correct length
    yi = 0. * xi

    # loop over polynomial coefficients
    for i, ai in enumerate(a):
        yi = yi + ai * xi ** i

    return yi
