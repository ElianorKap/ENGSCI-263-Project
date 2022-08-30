from benchmarking import *
from math import isclose
import statistics
from q_array import *

# Initalsing default input arguments (equal to those used in our model)
a = 18.1
B = 0.026
p0 = 10.87
q = statistics.mean(find_q())

def test_1():
    ''' Testing input arguments with values equal to those used in the model

            Parameters:
            -----------
            None

            Returns:
            --------
            Nonw
    '''
    analytical_solution, numerical_solution = benchmarking(a, B, p0, q, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 1 Passed')

def test_2():
    ''' Testing edge case where a = 0, keeping remaining parameters the same

            Parameters:
            -----------
            None

            Returns:
            --------
            None
    '''
    analytical_solution, numerical_solution = benchmarking(0,B,0,q, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 2 Passed')

def test_3():
    ''' Testing edge case where B tends towards 0, keeping remaining parameters the same

            Parameters:
            -----------
            None

            Returns:
            --------
            None
    '''
    analytical_solution, numerical_solution = benchmarking(a, 0.001, p0, q, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 3 Passed')


def test_4():
    ''' Testing edge case where p0 = 0, keeping remaining parameters the same

            Parameters:
            -----------
            None

            Returns:
            --------
            None
    '''
    analytical_solution, numerical_solution = benchmarking(a, B, 0, q, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 4 Passed')

def test_5():
    ''' Testing edge case where q = 0, keeping remaining parameters the same

            Parameters:
            -----------
            None

            Returns:
            --------
            None
    '''
    analytical_solution, numerical_solution = benchmarking(a, B, p0, 0, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 5 Passed')

