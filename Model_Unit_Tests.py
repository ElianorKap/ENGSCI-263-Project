from benchmarking import *
from math import isclose
import statistics
from q_array import *

def test_1():
    # Testing input arguments with values equal to those used in the model
    a = 18.1
    B = 0.026
    p0 = 10.87
    q = statistics.mean(find_q())

    analytical_solution, numerical_solution = benchmarking(a, B, p0, q, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 1 Passed')


def test_2():
    analytical_solution, numerical_solution = benchmarking(1,1,-1,1, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 2 Passed')


def test_3():
    analytical_solution, numerical_solution = benchmarking(1,1,-1,100, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 3 Passed')


def test_4():
    analytical_solution, numerical_solution = benchmarking(0,1,-1,1, False)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)
    print('Test 4 Passed')

