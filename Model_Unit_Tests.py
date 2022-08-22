from benchmarking import *
from math import isclose

def test_1():
    # Expected parameter values
    analytical_solution, numerical_solution = benchmarking(1,1,-1,0)
    if isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0):
        print("Test 1 Passed")
    else:
        print("Test 1 Failed")

def test_2():
    # Expected parameter values
    analytical_solution, numerical_solution = benchmarking(5,1,-1,1)
    if isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0):
        print("Test 2 Passed")
    else:
        print("Test 2 Failed")

def test_3():
    analytical_solution, numerical_solution = benchmarking(1,1,-1,100)
    if isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0):
        print("Test 3 Passed")
    else:
        print("Test 3 Failed")

def test_4():
    analytical_solution, numerical_solution = benchmarking(0,1,-1,1)
    if isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0):
        print("Test 4 Passed")
    else:
        print("Test 4 Failed")
