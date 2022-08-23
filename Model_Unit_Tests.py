from benchmarking import *
from math import isclose

def test_1():
    # Test 1:
    # a = 1, B = 1, q = -1, p0 = 0
    analytical_solution, numerical_solution = benchmarking(1,1,-1,0)
    if isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0):
        print("Test 1 Passed")
    else:
        print("Test 1 Failed")

def test_2():
    # Test 2:  Modifying parameter values a and B
    # a = 5, B = 0, q = -1, p0 = 0
    analytical_solution, numerical_solution = benchmarking(5,0,-1,0)
    if isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0):
        print("Test 2 Passed")
    else:
        print("Test 2 Failed")

def test_3():
    # Test 3:
    # Modifying initial value of solution: a = 1, B = 1, q = -1, p0 = 100
    analytical_solution, numerical_solution = benchmarking(1,1,-1,100)
    if isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0):
        print("Test 3 Passed")
    else:
        print("Test 3 Failed")

def test_4():
    # Test 4:  Edge case; setting parameter values a and B to 0
    # a = 0, B = 0, q = -1, p0 = 1
    analytical_solution, numerical_solution = benchmarking(0,0,-1,1)
    if isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0):
        print("Test 4 Passed")
    else:
        print("Test 4 Failed")
