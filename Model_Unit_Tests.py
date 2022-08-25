from benchmarking import *
from math import isclose

def test_1():
    analytical_solution, numerical_solution = benchmarking(1,1,-1,0)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)


def test_2():
    analytical_solution, numerical_solution = benchmarking(5,1,-1,1)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)


def test_3():
    analytical_solution, numerical_solution = benchmarking(1,1,-1,100)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)


def test_4():
    analytical_solution, numerical_solution = benchmarking(0,1,-1,1)
    assert isclose(analytical_solution, numerical_solution, rel_tol=1e-3, abs_tol=0.0)

