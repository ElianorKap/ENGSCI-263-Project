if __name__ == '__main__':

# Global variables for benchmarking input paramters
    from q_array import *
    import statistics
    a = 18.1
    B = 0.026
    p0 = 10.87
    q = statistics.mean(find_q())

#1. Unit tests
    from Model_Unit_Tests import *
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()

# 2. Benchmarking
    from benchmarking import *
    benchmarking(a, B, q, p0, True)
                  
# 3.0 Model Calibration -> Pre manual tweaks
    from lab2_curve_fitting import *
    plot_kettle_model()
# 3.5 Model Calibration -> Post manual tweaking
    from practice import *
    plot_kettle_model()
    
# 4. Predictions
    from June_sdlab_functions import *
    from JuneUncertainty import *
    main(False, False, True, True, True)
    
#5. Uncertainty analysis
    from Model_uncertainty import *
    present_plots()