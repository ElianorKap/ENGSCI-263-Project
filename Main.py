if __name__ == '__main__':
    from benchmarking import *
    from June_sdlab_functions import *
    from JuneUncertainty import *
    from lab2_curve_fitting import *
    from Model_Unit_Tests import *
#1. Unit tests
    test_1()
    test_2()
    test_3()
    test_4()
# 2. Benchmarking
    benchmarking()
# 3. Model Calibration
    plot_kettle_model()
# 4. Model Uncertainty
    main()
