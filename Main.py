if __name__ == '__main__':

# Global variables for benchmarking input paramters
from q_array import *
t , p = np.genfromtxt( 'gs_pres.txt' , delimiter=',', skip_header = 1 ).T
a = 18.1
B = 0.026
p0 = 10.87
q = find_q()
    
#1. Unit tests
    from Model_Unit_Tests import *
    test_1()
    test_2()
    test_3()
    test_4()
    
# 2. Benchmarking
    from benchmarking import *
    benchmarking(t, p, q, a, B, p0)
                  
# 3. Model Calibration
    from practice import *
    plot_kettle_model() 
    
# 4. Predictions
    from June_sdlab_functions import *
    from JuneUncertainty import *
    main() 
    
#5. Uncertainty
    from Uncertainty import *
