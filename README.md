# ENGSCI 263 Project: Gas Storage


## Overview


FlexGas have retained us to conduct a computer modelling study of the Natural Gas Storage at Ahuroa, Taranaki, that measures the pressure changes, 
current and future gas leakage and the rate at which the 6 PJ of cushion gas (current storage) is being eroded. 

Through this computer modelling study, we want to create a model that helps us quantify the current and future gas leakage in response to different 
pressure changes. We will achieve this by quantifying the monthly mass flow rates into and out of the reservoir combined with numerous physical 
principles. Additionally, we wish to estimate the quantity of gas leakage needed to achieve each outcome of the consent to double the capacity. 
Through this model, we will also get immediate insight into the trend of customer demand for natural gas and optimise injection/production rate to meet 
this demand.

We will perform unit testing, benchmarking and calibration to ensure our model matches the actual underlying process of the pressure change and gas 
leakage of the Natural Gas Storage at Ahuroa, Taranaki. We will then use this model to make accurate predictions while considering the uncertainty of 
our data. Also, provide recommendations for FlexGas to make the most financially optimal decision while eliminating or reducing gas leakage to a minimum 
to prevent local farmer's properties from catching fire. This repository will contain functions that produce all the figures needed to complete the required analysis.

## Key Tasks

This repository will contain python functions to perform the following tasks:

* Main
  * `Main.py` &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;&ensp;- this function will produce all the model figures in the group projects
* Unit Testing
  * `Model_Unit_Testing.py`&nbsp;&ensp;&ensp;- this function will compare the output of a function to a worked solution
* Benchmarking
  * `benchmarking.py`&emsp;&emsp;&emsp;&ensp;&nbsp;&ensp;- this function will solve the ODE analytically under a simplified condition and verify the
  numerical solver by returning a similar solution for the same parameters
* Calibration
  * `lab2_curve_fitting.py`&emsp;&nbsp;- this function will help us choose good parameter values for the model such that it is a good approximation of
  reality
  * `task2.py`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;- this function is used to test the sensitivity of the parameter and help us get a rough 
  set of parameter value
* Prediction 
  * `JuneUncertainty.py`&emsp;&emsp;&nbsp;&ensp;- this function will perform calculation and generate plots for prediction based on different scenario 
  such as initial gas leakage and doubling the pressure capacity etc
* Uncertainty
  * `ellie_justin_uncertainty.py` - this function will also quantify and plot the
  uncertainty of the model as well as constructing a future forecast
  * `ellie_lumped_parameter_model.py` -
  * `ellie_plotting.py`&emsp;&emsp;&nbsp;&ensp;&nbsp; - 


Others

* `June_sdlab_functions.py`&emsp;&ensp;&nbsp; - this function is used to calculate the interpolation values
* `gs_mass.txt`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&ensp;&nbsp; - historical mass data file
* `gs_pres.txt`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp; - historical pressure data file
* `practice.py`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp; - practice function
* `modelFitImproved.png`&emsp;&emsp;&emsp;&nbsp; - image of the improved model fit

## Instructions  


1. Open terminal at a folder that you will use for this project
2. Use the `git clone` command to clone this repository into that folder
3. Open `Main.py` 
4. Within the `Main.py` function, you will need to input values for benchmarking() and main() functions in order to plot the figures within those functions.
5. For benchmarking() function, you can input the following parameters 18.1,0.026,25.6,10.87
6. For main() function, you will need to input False,True,True,True as the parameters
7. Run `Main.py` which will produce all the model figures used in the group projects.
