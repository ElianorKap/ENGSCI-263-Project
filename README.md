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

## Libraries Used (REQUIRED TO RUN Main.py)

* `numpy`
* `matplotlib` - pyplot
* `matplotlib` - cm
* `math` - isclose
* `pandas`
* `statistics`
* `scipy.optimize` - curve_fit 
* `mpl_toolkits.mplot3d` - Axes3D
* `cProfile` - label

## Key Tasks

This repository will contain python functions to perform the following tasks:

* Main
  * `Main.py` &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;&ensp;- this function will produce all the model figures in the group projects
* Unit Testing
  * `Model_Unit_Testing.py`&nbsp;&ensp;&ensp;- this function will compare the output of a function to a worked solution
* Benchmarking
  * `benchmarking.py`&emsp;&emsp;&emsp;&ensp;&ensp;&ensp;- this function will solve the ODE analytically under a simplified condition and verify the
  numerical solver by returning a similar solution for the same parameters
* Calibration
  * `lab2_curve_fitting.py`&emsp;&nbsp;- this contains a curve fitting function which will help us choose good parameter values for the model such that it is a good approximation of
  reality
  * `newpractice.py`&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp; - Generates plot of our finalised parameter estimates upon having tweaked curve fitting output.
  set of parameter value
* Prediction 
  * `JuneUncertainty.py`&emsp;&emsp;&nbsp;&ensp;- this function will perform calculation and generate plots for prediction based on different scenario 
  such as initial gas leakage and doubling the pressure capacity etc
* Uncertainty
  * `Model_uncertainty.py`&emsp;&ensp; - this function will also quantify and plot the uncertainty of the model as well as constructing a future forecast
  * `lpm3.py`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; - this function contains the functions necessary for calculating the values of uncertainty
  * `plotting2.py`&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp; - this function contains the function necessary for plotting the uncertainty
  * `Plot_histogram.py`&emsp;&emsp;&emsp; - this function contains the function necessary for plotting the parameter frequency density 


Additional:

* `June_sdlab_functions.py`&emsp;&ensp;&nbsp; - this function is used to calculate the interpolation values
* `gs_mass.txt`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&ensp;&nbsp; - historical mass data file
* `gs_pres.txt`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp; - historical pressure data file
* `modelFitImproved.png`&emsp;&emsp;&emsp;&nbsp; - image of the improved model fit
* `q_array.py`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp; - this function finds mass rate at times over the injection/extraction period
* `Benchmark plot 1.png`&emsp;&emsp;&emsp;&nbsp; - image of the benchmarking plot 
* `Benchmark plot 2.png`&emsp;&emsp;&emsp;&nbsp; - image of the timestep convergence plot 
* `Benchmark plot 3.png`&emsp;&emsp;&emsp;&nbsp; - image of error analysis plot 
* `Histogram a.png`&emsp;&emsp;&emsp;&nbsp;&emsp;&emsp;&ensp; - histogram plot for parameter 'a'
* `Histogram b.png`&emsp;&emsp;&emsp;&nbsp;&emsp;&emsp;&ensp; - histogram plot for parameter 'd'
* `Observation_plots.py`&emsp;&emsp;&emsp;&nbsp; - Plot to view their pressure and mass observation data
* `Uncertainty plot 1.png`&emsp;&nbsp;&ensp;&ensp; - image of the uncertainty plot
* `Uncertainty plot 2.png`&emsp;&nbsp;&ensp;&ensp; - image of the uncertainty plot
* `lpm2.py`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; - copy of `lpm3.py` 
* `practice.py`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; - copy of `newpractice.py`
* `meshgrid with samples.png`&ensp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp; - plot of meshgrid with samples
* `posterior meshgrid without samples.png`&emsp;&emsp;&emsp;&emsp; - image of posterior meshgrid
* `posterior meshgrid without samples 2.png`&emsp;&emsp;&emsp; - image of posterior meshgrid2


## Instructions  


1. Open terminal at a folder that you will use for this project
2. Use the `git clone` command to clone this repository into a folder of choice
3. Open `Main.py` 
4. Run `Main.py` which will produce all the model figures used in the group projects.

## Ownership

- Ayaan Saiyad
- Elianor Kapelevich
- June Terzaghi
- Yan Liu
- Alex Xie
