

# INSTRUCTIONS:
# Jump to section "if __name__ == "__main__":" at bottom of this file.

# import modules and functions
import numpy as np
from ellie_lumped_parameter_model import *
from ellie_plotting import *

def grid_search():
	''' This function implements a grid search to compute the posterior over a and b.

		Returns:
		--------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		P : array-like
			Posterior probability distribution.
	'''
	# **to do**
	# 1. DEFINE parameter ranges for the grid search
	# 2. COMPUTE the sum-of-squares objective function for each parameter combination
	# 3. COMPUTE the posterior probability distribution
	# 4. ANSWER the questions in the lab document

	# 1. define parameter ranges for the grid search
	a_best = 18.1
	b_best = 0.026

	# number of values considered for each parameter within a given interval
	N = 51

	# vectors of parameter values
	a = np.linspace(a_best/2,a_best*1.5, N)
	b = np.linspace(b_best/2,b_best*1.5, N)

	# grid of parameter values: returns every possible combination of parameters in a and b
	A, B = np.meshgrid(a, b, indexing='ij')

	# empty 2D matrix for objective function
	S = np.zeros(A.shape)

	# data for calibration
	tp,po = np.genfromtxt('gs_pres.txt', delimiter = ',', skip_header = 1).T

	# error variance - 2 bar
	v = 0.05

	# grid search algorithm
	for i in range(len(a)):
		for j in range(len(b)):
			# 3. compute the sum of squares objective function at each value 
			#pm =
			#S[i,j] =
			t, pm = solve_lpm(lpm, 2009, 2019, 0.25, 25.16, [10.87, a[i], b[j]])
			S[i,j] = np.sum((po-pm)**2)/v

	# 4. compute the posterior
	#P=
	P = np.exp(-S/2.)

	# normalize to a probability density function
	Pint = np.sum(P)*(a[1]-a[0])*(b[1]-b[0])
	P = P/Pint

	# plot posterior parameter distribution
	plot_posterior(a, b, P=P)

	return a,b,P
	
####################################################################################
#
# Task3: Open fun_with_multivariate_normals.py and complete the exercises.
#
####################################################################################
	
####################################################################################
#
# Task 4: Sample from the posterior.
#
####################################################################################
def construct_samples(a,b,P,N_samples):
	''' This function constructs samples from a multivariate normal distribution
	    fitted to the data.

		Parameters:
		-----------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.

		Returns:
		--------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# **to do**
	# 1. FIGURE OUT how to use the multivariate normal functionality in numpy
	#    to generate parameter samples
	# 2. ANSWER the questions in the lab document

	# compute properties (fitting) of multivariate normal distribution
	# mean = a vector of parameter means
	# covariance = a matrix of parameter variances and correlations
	A, B = np.meshgrid(a,b,indexing='ij')
	mean, covariance = fit_mvn([A,B], P)

	# 1. create samples using numpy function multivariate_normal (Google it)
	#samples=
	samples = np.random.multivariate_normal(mean, covariance, N_samples)

	# plot samples and predictions
	plot_samples(a, b, P=P, samples=samples)

	return samples
	
####################################################################################
#
# Task 5: Make predictions for your samples.
#
####################################################################################
def model_ensemble(samples):
	''' Runs the model for given parameter samples and plots the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# **to do**
	# Run your parameter samples through the model and plot the predictions.

	# 1. choose a time vector to evaluate your model between 1953 and 2012 
	# t =
	t = np.linspace(2009, 2019, 101)

	# 2. create a figure and axes (see TASK 1)
	#f,ax =
	f,ax = plt.subplots(1,1)

	# 3. for each sample, solve and plot the model  (see TASK 1)
	for a,b in samples:
		#pm=
		#ax.plot(
		#*hint* use lw= and alpha= to set linewidth and transparency
		t, pm = solve_lpm(lpm, 2009, 2019, 0.1, 25.16, [10.87, a, b])
		ax.plot(t,pm,'k-', lw=0.25,alpha=0.2)
	ax.plot([],[],'k-', lw=0.5,alpha=0.4, label='model ensemble')

	# get the data
	tp,po = np.genfromtxt('gs_pres.txt', delimiter = ',', skip_header = 1).T
	ax.axvline(2009, color='b', linestyle=':', label='calibration/forecast')
	
	# 4. plot Wairakei data as error bars
	# *hint* see TASK 1 for appropriate plotting commands
	v = 0.05
	ax.errorbar(tp,po,yerr=v,fmt='ro', label='data')
	ax.set_xlabel('time')
	ax.set_ylabel('pressure')
	ax.legend()
	plt.show()

if __name__=="__main__":
	# Comment/uncomment each of the functions below as you complete the tasks
	
	# TASK 1: Read the instructions in the function definition.
	#get_familiar_with_model()
	
	# TASK 2: Read the instructions in the function definition.
	a,b,posterior = grid_search()
	
	# TASK 3: Open the file fun_with_multivariate_normals.py and complete the tasks.

	# TASK 4: Read the instructions in the function definition.
	# this task relies on the output of TASK 2, so don't comment that command
	N = 100
	samples = construct_samples(a, b, posterior, N)

	# TASK 5: Read the instructions in the function definition.
	# this task relies on the output of TASKS 2 and 3, so don't comment those commands
	model_ensemble(samples)


